import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from collections import deque
import cv2

class QuadcopterGatesVec(VecEnv):
    """Custom Gymnasium environment that simulates a drone flying through gates

    Allows for multiple environments in native C++ threads.
    Also follows a curriculum learning strategy to increase difficulty.
    Attributes:
        wrapper: C++ wrapper that handles physics/simulation
        _observation: observation matrix for each env
        _reward: current reward for each environment
        rewards: reward for each environment at each timestep
        goal_xyz: goal XYZ position for drone to reach
        goal_rpy: goal RPY orientation for drone to reach
    """
    def __init__(
            self, 
            impl, 
            max_memory_space: int = 200,
            use_cam: bool = False,
            training: bool = True
         ):
        self.wrapper = impl

        # get observations and actions # from wrapper and add others
        self.num_drone_obs = self.wrapper.getObsDim()
        self.num_full_obs = self.num_drone_obs + 22 
        self.num_acts = self.wrapper.getActDim()
        self.max_episode_steps = 1200
        print(f"[OBSERVATION STATE DIM]: {self.num_drone_obs}")
        print(f"[ACTION STATE DIM]: {self.num_acts}")

        # [x_to_near_gate, y_to_near_gate, z_to_near_gate]
        # 9D rot mat for [yaw, pitch, roll]
        # [x_vel, y_vel, z_vel]
        # [roll, pitch, yaw vel]
        # [previous_thrust, prev_pitch, prev_yaw, prev_roll]
        # [gate_corner_x, gate_corner_y, gate_corner_z] x 4
        self._observation_space = spaces.Box(
            np.ones(self.num_full_obs) * -np.inf,
            np.ones(self.num_full_obs) * np.inf, dtype=np.float32)
        
        # [collective thrust, roll, pitch, yaw] rates
        self._action_space = spaces.Box(
            low=np.ones(self.num_acts) * -1.,
            high=np.ones(self.num_acts) * 1.,
            dtype=np.float32)
        
        # numpy array of all drone obs, rews since multi environments
        self._drone_obs = np.zeros([self.num_envs, self.num_drone_obs], dtype=np.float32)
        self._full_obs = np.zeros([self.num_envs, self.num_full_obs], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool_)
        self.rewards = [[] for _ in range(self.num_envs)]
        
        # extra info for individual environments if needed
        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self._extraInfo = np.zeros([self.num_envs,
                                    len(self._extraInfoNames)], dtype=np.float32)

        # reward coefficients
        self.lin_vel_coef = 2
        self.ang_vel_coef = -0.002
        self.act_coef = -0.10
        self.offset_coef = 2
        self.perception_coef = -0.1

        # gate metrics (unity model is 100x100x100, so 1mx1mx1m)
        self.half_w = 1.5  # half width of gate (real is 0.5)
        self.half_h = 1.5  # half height of gate (real is 0.5)
        self.gate_depth = 0.5  # depth of gate
        self.v_max = 99
        self.sim_dt = 0.00833333333

        # camera vars
        self.use_cam = use_cam

        # goal gates
        self.gates = np.zeros((0, 3), dtype=np.float32)
        self.rot_mats = np.zeros((0, 3, 3), dtype=np.float32)
        self.cur_gate = np.zeros(self.num_envs, dtype=int)
        self.drone_pos = np.zeros((self.num_envs, 3), dtype=np.float32)

        # curriculum learning vars
        self.ep_successes = deque(maxlen=max_memory_space)
        self.randomize_gates = False
        self.training = training

        self._prev_action = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)
        self._last_imgs = np.zeros((self.num_envs, 320, 320, 3), dtype=np.float32)
        self._prev_gate_dir = np.zeros((self.num_envs, 3), dtype=np.float32)

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def _compute_reward(self, action: np.ndarray):
        """Computes reward of drone in environment.
        
        Args:
            action: [normalized_thrust, roll, pitch, yaw] rates
        Returns:
            reward based on drone observation state and current action
        """
        
        # compute action-based rewards
        gate_dir = self._full_obs[:, 0:3]
        ang_vel = self._full_obs[:, 15:18]

        # compute normalized gate direction then velocity twrds gate and gate normal
        gate_dist = np.linalg.norm(gate_dir, axis=1, keepdims=True).clip(min=1e-6)
        gate_dir_norm = gate_dir / gate_dist

        # compute cosine of norm gate dir and forward-facing camera vec
        # only penalize if above 60 degrees
        forward_axis = self._full_obs[:, 3:12].reshape(self.num_envs, 3, 3)[:, :, 1]
        up_axis = self._full_obs[:, 3:12].reshape(self.num_envs, 3, 3)[:, :, 2]
        cam_forward = forward_axis * np.cos(np.pi/4) + up_axis * np.sin(np.pi/4)
        camera_dev = np.sum(gate_dir_norm * cam_forward, axis=1)
        camera_penalty = np.maximum(0.0, 0.5 - camera_dev)

        # how far did we move from starting value to new val (speed + position)
        prev_dist = np.linalg.norm(self._prev_gate_dir, axis=1)
        progress = np.clip(prev_dist - gate_dist.squeeze(), -self.v_max * self.sim_dt, self.v_max * self.sim_dt)

        # prevent instantaneous switching of motors to high/low rpms ("bang bang" motion)
        excess_change = np.maximum(0.0, np.abs(self._prev_action - action) - 0.3)

        step_rew = (
            self.lin_vel_coef * progress +
            self.ang_vel_coef * np.sum(ang_vel**2, axis=1) +  # small penalty towards unstable angular vel
            self.act_coef * np.sum(excess_change, axis=1) +
            self.perception_coef * camera_penalty # penalty for not being in the orientation of the gate
        ).astype(np.float32)

        self._reward = np.where(self._done, self._reward, step_rew)
        self._prev_gate_dir = gate_dir.copy()

        # update current gate selection + give reward if position is close
        for i in range(self.num_envs):
            if self.cur_gate[i] >= len(self.gates):
                continue
            # get coordinates in gate local space:
            # idx 0 = left/right offset from gate center
            # idx 1 = distance along the approach axis (forward/backward)
            # idx 2 = up/down offset from gate center
            local_positions = self.rot_mats[self.cur_gate[i]].T @ self._full_obs[i, 0:3]

            ### find whether drone has gone through or hit gate
            on_plane = abs(local_positions[1]) < self.gate_depth
            in_opening = (
                abs(local_positions[0]) < self.half_w 
                and abs(local_positions[2]) < self.half_h
            )
            if on_plane and not in_opening:
                self._reward[i] -= 2
                #self._done[i] = True
            elif on_plane and in_opening:
                self._reward[i] += 30  # old 30
                self._reward[i] -= self.offset_coef * (local_positions[0]**2 + local_positions[2]**2)
                self.cur_gate[i] += 1
                if self.cur_gate[i] < len(self.gates):
                    self._prev_gate_dir[i] = self.gates[self.cur_gate[i]] - self.drone_pos[i]
            
            if self.cur_gate[i] >= len(self.gates):
                self._done[i] = True
            else:
                self._full_obs[i, 0:3] = self.gates[self.cur_gate[i]] - self.drone_pos[i]


            # if done, give a time-based bonus
            if self._done[i] and self.cur_gate[i] >= len(self.gates):
                self._reward[i] += 50 + 25 * (1.0 - len(self.rewards[i]) / self.max_episode_steps)  # old was 50
            # if done and drone did not go thru all gates, give penalty
            elif self._done[i] and self.cur_gate[i] < len(self.gates):
                self._reward[i] -= 50
    

    def _update_observation(self):
        """Updates observations recieved from C++ wrapper."""
        # update to relative pos between gate and drone
        self.drone_pos = self._drone_obs[:, 0:3].copy()
        cur_gate_idx = self.cur_gate.clip(max=len(self.gates)-1)
        self._full_obs[:, 0:3] = self.gates[cur_gate_idx] - self.drone_pos

        # move other observations
        self._full_obs[:, 12:15] = self._drone_obs[:, 6:9].copy()
        self._full_obs[:, 15:18] = self._drone_obs[:, 9:12].copy()

        # update angles to 9d rotation mat
        # see https://arxiv.org/pdf/2509.17274 section III and IV for full details
        self._full_obs[:, 3:12] = self.convert_euler_to_rot_mat(self._drone_obs[:, 3:6].copy())

        # encase previous action
        self._full_obs[:, 18:22] = self._prev_action

        # get current gates (x,y,z) for all 4 corners
        center = self.gates[cur_gate_idx]
        right = self.rot_mats[cur_gate_idx, :, 0] * self.half_w
        up = self.rot_mats[cur_gate_idx, :, 2] * self.half_h

        if not self.use_cam:  # use priviledged learning
            self._full_obs[:, 22:34] = np.concatenate([
                (center + right + up) - self.drone_pos, # top right
                (center - right + up) - self.drone_pos, # top left
                (center + right - up) - self.drone_pos, # bottom right
                (center - right - up) - self.drone_pos  # bottom left
            ], axis=1)
        else:  # use onboard camera
            self._last_imgs = self.wrapper.getRGBImage()
            # TODO: get gate xyz pose from a model and push it to the full obs
            self._full_obs[:, 22:34] = np.concatenate([
                (center + right + up) - self.drone_pos, # top right
                (center - right + up) - self.drone_pos, # top left
                (center + right - up) - self.drone_pos, # bottom right
                (center - right - up) - self.drone_pos  # bottom left
            ], axis=1)


    def step(self, action: np.ndarray):
        """Computes step of drone in environment.

        Args:
            action: [normalized thrust, roll, pitch, yaw] rates
        Returns:
            observation, reward, done, env information
        """
        # values are clamped in c++ code
        self.wrapper.step(action, self._drone_obs,
                          self._reward, self._done, self._extraInfo)
        self._update_observation()
        self._prev_action = action.copy()

        # compute reward and if -1 dont adjust, but if 0 we update reward
        self._compute_reward(action)

        # update environments with additional info
        if len(self._extraInfoNames) != 0:
            info = [{'extra_info': {
                self._extraInfoNames[j]: self._extraInfo[i, j] for j in range(0, len(self._extraInfoNames))
            }} for i in range(self.num_envs)]
        else:
            info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            # update reward information if environment is finished
            # update memory to know whether drone crashes (-1 penalty) or not
            self.rewards[i].append(self._reward[i])
            if self._done[i]:
                eplen = len(self.rewards[i])
                eprew = sum(self.rewards[i])
                self.ep_successes.append(self.cur_gate[i] >= len(self.gates))
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                self.rewards[i].clear()

        self._prev_action[self._done] = 0

        return self._full_obs.copy(), self._reward.copy(), \
            self._done.copy(), info.copy()

    def stepUnity(self, action, send_id):
        """Call a step in unity if wrapper is attached.

        Args:
            action: [normalized thrust, roll, pitch, yaw] rates
            send_id: ID for unity enivonrment
        """
        receive_id = self.wrapper.stepUnity(action, self._drone_obs,
                                            self._reward, self._done, self._extraInfo, send_id)

        return receive_id

    def sample_actions(self):
        """Sample an example action from the action space.

        Returns:
            action: [normalized thrust, roll, pitch, yaw] rates
        """
        actions = []
        for _ in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float32)

    def reset(self):
        """Resets drone environment."""
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.cur_gate = np.zeros(self.num_envs, dtype=int)
        self._prev_action[:] = 0

        # randomize course if doing well enough
        if self.randomize_gates:
            self.modifyResetPosition(np.array([-5, 20, -5, 23, 5, 14], dtype=np.float32))
        else:
            self.modifyResetPosition(np.array([0, 1, 0, 1, 5, 6], dtype=np.float32))

        # start each drone at a random x, y, z
        # else it spawns anywhere random from 0-1
        # lin velocity is randomized from 0-1 too
        self.wrapper.reset(self._drone_obs)

        # select closest gate to drones starting point to use
        if self.training:
            for i in range(self.num_envs):
                best_score = float("inf")
                for j in range(len(self.gates)):
                    dist = np.linalg.norm(self.gates[j] - self._drone_obs[i, 0:3])
                    if dist < best_score:
                        best_score = dist
                        self.cur_gate[i] = j

        self._update_observation()
        self._prev_gate_dir = np.zeros((self.num_envs, 3), dtype=np.float32)
        return self._full_obs.copy()

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()
        return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def connectUnity(self):
        self.wrapper.connectUnity()

    def addGate(self, positions: np.ndarray, rotations: np.ndarray):
        """Adds a static gate to each drone environment.

        Adds to recurring memory of gates for simulation training.
        Note that this MUST be called BEFORE `.connectUnity()`, or else
        gate will not appear.
        Args:
            positions: matrix of [X,Y,Z] coordinates for each gate.
        """
        self.wrapper.addGate(positions, rotations)
        self.gates = positions.copy()
        self.rot_mats = self.convert_quat_to_rot_mat(rotations.copy())

    def modifyResetPosition(self, positions: np.ndarray):
        """Modify min/max x,y,z positions for drone to initialize from.
        
        Args:
            positions: list of [MIN_X, MAX_X, MIN_Y, MAX_Y, MIN_Z, MAX_Z] coords
        """
        self.wrapper.modifyResetPositions(positions)

    def disconnectUnity(self):
        self.wrapper.disconnectUnity()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def render_mode(self):
        return None
    
    @property
    def rgb_image(self):
        return self._last_imgs

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    def start_recording_video(self, file_name):
        raise RuntimeError('This method is not implemented')

    def stop_recording_video(self):
        raise RuntimeError('This method is not implemented')

    def curriculum_callback(self):
        """Increase difficulty of gate problem if consistently successful.
        """
        if not self.ep_successes:
            return
        
        success_rate = sum(self.ep_successes) / len(self.ep_successes)
        if not self.randomize_gates and success_rate >= 0.9:
            self.randomize_gates = True
            self.ep_successes.clear()
        elif self.randomize_gates and success_rate < 0.6:
            self.randomize_gates = False
    
    
    def convert_euler_to_rot_mat(self, euler: np.ndarray):
        """Converts a batch of euler angles to a rotation matrix."""
        # http://close-range.com/docs/Computing_Euler_angles_from_a_rotation_matrix.pdf (its reversed)
        y, p, r = euler[:, 0], euler[:, 1], euler[:, 2]
        cy, sy = np.cos(y), np.sin(y)
        cp, sp = np.cos(p), np.sin(p)
        cr, sr = np.cos(r), np.sin(r)

        c0 = np.stack([cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], axis=1)
        c1 = np.stack([sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], axis=1)
        c2 = np.stack([-sp, cp*sr, cp*cr], axis=1)

        return np.concatenate([c0, c1, c2], axis=1)

    def convert_quat_to_rot_mat(self, q: np.ndarray):
        """Converts a quaternion to a rotation matrix."""
        # https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        mats = np.zeros((len(w), 3, 3), dtype=np.float32)
        mats[:, 0, 0] = 2 * (w**2 + x**2) - 1
        mats[:, 0, 1] = 2 * (x*y - w*z)
        mats[:, 0, 2] = 2 * (x*z + w*y)
        mats[:, 1, 0] = 2 * (x*y + w*z)
        mats[:, 1, 1] = 2 * (w**2 + y**2) - 1
        mats[:, 1, 2] = 2 * (y*z - w*x)
        mats[:, 2, 0] = 2 * (x*z - w*y)
        mats[:, 2, 1] = 2 * (y*z + w*x)
        mats[:, 2, 2] = 2 * (w**2 + z**2) - 1
        return mats

    def step_async(self, actions):
        self._async_actions = actions

    def step_wait(self):
        return self.step(self._async_actions)

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment.

        Args:
            attr_name: (str) The name of the attribute whose value to return
            indices: (list,int) Indices of envs to get attribute from
        Returns:
            (list) List of values of 'attr_name' in all environments
        """
        num = len(indices) if indices is not None else self.num_envs
        if hasattr(self, attr_name):
            return [getattr(self, attr_name)] * num
        return [None] * num

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments.
        
        Args:
            attr_name: (str) The name of attribute to assign new value
            value: (obj) Value to assign to `attr_name`
            indices: (list,int) Indices of envs to assign value
        """
        raise RuntimeError('This method is not implemented')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments.

        Args:
            method_name: (str) The name of the environment method to invoke.
            indices: (list,int) Indices of envs whose method to call
            method_args: (tuple) Any positional arguments to provide in the call
            method_kwargs: (dict) Any keyword arguments to provide in the call
        Returns:
            (list) List of items returned by the environment's method call
        """
        raise RuntimeError('This method is not implemented')

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs