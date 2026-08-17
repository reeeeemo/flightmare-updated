import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from rpg_baselines.common.create_gate_environment import create_gates
from rpg_baselines.common.vision_stack import VisionStack
from collections import deque
import cv2
from ultralytics import YOLO
import torch

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
            max_memory_space: int = 10000,
            use_cam: bool = False,
            vision_weights: str = "",
            phase: int = 1,
            init_gate_num: int = 1,
            target_gate_num: int = 14,
            crash_det: bool = False,
            is_rendering: bool = False,
            pad_launch: bool = False,
         ):
        if phase not in [1, 2, 3]:
            raise ValueError("Cannot train on unknown phase.")
        self.wrapper = impl
        
        # ------------------------------------
        # PPO OBS / ACTION VARIABLES
        # ------------------------------------
        self.num_drone_obs = self.wrapper.getObsDim()
        self.num_full_obs = self.num_drone_obs + 22
        self.num_acts = self.wrapper.getActDim()
        self.max_episode_steps = 2400 if phase in (1,2) else 4800
        print(f"[OBSERVATION STATE DIM]: {self.num_drone_obs}")
        print(f"[ACTION STATE DIM]: {self.num_acts}")

        # [x_to_near_gate, y_to_near_gate, z_to_near_gate]
        # 9D rot mat for [yaw, pitch, roll]
        # [x_vel, y_vel, z_vel]
        # [roll, pitch, yaw vel]
        # [previous_thrust, prev_pitch, prev_yaw, prev_roll]
        # [gate_corner_x, gate_corner_y, gate_corner_z] x 4
        # [x_to_2nd_gate_rotmat, y_to_2nd_gate_rotmat, z_to_2nd_gate_rotmat]
        self._observation_space = spaces.Box(
            np.ones(self.num_full_obs) * -np.inf,
            np.ones(self.num_full_obs) * np.inf, dtype=np.float32)
        
        # [collective thrust, roll, pitch, yaw] rates
        self._action_space = spaces.Box(
            low=np.ones(self.num_acts) * -1.,
            high=np.ones(self.num_acts) * 1.,
            dtype=np.float32)
        
        # for seperate normalization -- ensure vecnormalize does not use indices 15, 16, 17.
        self.omega_max = [15.0, 15.0, 6.0]
        
        # domain randomization
        self.is_rendering = is_rendering
        self.pad_launch = pad_launch
        self.wrapper.setPadLaunch(pad_launch)
        
        # ------------------------------------
        # PPO REWARD VARIABLES
        # ------------------------------------
        # numpy array of all drone obs, rews since multi environments
        self._drone_obs = np.zeros([self.num_envs, self.num_drone_obs], dtype=np.float32)
        self._full_obs = np.zeros([self.num_envs, self.num_full_obs], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool_)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.gate_crossed = False
        
        # extra info for individual environments if needed
        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self._extraInfo = np.zeros([self.num_envs,
                                    len(self._extraInfoNames)], dtype=np.float32)
        
        # vars to hold gate positions for drone if no dets
        self._seen_cur_gate = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._seen_xyz_corners = np.zeros((self.num_envs, 12), dtype=np.float32)
        
        # vars to accumulate drift
        self._cur_gate_drift = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._cur_xyz_drift = np.zeros((self.num_envs, 12), dtype=np.float32)
        
        # ------------------------------------
        # REWARD COEFFICIENTS
        # ------------------------------------
        self.lin_vel_coef = 1
        self.ang_vel_coef = -0.001
        self.act_coef = -0.01
        self.offset_coef = 0
        self.perception_coef = -0.01 if phase != 3 else -0.1
        self.gate_bonus = 10 if phase != 3 else 30

        # ------------------------------------
        # GATE VARIABLES
        # ------------------------------------
        # half of inner width/height + account for drone size scoring
        self.half_w = 0.75
        self.half_h = 0.75
        self.score_half = 0.61
        self.gate_depth = 0.26  # depth of gate
        self.v_max = 99
        self.sim_dt = 0.00833333333  # 120hz
        self.gates = np.zeros((0, 3), dtype=np.float32)
        self.cur_gate = np.zeros(self.num_envs, dtype=int)
        self._prev_gate_dir = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.flat_probability = [0.8, 0.6, 0.4][phase-1]
        
        # ---------- FLOOR / WALL BORDERS BASED ON GATE POS/ROTS ----------
        self._lowest_z = 2.0 + (2.0*phase)
        self.wall_xmin = 0.0
        self.wall_xmax = 0.0

        # ------------------------------------
        # CAMERA VARIABLES
        # ------------------------------------
        self.camera_penalty = np.zeros((self.num_envs), dtype=np.float32)
        self.use_cam = use_cam
        self.vision_stack = VisionStack(self.num_envs, vision_weights)

        self.object_points = np.array([  # inner depth: 0.75x0.75
            [-0.75, 0.75, 0], # top left
            [0.75, 0.75, 0], # top right
            [0.75, -0.75, 0], # bottom right
            [-0.75, -0.75, 0] # bottom left
        ], dtype=np.float32)

        # fx = 320 = fy since square pixels
        self.camera_matrix = np.array([
            [180, 0, 320],
            [0, 180, 180],
            [0, 0, 1]
        ], dtype=np.float32)

        # 20 degree tilt upwards
        theta = np.radians(20)
        self.R_body_cam = np.array([
            [1, 0, 0],
            [0, np.sin(theta), np.cos(theta)],
            [0, -np.cos(theta), np.sin(theta)],
        ])

        # ------------------------------------
        # ENVIRONMENT VARIABLES
        # ------------------------------------
        self.rot_mats = np.zeros((0, 3, 3), dtype=np.float32)
        self.drone_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        self._prev_action = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)
        self.crash_det = crash_det

        # ------------------------------------
        # CURRICULUM LEARNING VARIABLES
        # ------------------------------------
        self.ep_successes = deque(maxlen=max_memory_space)
        self.n_gates = init_gate_num
        self.start_gate = init_gate_num
        self.n_gates_target = target_gate_num
        self.p_target = 0.85  # 85% per gate reliability has to be hit
        self.phase = phase
        self.injection_rate = 0.75
        self.training_seeds = [1, 20, 40, 0, 3, 6, 9, 7]

        # ------------------------------------
        # VISION INFERENCE VARS
        # ------------------------------------
        self._last_imgs = np.zeros((self.num_envs, 640, 360, 3), dtype=np.float32)
        self.filtered_gate = [None for _ in range(self.num_envs)]
        self.filtered_corners = [None for _ in range(self.num_envs)]
        
        # ------------------------------------
        # DATA ANALYSIS VARIABLES
        # ------------------------------------
        self.gate_hits = np.full((self.num_envs, self.n_gates_target, 3), np.nan, dtype=np.float32)
        self.gate_crosses = np.full((self.num_envs, self.n_gates_target, 3), np.nan, dtype=np.float32)

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def _compute_reward(self, action: np.ndarray):
        """Computes reward of drone in environment.
        
        Args:
            action: [normalized_thrust, roll, pitch, yaw] rates
        Returns:
            reward based on drone observation state and current action
        """
        ep_lens = np.array([len(self.rewards[i]) for i in range(self.num_envs)], dtype=np.int16)

        # ------------------------------------
        # INSTA-FAIL IF PAD LAUNCHING AND DOWNWARD FIRST CMD
        # ------------------------------------
        if self.pad_launch and not self.vision_stack.is_enabled():
            first_step = np.array([len(self.rewards[i]) == 0 for i in range(self.num_envs)])
            sink = first_step & (action[:, 0] < 0)
            self._reward[sink] -= 10
            self._done[sink] = True
        
        # ------------------------------------
        # INSTA-FAIL IF DRONE HITS WALL BOUNDARIES
        # ------------------------------------
        out = ((self.drone_pos[:, 0] < self.wall_xmin) | (self.drone_pos[:, 0] > self.wall_xmax)) & (~self._done)
        self._reward[out] -= 10
        self._done[out] = True
        
        # ------------------------------------
        # GATE INDICES 
        # ------------------------------------
        prev_gate_idx = np.maximum(self.cur_gate-1, 0).clip(max=len(self.gates)-1)
        cur_gate_idx = self.cur_gate.clip(max=len(self.gates)-1)
        
        # ------------------------------------
        # GATE DIRECTION, NORMAL, VELOCITY
        # ------------------------------------
        gate_dir = self.gt_gate_dir
        ang_vel = self._drone_obs[:, 9:12].copy()
        gate_dist = np.linalg.norm(gate_dir, axis=1, keepdims=True).clip(min=1e-6)
        gate_dir_norm = gate_dir / gate_dist

        # ------------------------------------
        # COMPUTE CAMERA DEVIATION ABOVE 60 DEGREES
        # ------------------------------------
        # cosine of norm gate dir and forward-facing camera vec
        forward_axis = self._full_obs[:, 3:12].reshape(self.num_envs, 3, 3)[:, :, 1]
        up_axis = self._full_obs[:, 3:12].reshape(self.num_envs, 3, 3)[:, :, 2]
        cam_forward = forward_axis * np.cos(np.radians(20)) + up_axis * np.sin(np.radians(20))
        camera_dev = np.sum(gate_dir_norm * cam_forward, axis=1)
        self.camera_penalty = np.maximum(0.0, np.cos(np.radians(60)) - camera_dev)
        self.camera_penalty[gate_dist.squeeze() < 2.0] = 0
        
        # ------------------------------------
        # COMPUTE DRONE-TO-GATE PROGRESS + EXCESS CHANGE REWARDS
        # ------------------------------------
        # caps at max velocity allowed each timestep
        prev_dist = np.linalg.norm(self._prev_gate_dir, axis=1)
        progress = np.clip(prev_dist - gate_dist.squeeze(), -self.v_max * self.sim_dt, self.v_max * self.sim_dt)
        # prevent instantaneous switching of motors to high/low rpms ("bang bang" motion)
        excess_change = np.maximum(0.0, np.abs(self._prev_action - action) - 0.5) # instead of 0.3
        
        # ------------------------------------
        # STEP REWARD CALCULATION
        # ------------------------------------
        step_rew = (
            self.lin_vel_coef * progress +
            self.ang_vel_coef * np.sum(ang_vel**2, axis=1) +  # small penalty towards unstable angular vel
            self.act_coef * np.sum(excess_change, axis=1) +
            self.perception_coef * self.camera_penalty # penalty for not being in the orientation of the gate
        ).astype(np.float32)

        self._reward = np.where(self._done, self._reward, step_rew)
        self._prev_gate_dir = gate_dir.copy()

        # ------------------------------------
        # UPDATE CUR GATE AND REWARD/PUNISH CLOSENESS TO GATE
        # ------------------------------------
        # get coordinates in gate local space:
        # idx 0 = left/right offset from gate center
        # idx 1 = distance along the approach axis (forward/backward)
        # idx 2 = up/down offset from gate center
        gate_vecs_all = self.gates[None, :, :] - self.drone_pos[:, None, :]
        local_all = np.einsum('gji,ngj->ngi', self.rot_mats, gate_vecs_all)  # rotmat_gate.T @ gate_dir
        on_plane_all = np.abs(local_all[:, :, 1]) < self.gate_depth
        in_opening_all = (
            (np.abs(local_all[:, :, 0]) < self.score_half) &
            (np.abs(local_all[:, :, 2]) < self.score_half)
        )
        
        # --------------------
        # FIND WHETHER DRONE HIT FRAME AND REWARD ACCORDINGLY 
        # --------------------
        frame_hit = on_plane_all & (~in_opening_all)
        n_hits = frame_hit.sum(axis=1)
        active = (~self._done) & (self.cur_gate < len(self.gates))
        if self.crash_det:
            self._done[active & (n_hits > 0)] = True
        else:
            self._reward[active] += (-0.5 * n_hits[active]).astype(np.float32)
            hit = active & (n_hits > 0)
            self.gate_hits[hit, self.cur_gate[hit]] = self.drone_pos[hit]
            
        # --------------------
        # IF WENT THROUGH FRAME
        # --------------------
        env_idx = np.arange(self.num_envs)
        cur = self.cur_gate.clip(max=len(self.gates) - 1)
        local_cur = local_all[env_idx, cur]
        
        entry_plane = (local_cur[:, 1] >= 0) & (local_cur[:, 1] < self.gate_depth)
        in_opening_cur = (np.abs(local_cur[:, 0]) < self.score_half) & (np.abs(local_cur[:, 2]) < self.score_half)
        threaded = active & entry_plane & in_opening_cur
        
        offset_bonus = self.offset_coef * (1.0 - (local_cur[:, 0]/self.half_w)**2 - (local_cur[:,2]/self.half_h)**2)
        self._reward[threaded] += self.gate_bonus + offset_bonus[threaded]
        self.gate_crosses[threaded, self.cur_gate[threaded]] = self.drone_pos[threaded]
        self.cur_gate[threaded] += 1
        
        # if not at end, set prev dir and re-update obs since it will be stale
        not_end = threaded & (self.cur_gate < len(self.gates))
        self._prev_gate_dir[not_end] = self.gates[self.cur_gate[not_end]] - self.drone_pos[not_end]
        if not self.vision_stack.is_enabled() and threaded.any():
            self.gate_crossed = True
        else:
            self.vision_stack.set_prev_gate(not_end, self._full_obs[not_end, 0:3])
            self.vision_stack.reset_cur_gate(not_end)
        
        # --------------------
        # SET DONE FLAG + GIVE TIME-BASED BONUS IF COMPLETION
        # --------------------
        finished = self.cur_gate >= len(self.gates)
        self._done[finished] = True
        comp_bonus = 50 + 25 * (1.0 - ep_lens / self.max_episode_steps)
        self._reward[finished] += comp_bonus[finished]
        
        # --------------------
        # GIVE PENALTY IF UNCOMPLETE COURSE, TIMEOUT
        # --------------------
        # rudimentary way for timeout since it gives penalty a timestep before end, but C++ will take over if not
        # also drone should be flying under the time limit anyways
        timed_out = (
            (~self._done & (ep_lens >= self.max_episode_steps - 1)) | 
            (self._done & (self.cur_gate < len(self.gates))))
        self._done[timed_out] = True
        self._reward[timed_out] -= 10

    def _update_observation(self, action: np.ndarray):
        """Updates observations recieved from C++ wrapper."""
        self.drone_pos = self._drone_obs[:, 0:3].copy()
        
        # update to relative pos between gate and drone
        cur_gate_idx = self.cur_gate.clip(max=len(self.gates)-1)
        self.gt_gate_dir = self.gates[cur_gate_idx] - self.drone_pos
        
        # move other observations
        self._full_obs[:, 12:15] = self._drone_obs[:, 6:9].copy()
        self._full_obs[:, 15:18] = self._drone_obs[:, 9:12].copy() / self.omega_max

        # update angles to 9d rotation mat
        # see https://arxiv.org/pdf/2509.17274 section III and IV for full details
        self._full_obs[:, 3:12] = self.convert_euler_to_rot_mat(self._drone_obs[:, 3:6].copy())

        # encase previous action
        self._full_obs[:, 18:22] = action

        # get current gates (x,y,z) for all 4 corners
        center = self.gates[cur_gate_idx]
        right = self.rot_mats[cur_gate_idx, :, 0] * self.half_w
        up = self.rot_mats[cur_gate_idx, :, 2] * self.half_h
        
        # get images from camera if using
        if self.use_cam:
            self._last_imgs = self.wrapper.getRGBImage()
            frames = np.array(self._last_imgs, dtype=np.uint8)

        if not self.vision_stack.is_enabled():
            ### noise training if requested else priviledged learning ###
            add_noise = np.random.uniform(0, 1) < self.injection_rate
            noise = np.zeros(12)
            if self.phase == 3 and add_noise:
                noise = np.random.uniform(-0.5, 0.5, size=(self.num_envs, 12))

            self._full_obs[:, 22:34] = np.concatenate([
                (center + right + up) - self.drone_pos, # top right
                (center - right + up) - self.drone_pos, # top left
                (center + right - up) - self.drone_pos, # bottom right
                (center - right - up) - self.drone_pos  # bottom left
            ], axis=1) + noise
            
            # add noise to gate distance if noise training
            if self.phase != 3 or not add_noise:
                self._full_obs[:, 0:3] = self.gates[cur_gate_idx] - self.drone_pos
            else:
                tmp_gate_dir = self._full_obs[:, 22:34].reshape(self.num_envs, 4, 3).mean(axis=1)
                far_enough = np.linalg.norm(tmp_gate_dir, axis=1) > 2
                self._full_obs[far_enough, 0:3] = tmp_gate_dir[far_enough]

            # ------------------------------------
            # CAMERA DEVIATION -- FOV HOLD // 30 HZ FROZEN
            # ------------------------------------
            if self.phase >= 2:
                steps = np.array([len(self.rewards[i]) for i in range(self.num_envs)])
                fresh = (steps % 4 == 0)  # 120 / 30 = every 4 timesteps fresh vals

                has_deviated = (self.camera_penalty > 0) | ~(fresh)
                self._cur_gate_drift[~has_deviated] = 0
                self._cur_xyz_drift[~has_deviated] = 0
                
                R = self._full_obs[:, 3:12].reshape(self.num_envs, 3,3)
                
                # set if not deviating
                self._seen_cur_gate[~has_deviated] = self._full_obs[~has_deviated, 0:3]
                self._seen_xyz_corners[~has_deviated] = self._full_obs[~has_deviated, 22:34]
                # if deviated, dead reckon via imu estimated velocity
                self._cur_xyz_drift[has_deviated] -= np.tile(self._full_obs[has_deviated, 12:15], 4) * self.sim_dt
                self._cur_gate_drift[has_deviated] -= self._full_obs[has_deviated, 12:15] * self.sim_dt
                
                self._full_obs[has_deviated, 22:34] = self._seen_xyz_corners[has_deviated] + self._cur_xyz_drift[has_deviated]
                self._full_obs[has_deviated, 0:3] = self._seen_cur_gate[has_deviated] + self._cur_gate_drift[has_deviated]

        elif frames.size != 0:
            
            # propagate prev gate that was crossed by velocity.
            ( n_filtered_corners, n_filtered_xyzs) = self.vision_stack.get_gate(
                frames = frames, 
                rotation = self._full_obs[:, 3:12], 
                velocity = self._full_obs[:, 12:15],
                sim_dt = self.sim_dt,
                cur_gate = self._full_obs[:, 0:3],
                cur_corners = self._full_obs[:, 22:34]
            )
            
            self._full_obs[:, 0:3] = n_filtered_xyzs
            self._full_obs[:, 22:34] = n_filtered_corners


    def step(self, action: np.ndarray):
        """Computes step of drone in environment.

        Args:
            action: [normalized thrust, pitch, roll, yaw] rates
        Returns:
            observation, reward, done, env information
        """
        # pitch: (pos, up), roll: (pos, right), yaw: (pos, left)
        
        # ------------------------------------
        # OBS / REWARD RECIEVED 
        # ------------------------------------
        self.wrapper.step(action, self._drone_obs,
                          self._reward, self._done, self._extraInfo) # step in c++ env
        self._update_observation(action)
        self._compute_reward(action)
        if self.gate_crossed:
            self.gate_crossed = False
            self._update_observation(action)
        
        self._prev_action = action.copy()
        
        # ------------------------------------
        # ADDITIONAL INFO UPDATE FOR ALL ENVS
        # ------------------------------------
        if len(self._extraInfoNames) != 0:
            info = [{'extra_info': {
                self._extraInfoNames[j]: self._extraInfo[i, j] for j in range(0, len(self._extraInfoNames))
            }} for i in range(self.num_envs)]
        else:
            info = [{} for i in range(self.num_envs)]

        # ------------------------------------
        # UPDATE REWARD / IS DONE INFO
        # ------------------------------------
        for i in range(self.num_envs):
            # update memory to know whether drone crashes (-1 penalty) or not
            self.rewards[i].append(self._reward[i])
            if self._done[i]:
                eplen = len(self.rewards[i])
                eprew = sum(self.rewards[i])
                self.ep_successes.append(self.cur_gate[i] >= len(self.gates))
                epinfo = {
                    "r": eprew, "l": eplen,
                    "gate_hits": self.gate_hits[i].copy(),
                    "gate_crosses": self.gate_crosses[i].copy()
                }
                info[i]['episode'] = epinfo
                self.rewards[i].clear()

        # ------------------------------------
        # RESET COMMANDS FOR MULTI-ENVS
        # ------------------------------------
        #n = self._done.sum()
        self._prev_action[self._done] = 0
        self.cur_gate[self._done] = 0
        self.gate_hits[self._done] = np.nan
        self.gate_crosses[self._done] = np.nan
        if self.phase >= 2:
            self._seen_cur_gate[self._done] = 0
            self._seen_xyz_corners[self._done] = 0
            self._cur_gate_drift[self._done] = 0
            self._cur_xyz_drift[self._done] = 0

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

        # spawns anywhere random from 0-1
        # lin velocity is randomized from 0-1 too
        self.wrapper.reset(self._drone_obs, (self.phase == 3 and not self.is_rendering))
        
        self._prev_gate_dir = self.gates[self.cur_gate] - self._drone_obs[:, 0:3]
        
        self._update_observation(self._prev_action)
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
        
        # compute min and max wall distances
        self.wall_xmin = min(self.gates[:, 0].min(), 0.0) - np.random.uniform(3.0, 6.0)
        self.wall_xmax = max(self.gates[:, 0].max(), 0.0) + np.random.uniform(3.0, 6.0)
        
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
        """Increase # of random gates if consistently successful during training.
        """
        if not self.ep_successes:
            return
        
        success_rate = sum(self.ep_successes) / len(self.ep_successes)
        
        # ----------
        # randomize gate position / number of gates dependent on success rate
        # ----------
        self.set_random_rotation_gate(success_rate)
    
    
    def set_random_rotation_gate(self, success_rate: float):
        """Set random position and rotations of gates.
        
        Set number of gates and increase if success rate is
        greater than course completion rate (probability_per_gate^n_gates).
        """
        # ------------------------------------
        # SUCCESS RATE / N_GATES MANAGER
        # ------------------------------------
        # anyhting under p^5 (>0.45) overfits since completion % is so high, causing std degradation
        # anything over p^5 (<0.45) struggles with small sample sizes, cannot hit 45% course completion
        # ^^ note that 45% course completion for p^11 = 0.93% per gate accuracy :p
        threshold = min(0.65, self.p_target**self.n_gates)
        if success_rate >= threshold:
            self.n_gates = min(self.n_gates+1, self.n_gates_target)
            self.ep_successes.clear()
        n_gates = self.n_gates
        
        # ------------------------------------
        # SEEDING
        # ------------------------------------
        chosen_seed = np.random.choice(self.training_seeds)
        saved_state = np.random.get_state()
        np.random.seed(chosen_seed)

        # ------------------------------------
        # POSITION / ROTATION RANDOMIZATION
        # ------------------------------------
        positions, rotations = create_gates(n_gates, self.phase)
            
        # ---------- C++ WRAPPER CALLBACKS ----------
        self.addGate(positions, rotations)
        np.random.set_state(saved_state)
        self.wrapper.setLowestZ(min(min(self.gates[:, 2]) - self._lowest_z, -1.0))
    
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