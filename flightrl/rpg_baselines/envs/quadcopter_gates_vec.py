import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from collections import deque
import random as rand

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
    def __init__(self, 
                 impl, 
                 max_memory_space: int = 200
         ):
        self.wrapper = impl
        self.num_obs = self.wrapper.getObsDim()
        self.num_acts = self.wrapper.getActDim()
        self.max_episode_steps = 300
        print(f"[OBSERVATION STATE DIM]: {self.num_obs}")
        print(f"[ACTION STATE DIM]: {self.num_acts}")

        # [x_to_near_gate, y_to_near_gate, z_to_near_gate]
        # [yaw, pitch, roll]
        # [x_vel, y_vel, z_vel]
        # [roll, pitch, yaw vel]
        self._observation_space = spaces.Box(
            np.ones(self.num_obs) * -np.inf,
            np.ones(self.num_obs) * np.inf, dtype=np.float32)
        
        # [collective thrust, roll, pitch, yaw] rates
        self._action_space = spaces.Box(
            low=np.ones(self.num_acts) * -1.,
            high=np.ones(self.num_acts) * 1.,
            dtype=np.float32)
        
        # numpy array of all drone obs, rews since multi environments
        self._observation = np.zeros([self.num_envs, self.num_obs],
                                     dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool_)
        self.rewards = [[] for _ in range(self.num_envs)]
        
        # extra info for individual environments if needed
        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self._extraInfo = np.zeros([self.num_envs,
                                    len(self._extraInfoNames)], dtype=np.float32)

        # reward coefficients
        self.pos_coef = -0.002
        self.lin_vel_coef = 0.04
        self.ang_vel_coef = -0.0001
        self.act_coef = -0.0002

        # goal gates
        self.gates = np.zeros((0, 3), dtype=np.float32)
        self.cur_gate = np.zeros(self.num_envs, dtype=int)
        self.drone_pos = np.zeros((self.num_envs, 3), dtype=np.float32)

        self.ep_successes = deque(maxlen=max_memory_space)
        self.randomize_gates = False


    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def _compute_reward(self, action: np.ndarray):
        """Computes reward of drone in environment.
        
        Args:
            action: [normalized_thrust, roll, pitch, yaw] rates
        Returns:
            reward based on drone observation state and current action
        """
        
        gate_dir = self._observation[:, 0:3]
        vel = self._observation[:, 6:9]
        ang_vel = self._observation[:, 9:12]

        # compute normalized gate direction then velocity twrds gate
        gate_dist = np.linalg.norm(gate_dir, axis=1, keepdims=True).clip(min=1e-6)
        vel_towards_gate = np.sum(vel * (gate_dir / gate_dist), axis=1)

        # pos_penalty = self.pos_coef * np.sum(gate_dir**2, axis=1) / (gate_dist.squeeze() + 1e-6)

        # position, orientation, linear and angular velocity
        # and penalty for having to use an action
        return (
            self.pos_coef * gate_dist.squeeze() + 
            self.lin_vel_coef * vel_towards_gate +
            self.ang_vel_coef * np.sum(ang_vel**2, axis=1) +
            self.act_coef * np.linalg.norm(action, axis=1)
        ).astype(np.float32)
        # 0.05 was old survival bonus

    def step(self, action: np.ndarray):
        """Computes step of drone in environment.

        Args:
            action: [normalized thrust, roll, pitch, yaw] rates
        Returns:
            observation, reward, done, env information
        """
        # values are clamped in c++ code
        self.wrapper.step(action, self._observation,
                          self._reward, self._done, self._extraInfo)
        # update observation to estimated gate position - estimated drone pos
        self.drone_pos = self._observation[:, 0:3].copy()
        self._observation[:, 0:3] = self.gates[self.cur_gate] - self.drone_pos

        # compute reward and if -1 dont adjust, but if 0 we update reward
        step_reward = self._compute_reward(action)
        self._reward = np.where(self._done, self._reward, step_reward)

        # update environments with additional info
        if len(self._extraInfoNames) != 0:
            info = [{'extra_info': {
                self._extraInfoNames[j]: self._extraInfo[i, j] for j in range(0, len(self._extraInfoNames))
            }} for i in range(self.num_envs)]
        else:
            info = [{} for i in range(self.num_envs)]


        for i in range(self.num_envs):
            # update current gate selection + give reward if position is close
            dist = np.linalg.norm(self._observation[i, 0:3])
            if dist < 1.5:
                if self._reward[i] != -1:  # or whatever the crash reward is
                    self._reward[i] += 2.0
                self.cur_gate[i] += 1
                if self.cur_gate[i] >= len(self.gates):
                    self._done[i] = True
                else:
                    self._observation[i, 0:3] = self.gates[self.cur_gate[i]] - self.drone_pos[i]

            # if done, give a time-based bonus (lower the better)
            if self._done[i] and self.cur_gate[i] >= len(self.gates):
                self._reward[i] += 2.0 * (1.0 - len(self.rewards[i]) / self.max_episode_steps)
            # if done, no crashes, and drone did not go thru all gates, give penalty
            elif self._done[i] and self.cur_gate[i] < len(self.gates):
                if self._reward[i] != -1:
                    self._reward[i] -= 2.0

            # update reward information if environment is finished
            # update memory to know whether drone crashes (-1 penalty) or not
            self.rewards[i].append(self._reward[i])
            if self._done[i]:
                eplen = len(self.rewards[i])
                eprew = sum(self.rewards[i])
                self.ep_successes.append(self._reward[i] != -1 and self.cur_gate[i] >= len(self.gates))
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                self.rewards[i].clear()
                self.cur_gate[i] = 0

        return self._observation.copy(), self._reward.copy(), \
            self._done.copy(), info.copy()

    def stepUnity(self, action, send_id):
        """Call a step in unity if wrapper is attached.

        Args:
            action: [normalized thrust, roll, pitch, yaw] rates
            send_id: ID for unity enivonrment
        """
        receive_id = self.wrapper.stepUnity(action, self._observation,
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

        if self.randomize_gates:
            self.randomize_gates = False
            self.gates = np.array([
                [0.0, rand.uniform(0.0, 5.0), rand.uniform(0.0, 5.0)],
                [rand.uniform(-2.0, 2.0), rand.uniform(5.0, 10.0), rand.uniform(5.0, 10.0)],
                [rand.uniform(-2.0, 2.0), rand.uniform(7.0, 15.0), rand.uniform(10.0, 15.0)],
                [rand.uniform(-4.0, 4.0), rand.uniform(8.0, 16.0), rand.uniform(17.0, 25.0)]
            ], dtype=np.float32)

        self.wrapper.reset(self._observation)
        return self._observation.copy()

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
        if success_rate >= 0.9:
            self.randomize_gates = True
            self.ep_successes.clear()

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