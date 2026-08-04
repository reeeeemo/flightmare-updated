from ruamel.yaml import YAML

from pathlib import Path
from io import StringIO
import argparse
import numpy as np
import torch
import os
import random as rand

from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from rpg_baselines.common.test_model import test_model
from rpg_baselines.envs.quadcopter_gates_vec import QuadcopterGatesVec
import rpg_baselines.common.util as U
from stable_baselines3.common.utils import LinearSchedule

from flightgym import QuadrotorEnv_v1


# Trains drone on its ability to hover.
# Allows for rendering via unity and saved weights

# Use: python3 train_drone_gates.py
#   --render <1/0>  (default 1)
#   --train <1/0>   (default 0)
#   --save_dir <save_dir>   (default ./saved)
#   --seed <seed_int>   (default 0)
#   --weight <saved_quadcopter_zip> (can also use -w)
#   --norm_weight <saved_quadcopter_normalization_stats> (can also use -wn)
#   --camera <1/0> (default 0)
#   --vision_weights <saved_vision_pt> (can also use --wv)
#   --randomize <1/0> (default 0) (can also use --r)
#   --build_dataset <1/0> (default 0) (can also use --bd)
# Example:
# python3 train_drone_gates.py
#   --render 1
#   --train 0
#   --weight ./saved/quadrotor_env.zip
#   --norm_weight ./saved/quadrotor_env/vec_normalize.pkl (optional)

class SelectiveVecNormalize(VecNormalize):
    def __init__(self, venv,
                 norm_obs=True, 
                 norm_reward=True,
                 exclude_indices: list = None, **kwargs
       ):
        super().__init__(venv, norm_obs=norm_obs, norm_reward=norm_reward, **kwargs)
        self.exclude_indices = list(exclude_indices) if exclude_indices else []
    
    def normalize_obs(self, obs):
        normed = super().normalize_obs(obs)
        normed[:, self.exclude_indices] = obs[:, self.exclude_indices]
        return normed

class CurriculumCallback(BaseCallback):
    def _on_rollout_start(self) -> None:
        self.training_env.curriculum_callback()
        with torch.no_grad():
            self.model.policy.log_std.data.clamp_(max=0.0,min=-1.897)
        self.logger.record("curriculum/n_gates", self.training_env.n_gates)
        self.logger.record("curriculum/inner_depth", self.training_env.half_h)
        self.logger.record("curriculum/max_std", float(self.model.policy.log_std.detach().exp().max()))
        ep_successes = self.training_env.venv.ep_successes
        if len(ep_successes):
            self.logger.record("curriculum/success_rate", float(sum(ep_successes) / len(ep_successes)))

    def _on_step(self) -> bool:
        return True
    
# entropy scheduler since determminism would be enhanced through curriculum learning
# and if large entropy towards the end of training it relies on noise.
class EntropySchedulerCallback(BaseCallback):
    """Entropy scheduler that goes from start -> end via 1->0 progress"""
    def __init__(self, start: float = 0.005, end: float = 0.001, end_fraction: float = 0.9):
        super().__init__()
        self.start = start
        self.end = end
        self.end_fraction = end_fraction
        self.schedule = LinearSchedule(start, end, end_fraction)

    def _on_step(self) -> bool:
        progress = (self.training_env.n_gates - self.training_env.start_gate) / max(1, (self.training_env.n_gates_target - self.training_env.start_gate))
        self.model.ent_coef = self.schedule(1-progress)

        return True

def configure_random_seed(seed, env=None):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=str(Path(__file__).resolve().parent),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight', type=str, default='./saved/quadrotor_env.zip',
                        help='trained weight path')
    parser.add_argument('-wn', '--norm_weight', type=str, default='',
                        help='trained normalization weights for model')
    parser.add_argument('--camera', type=int, default=0,
                        help="To add a camera onto each environment for detections")
    parser.add_argument('--wv', '--vision_weights', type=str, default='',
                        help="vision weights for camera inference")
    parser.add_argument('--r', '--randomize', type=int, default=0,
                        help="randomize gates activated")
    parser.add_argument('--bd', '--build_dataset', type=int, default=0,
                        help="whether to build YOLO-style pose dataset")
    parser.add_argument('--p', '--phase', type=int, default=1,
                        help="what phase of drone training (1, 2)")
    parser.add_argument('--rt', '--reset_timesteps', type=int, default=0,
                        help="whether to reset timesteps for a model or not")
    parser.add_argument('--ct', '--crash_detection', type=int, default=0,
                        help="whether to use crash detection or not")
    parser.add_argument('--pl', '--pad_launch', type=int, default=1,
                        help="whether to use a pad launch or random init")
    return parser

def edit_yaml(args) -> StringIO:
    """Set YAML for quadcopter environments."""
    yaml = YAML()
    cfg = yaml.load(open(os.environ["FLIGHTMARE_PATH"] +
                         "/flightlib/configs/vec_env.yaml", 'r'))

    # -----
    # 1 env if rendering, 100 for train
    # -----
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1
    else:
        cfg["env"]["num_envs"] = 100
        cfg["env"]["num_threads"] = 1
    
    cfg["env"]["render"] = "yes" if args.render else "no"
    cfg["env"]["camera"] = "yes" if args.camera else "no"
    
    stream = StringIO()
    yaml.dump(cfg, stream)
    
    return stream

def main():
    args = parser().parse_args()
    if (args.p not in (1, 2, 3)):
        raise ValueError("Phases (1, 2, 3) are only available to run.")

    stream = edit_yaml(args)

    # flies through gates
    # init gates is 1 for p1, but for p2 and p3 it should be learning gate geometry
    # not single gate behaviors
    env = QuadcopterGatesVec(
        QuadrotorEnv_v1(stream.getvalue(), False),
        use_cam=args.camera,
        vision_weights=args.wv,
        phase=args.p,
        init_gate_num=1,
        crash_det=args.ct,
        is_rendering=args.render,
        pad_launch=args.pl
    )
    env.randomize_gates = bool(args.r)

    # set random seed
    configure_random_seed(args.seed, env=env)

    # create file for saving stuff, or add gates if rendering
    if args.train:
        rsg_root = str(Path(__file__).resolve().parent)
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)

    # add gates to environment
    if args.r:
        n_gates = np.random.randint(6, 14)
        n_gates_arr = np.arange(n_gates)

        # randomize positions / rotations
        positions = np.zeros((n_gates, 3), dtype=np.float32)
        rotations = np.zeros((n_gates, 4), dtype=np.float32)
        flat_probability = [0.8, 0.6, 0.4][args.p-1]
        lowest_margin = 2.0 + (2.0 * args.p)
        for i in range(n_gates):
            old_pos_x = positions[i-1, 0] if i-1 >= 0 else 0
            old_pos_y = positions[i-1, 1] if i-1 >= 0 else 0
            old_pos_z = positions[i-1, 2] if i-1 >= 0 else 2
            prev_dx = positions[i-1, 0] - positions[i-2, 0] if i >= 2 else 0
            #-5, 5 for p1, -12 12 for p2
            random_x_range = (-5, 5) if args.p == 1 else (-8, 8)
            positions[i, 0] = old_pos_x + prev_dx * 0.4 + np.random.uniform(*random_x_range)
            #6-7 p1, 8-10 p2
            random_y_range = [(6, 7), (8, 10), (12, 25)][args.p-1]
            random_z_range = [(1, 2), (2, 3), (2, 3)][args.p-1]

            positions[i, 1] = old_pos_y + np.random.uniform(*random_y_range) # y always close but not intersecting/too close
            random_z = np.random.uniform(*random_z_range)
            
            if np.random.random() < flat_probability:
                positions[i, 2] = old_pos_z
            else:
                positions[i, 2] = np.random.uniform(old_pos_z-random_z, old_pos_z+random_z) #np.clip(np.random.uniform(old_pos_z-4, old_pos_z+4), 2.0, np.inf)
            
            # new rot based on approach angle from cur gate + noise
            approach_dx = positions[i, 0] - old_pos_x
            approach_dy = positions[i, 1] - old_pos_y
            random_yaw_range = (-np.pi/6, np.pi/6)
            new_yaw = np.arctan2(-approach_dx, approach_dy) + np.random.uniform(*random_yaw_range)
            new_yaw = np.clip(new_yaw, -np.pi/6, np.pi/6)
            half = new_yaw / 2
            
            rotations[i, 0] = 1 if args.p == 1 else np.cos(half)
            rotations[i, 3] = 0 if args.p == 1 else np.sin(half)
        env.wrapper.setLowestZ(min(min(positions[:, 2]) - lowest_margin, -1.0))
    else:
        positions = np.array([
            [0, 7.5, 7],
            [0, 13.5, 10],
            [3, 19.5, 12],
            [9, 21.5, 12],
            [15, 19.5, 12],
            [18, 13.5, 10],
            [18, 7.5, 7]
        ], dtype=np.float32)
        rotations = np.array([
            [1, 0, 0, 0],
            [np.cos(np.pi/8), np.sin(np.pi/8), 0, 0], # tilted up
            [-np.cos(np.pi/12), 0, 0, np.sin(np.pi/12)], # tiled right
            [-np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)], # right
            [np.cos(np.pi/3), 0, 0, -np.sin(np.pi/3)], # tiled left
            [np.cos(np.pi/12), np.sin(np.pi/12), 0, 0],
            [1, 0, 0, 0]
        ], dtype=np.float32)

    print(f"gate pos: {positions}")
    print(f"gate rot: {rotations}")
    env.addGate(positions, rotations)
    reset_timesteps = False

    if args.train:
        if args.weight == "./saved/quadrotor_env.zip":
            env = SelectiveVecNormalize(env, 
                                        norm_obs=True, 
                                        norm_reward=True,
                                        exclude_indices=[15, 16, 17])
            model = PPO(
                tensorboard_log=saver.data_dir,
                policy="MlpPolicy",  # check activation function
                policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                    net_arch=dict(pi=[64, 64], vf=[64, 64])), # old: 128, 128, 128, 128
                env=env,
                gae_lambda=0.95,
                gamma=0.9995,  # 0.999
                n_steps=2048,
                ent_coef=0.003,
                learning_rate=1e-4,
                vf_coef=0.5,
                max_grad_norm=0.5,
                batch_size=512,
                n_epochs=10,
                clip_range=0.2,
                verbose=1,
                device="cpu"
            )
        else:
            reset_timesteps = "hover" in args.weight  # only reset if we are using hover weights
            if not args.norm_weight:
                env = SelectiveVecNormalize(env, 
                                            norm_obs=True, 
                                            norm_reward=True,
                                            exclude_indices=[15, 16, 17])
            else:
                env = SelectiveVecNormalize.load(args.norm_weight, env)
            model = PPO.load(args.weight, env=env, device="cpu")
        
        total_timesteps = 1.2e8 if args.p in (1, 2) else 3.2e8
        starting_entropy = 0.005
        model.learn(
            total_timesteps=int(total_timesteps), 
            progress_bar=False,
            reset_num_timesteps=reset_timesteps,
            callback=[
                CurriculumCallback(), 
                EntropySchedulerCallback(start=starting_entropy, end=0.001)
            ]
        )
        model.save(saver.data_dir)
        env.save(saver.data_dir + "/vec_normalize.pkl")
        if args.render:
            env.disconnectUnity()

    # # Testing mode with a trained weight
    else:
        env = SelectiveVecNormalize.load(args.norm_weight, env)
        env.training = False
        model = PPO.load(args.weight, env=env, device="cpu")
        test_model(
            env, model, 
            num_rollouts=1,
            render=args.render, 
            weight_path=args.weight, 
            vid=args.camera, 
            vision_weights=args.wv,
            build_dataset=args.bd
        )


if __name__ == "__main__":
    main()
