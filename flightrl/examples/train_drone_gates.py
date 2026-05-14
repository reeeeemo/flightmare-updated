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
#   --randomize <1/0> (default 0) (can also use -r)
# Example:
# python3 train_drone_gates.py
#   --render 1
#   --train 0
#   --weight ./saved/quadrotor_env.zip
#   --norm_weight ./saved/quadrotor_env/vec_normalize.pkl (optional)

class CurriculumCallback(BaseCallback):
    def _on_rollout_start(self) -> None:
        self.training_env.curriculum_callback()
        self.logger.record("curriculum/randomize_gates", self.training_env.randomize_gates)

    def _on_step(self) -> bool:
        return True

def configure_random_seed(seed, env=None):
    if env is not None:
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
                        help="randomize gates activatd")
    return parser

def main():
    args = parser().parse_args()

    # set yaml for quadcopter environments
    yaml = YAML()
    cfg = yaml.load(open(os.environ["FLIGHTMARE_PATH"] +
                         "/flightlib/configs/vec_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"
    
    if args.camera:
        cfg["env"]["camera"] = "yes"
    else:
        cfg["env"]["camera"] = "no"

    stream = StringIO()
    yaml.dump(cfg, stream)

    # flies through gates
    env = QuadcopterGatesVec(
        QuadrotorEnv_v1(stream.getvalue(), False),
        use_cam=args.camera,
        training=(not args.render),
        vision_weights=args.wv
    )
    env.randomize_gates = bool(args.r)

    # set random seed
    configure_random_seed(args.seed, env=env)

    # create file for saving stuff, or add gates if rendering
    if not args.render:
        rsg_root = str(Path(__file__).resolve().parent)
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)

    # add gates to environment
    if args.r and args.render:
        n_gates = np.random.randint(6, 11)
        n_gates_arr = np.arange(n_gates)

        # randomize positions / rotations
        positions = np.zeros((n_gates, 3), dtype=np.float32)
        rotations = np.zeros((n_gates, 4), dtype=np.float32)
        for i in range(n_gates):
            old_pos_x = positions[i-1, 0] if i-1 >= 0 else 0
            old_pos_y = positions[i-1, 1] if i-1 >= 0 else 0
            old_pos_z = positions[i-1, 2] if i-1 >= 0 else 5
            prev_dx = positions[i-1, 0] - positions[i-2, 0] if i >= 2 else 0
            #-5, 5 for p1, -12 12 for p2
            positions[i, 0] = old_pos_x + prev_dx * 0.4 + np.random.uniform(-12, 12)
            #6-7 p1, 8-10 p2
            positions[i, 1] = old_pos_y + np.random.uniform(8,10) # y always close but not intersecting/too close
            positions[i, 2] = np.clip(np.random.uniform(old_pos_z-4, old_pos_z+4), 5.0, np.inf)
            
            # new rot based on approach angle from cur gate + noise
            approach_dx = positions[i, 0] - old_pos_x
            approach_dy = positions[i, 1] - old_pos_y
            new_yaw = np.arctan2(approach_dx, approach_dy) + np.random.uniform(-np.pi/4, np.pi/4)
            new_yaw = np.clip(new_yaw, -np.pi/4, np.pi/4)
            half = new_yaw / 2
            
            rotations[i, 0] = 1 #np.cos(half)
            rotations[i, 3] = 0 #np.sin(half)        
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
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            model = PPO(
                tensorboard_log=saver.data_dir,
                policy="MlpPolicy",  # check activation function
                policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                    net_arch=dict(pi=[64, 64], vf=[64, 64])), # 128, 128, 128, 128
                env=env,
                gae_lambda=0.95,
                gamma=0.999,  # 0.999
                # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
                n_steps=2048,
                ent_coef=0.005, # 0.005, 0.001 worked best so far
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
                env = VecNormalize(env, norm_obs=True, norm_reward=True)
            else:
                env = VecNormalize.load(args.norm_weight, env)
            model = PPO.load(args.weight, env=env, device="cpu")

        model.learn(
            total_timesteps=int(4e7), #normally 6e7
            progress_bar=False,
            reset_num_timesteps=reset_timesteps,
            callback=CurriculumCallback()
        )
        model.save(saver.data_dir)
        env.save(saver.data_dir + "/vec_normalize.pkl")

    # # Testing mode with a trained weight
    else:
        env = VecNormalize.load(args.norm_weight, env)
        env.training = False
        model = PPO.load(args.weight, env=env, device="cpu")
        test_model(env, model, render=args.render, weight_path=args.weight, vid=args.camera, vision_weights=args.wv)


if __name__ == "__main__":
    main()
