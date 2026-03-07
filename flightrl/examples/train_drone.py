from ruamel.yaml import YAML

from pathlib import Path
from io import StringIO
import argparse
import numpy as np
import torch
import os

from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rpg_baselines.common.test_model import test_model
from rpg_baselines.envs import quadcopter_hover_vec as wrapper
import rpg_baselines.common.util as U

from flightgym import QuadrotorEnv_v1


# Trains drone on its ability to hover.
# Allows for rendering via unity and saved weights

# Use: python3 train_drone.py
#   --render <1/0>  (default 1)
#   --train <1/0>   (default 0)
#   --save_dir <save_dir>   (default ./saved)
#   --seed <seed_int>   (default 0)
#   --weight <saved_quadcopter_zip> (can also use -w)
# Example:
# python3 train_drone.py
#   --render 1
#   --train 0
#   --weight ./saved/quadrotor_env.zip

class CurriculumCallback(BaseCallback):
    def _on_rollout_start(self) -> None:
        self.training_env.curriculum_callback()
        self.logger.record("curriculum/rot_mult", self.training_env.rot_mult)

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
    return parser

def main():
    args = parser().parse_args()
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

    stream = StringIO()
    yaml.dump(cfg, stream)
    env = wrapper.QuadcopterHoverVec(
        QuadrotorEnv_v1(stream.getvalue(), False),
        GOAL_XYZ=np.array([0.0, 0.0, 5.0]),
        GOAL_RPY=np.array([0.0, 0.0, 0.0])
    )

    # set random seed
    configure_random_seed(args.seed, env=env)

    #
    rsg_root = str(Path(__file__).resolve().parent)
    log_dir = rsg_root + '/saved'
    saver = U.ConfigurationSaver(log_dir=log_dir)
    if args.train:
        if args.weight == "./saved/quadrotor_env.zip":
            model = PPO(
                tensorboard_log=saver.data_dir,
                policy="MlpPolicy",  # check activation function
                policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                    net_arch=dict(pi=[64, 64], vf=[64, 64])),
                env=env,
                gae_lambda=0.95,
                gamma=0.99,  # lower 0.9 ~ 0.99
                # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
                n_steps=2048,
                ent_coef=0.005,
                learning_rate=3e-4,
                vf_coef=0.5,
                max_grad_norm=0.5,
                batch_size=512,
                n_epochs=10,
                clip_range=0.2,
                verbose=1,
                device="cpu"
            )
        else:
            model = PPO.load(args.weight, env=env, device="cpu")
        # https://flightmare.readthedocs.io/en/latest/python_references/flight_env_vec.html#FlightEnvVec
        model.learn(
            total_timesteps=int(2e7),  # 2e7
            progress_bar=False,
            reset_num_timesteps=False,
            callback=CurriculumCallback()
        )
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:
        model = PPO.load(args.weight, device="cpu")
        test_model(env, model, render=args.render, weight_path=args.weight)


if __name__ == "__main__":
    main()
