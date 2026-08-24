# Flightmare, Updated

![License](https://img.shields.io/badge/License-MIT-blue.svg)

> If you're curious to see this in use, checkout my [X profile](https://x.com/robobOxley)! I have posted several video clips using the policies trained in this simulator.

---

**Vision-based RL for Autonomous Drone Racing**: 📄 [Technical Writeup PDF](./FlightmareUpdatedWriteup.pdf)

---

**Flightmare** is a flexible modular quadrotor simulator.

This is an updated version of Flightmare that includes newer libraries and better docker support. Most if not all of the changes reside in the `flightrl` and `flightlib` folders.

I built this updated simulator to compete in the [Anduril AI Grand Prix](https://www.theaigrandprix.com/). Therefore the randomly generated gate courses, curriculum, and domain randomization were added for that competition. See [Training](./README.md#training) for details.

---

Flightmare is composed of two main components: a configurable rendering engine built on Unity and a flexible physics engine for dynamics simulation.
- Those two components are totally decoupled and can run independently from each other. 
- Flightmare also comes with several features: 
  - A large multi-modal sensor suite, including an interface to extract the 3D point-cloud of the scene
  - An API for reinforcement learning which can simulate hundreds (default: 100) of quadrotors in parallel
  - An integration with a virtual-reality headset for interaction with the simulated environment.

Flightmare can be used for various applications, including path-planning, reinforcement learning, visual-inertial odometry, deep learning, human-robot interaction, etc.

**[Website](https://uzh-rpg.github.io/flightmare/)** & 
**[Documentation](https://flightmare.readthedocs.io/)** 

[![FLIGHTMARE VIDEO](./docs/flightmare_main.png)](https://youtu.be/m9Mx1BCNGFU)

# What differs this from the original?
- While Flightmare is originally intended for Python 3.6 and Ubuntu 18.04, updates have been made to include **Python 3.9** and **Ubuntu 20.04**, which is what the Docker runs on. This also includes support for CUDA devices tested up to **CUDA 13.1**.
- Updated Dockerfile and `.devcontainer` for a streamlined coding process. (See [Installation](./README.md#installation) for details).
- Added support for newer libraries such as stable_baselines3 and PyTorch so the newest versions of models such as PPO or SAC can be run.
- 2 New environments for reinforcement-learning based training on a drone learning to hover and fly through gate objects.
  - Includes the ability to add gates from Python versus hard-coding in C++ then recompiling
  - Can also configure the initial random rotation of the drone at the starting state
  - Includes support for cumulative learning of the model and checking gate collision without rigidbodies.
- Allows for the collection of pose estimations from each gate flown through inside of an environment into a YOLO-style pose estimation dataset and to train a pose estimation model or segmentation model on the dataset collected.

> The updated content only includes `flightrl` and `flightlib`, as of the time of writing, no work has been done to update and test `flightros` or `flightrender` (I used unity standalone for any rendering). Proceed with caution.

## Installation

Build the dockerfile using `docker build -t flightmare .`.

- If only running on docker
  - `docker run --gpus all -it -d flightmare`


- Docker with Visual Studio Code
  - Ensure that DevContainers is installed as an extension
  - Press `CTRL+SHIFT+P` and select `Dev Containers: Attach to Running Container...`


After docker is built, to run anything in `flightrl`:
- Run `pip3 install -e .` inside of `flightrl`

> Run `export FLIGHTMARE_PATH=/workspace` to mitigate any other issues unrelated to packages. This also forces the [yaml](./flightlib/configs/quadrotor_env.yaml) to be considered during training/inference.

## Training

Training a policy on navigating autonomously through a series of gates can be done by running the `train_drone_gates.py`. Each phase increases course difficulty, and the number of gates within a phase increases as the policy's success rate crosses a threshold.

| Phase | Course | Extras |
| --- | --- | --- |
| 1 | Forward-facing gates, small vertical/lateral spacing differences | No yaw rotations |
| 2 | S-shaped Course | Turns capped at 40 degrees |
| 3 | S-shaped Course | Domain Randomization (mass, thrust, motor lag, etc.), perception noise, FOV drift |

Each phase's policy and VecNormalize weights were warm-started from the previous one.

The drone model that is trained is described below:
- 34 observations are inputted into a PPO policy
  - Drone-relative Current gate XYZ [0:3]
  - Body-relative drone rotation matrix [3:12]
  - Linear Velocity [12:15]
  - Angular Velocity [15:18]
  - Previous action outputted [18:22]
  - Drone-relative current gate XYZ in each of the gate's 4 corners [22:34]
- The drone outputs `[normalized_thrust, pitch, roll, yaw]` rates
- Dynamic floors/walls each episode to force precision at speed
- Dead reckoning during training to match potential camera dropouts (FOV drift, frozen images)
- Domain randomization 
  - +/- 20% mass, arm length
  - +/- 40% thrust map coefs
  - 0.01-0.1 motor_tau
  - +/- 30% kappa, motor_omega_min
  - 3000-5000 motor_omega_max
  - +/- 50% rate-loop gains (kn roll/pitch/yaw)
- Launches off of a designated pad.

--- 
`train_drone_gates.py`

Trains a PPO policy on the information given above.

**Args:**

| Flag | Default | Meaning | 
| --- | --- | --- |
| `--train` | 1 | Whether the policy inputted is being trained.
| `--render` | 0 | Whether the policy inputted is being rendered (requires unity standalone or `flightrender`)
| `--save_dir` | `./examples` | Directory to save checkpoints/training models
| `--seed` | 0 | Seeding for NumPy (random gate generation)
| `-w` / `--weight` | `./saved/quadrotor_env.zip` | Trained weight path (default is to be left if model is on phase 1)
| `-wn` / `--norm_weight` | `''` | Trained normalization weights for model
| `--camera` | 0 | To add a camera onto each environment for detection. **Requires render=1**
| `--wv` / `--vision_weights` | `''` | Vision weights for camera inference
| `--bd` / `--build_dataset` | 0 | Whether to build YOLO-style pose dataset
| `--p` / `--phase` | 1 | Current phase of drone training `[1, 2, 3]`
| `--ct` / `--crash_detection` | 0 | Whether to use crash detection or not
| `--pl` / `--pad_launch` | 1 | Whether to use pad launch or random initialization

---

`train_drone_hover.py`

Trains a PPO policy to hover at a GOAL_XYZ/RPY stated in-line.

**Args**
| Flag | Default | Meaning | 
| --- | --- | --- |
| `--train` | 1 | Whether the policy inputted is being trained.
| `--render` | 0 | Whether the policy inputted is being rendered (requires unity standalone or `flightrender`)
| `--save_dir` | `./examples` | Directory to save checkpoints/training models
| `--seed` | 0 | Seeding for NumPy (random gate generation)
| `-w` / `--weight` | `./saved/quadrotor_env.zip` | Trained weight path
| `-wn` / `--norm_weight` | `''` | Trained normalization weights for model

> If one desires to view their created dataset with labels overlaid or to train a vision model on said dataset, read [train_vision_model_sim.py](./flightrl/examples/train_vision_model_sim.py) and [view_dataset.py](./flightrl/examples/view_dataset.py)

## Publication of Original Flightmare

```
@inproceedings{song2020flightmare,
    title={Flightmare: A Flexible Quadrotor Simulator},
    author={Song, Yunlong and Naji, Selim and Kaufmann, Elia and Loquercio, Antonio and Scaramuzza, Davide},
    booktitle={Conference on Robot Learning},
    year={2020}
}
```

> **[PDF](http://rpg.ifi.uzh.ch/docs/CoRL20_Yunlong.pdf)**


## License
This project is released under the MIT License. Please review the [License file](LICENSE) for more details.
