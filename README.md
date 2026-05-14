# Flightmare - 左青龙

![Build Status](https://github.com/uzh-rpg/flightmare/workflows/CPP_CI/badge.svg) ![clang format](https://github.com/uzh-rpg/flightmare/workflows/clang_format/badge.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg) ![website]( https://img.shields.io/website-up-down-green-red/https/naereen.github.io.svg)


**Flightmare** is a flexible modular quadrotor simulator.

This is an updated version of Flightmare that includes newer libraries and better docker support. Most of the changes revolve around `flightrl` and creating a drone environment that is abstracted from the C++ based physics/simulation environment.

Flightmare is composed of two main components: a configurable rendering engine built on Unity and a flexible physics engine for dynamics simulation.
Those two components are totally decoupled and can run independently from each other. 
Flightmare comes with several desirable features: (i) a large multi-modal sensor suite, including an interface to extract the 3D point-cloud of the scene; (ii) an API for reinforcement learning which can simulate hundreds of quadrotors in parallel; and (iii) an integration with a virtual-reality headset for interaction with the simulated environment.
Flightmare can be used for various applications, including path-planning, reinforcement learning, visual-inertial odometry, deep learning, human-robot interaction, etc.

**[Website](https://uzh-rpg.github.io/flightmare/)** & 
**[Documentation](https://flightmare.readthedocs.io/)** 

[![FLIGHTMARE VIDEO](./docs/flightmare_main.png)](https://youtu.be/m9Mx1BCNGFU)

## Updated Content
- While Flightmare is originally intended for Python 3.6 and Ubuntu 18.04, updates have been made to include **Python 3.9** and **Ubuntu 20.04**, which is what the Docker runs on. This also includes support for CUDA devices up to **CUDA 12.4**.
- Updated Dockerfile and `.devcontainer` for a streamlined coding process.
- Added support for stable_baselines3 and PyTorch so models such as PPO can be run.
- 2 New environments for reinforcement learning based training on a drone learning to hover and fly through gate objects.
  - Includes the ability to add gates from Python versus hard-coding in C++ then recompiling
  - Can also configure the initial random rotation of the drone at the starting state
  - Includes support for cumulative learning of the model and checking gate collision without rigidbodies.

> The updated content only includes `flightrl` and `flightlib`, as of the time of writing, no work has been done to update and test `flightros` or `flightrender` (we have used unity standalone for any rendering). Proceed with caution.

## Installation

Build the dockerfile using `docker build -t flightmare .`.

- If only running on docker
  - `docker run --gpus all -it -d flightmare`


- Docker w/Visual Studio Code
  - Ensure that DevContainers is installed as an extension
  - Press `CTRL+SHIFT+P` and select `Dev Containers: Attach to Running Container...`


After docker is built, to run anything in `flightrl`:
- Run `pip3 install -e .` inside of `flightrl`

> Run `export FLIGHTMARE_PATH=/workspace` if running into any other issues unrelated to packages.


## Publication

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
