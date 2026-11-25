# Cyclo Solution

Cyclo Solution provides modular ROS 2 solution packages for ROBOTIS Physical
AI systems. Each solution is organized as a feature group so that perception,
planning, and manipulation capabilities can be added independently.

## Prerequisites

Cyclo Solution follows the
[NVIDIA Isaac ROS system requirements](https://nvidia-isaac-ros.github.io/v/release-4.6/getting_started/index.html#system-requirements),
including its supported NVIDIA GPU, operating system, driver, Docker, and
NVIDIA Container Toolkit requirements.

## Build and start on the workstation

On an x86_64 GPU workstation with Docker Compose 2.30 or later, run:

```bash
git clone https://github.com/ROBOTIS-GIT/cyclo_solution.git
cd cyclo_solution
./docker/container.sh start
./docker/container.sh enter
```

The image builds locally from the public CUDA base image and ROS packages.
The Dockerfile clones `cyclo_solution` from `main` and `robotis_interfaces`
from `jazzy`. Docker caches this step. To pick up new commits on those
branches, run the following commands from the repository root:

```bash
docker compose -f docker/docker-compose.yml build --no-cache
./docker/container.sh start
```

## Available Solutions

### cuMotion

The [`cyclo_cumotion`](cyclo_cumotion) feature group provides GPU-accelerated
motion planning for the AI Worker FFW platform using MoveIt 2 and NVIDIA Isaac
ROS cuMotion.

It includes:

- GPU-accelerated single-arm, dual-arm, and whole-body motion planning
- Depth-aware collision mapping and robot segmentation
- Object attachment and RViz trajectory visualization

Setup instructions and usage examples are provided in the
[Isaac cuMotion guide](https://docs.robotis.com/docs/systems/aiworker/resources/technical_story/isaac_cumotion/).

## Planned Features

Future releases will add more feature-oriented solution groups, including:

- Perception and object-understanding pipelines
- Reusable manipulation and task-execution workflows
- Support for additional ROBOTIS Physical AI platforms and sensor configurations

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE)
and [NOTICE](NOTICE) for details.
