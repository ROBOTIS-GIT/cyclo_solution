# Cyclo Solution

This repository provides GPU-accelerated motion planning packages for the
ROBOTIS Physical AI lineup. It currently integrates MoveIt 2, NVIDIA Isaac ROS
cuMotion, and nvblox for FFW robots.

## Repository Structure

```text
├── cyclo_cumotion/
│   ├── cyclo_cumotion_bringup/
│   ├── cyclo_cumotion_description/
│   ├── cyclo_cumotion_moveit_config/
│   ├── cyclo_cumotion_nvblox/
│   ├── cyclo_cumotion_object_attachment/
│   └── cyclo_cumotion_robot_segmenter/
└── docker/
```

## Directory Description

- `cyclo_cumotion_bringup`: Integrated launch files.
- `cyclo_cumotion_description`: FFW planning model.
- `cyclo_cumotion_moveit_config`: MoveIt and cuMotion configuration.
- `cyclo_cumotion_nvblox`: Multi-camera nvblox integration.
- `cyclo_cumotion_object_attachment`: Predefined object attachment.
- `cyclo_cumotion_robot_segmenter`: Robot removal from depth images.
- `docker`: Container image and runtime scripts.

## Install and Build

Start the released Docker image and enter the container:

```bash
cd cyclo_solution
./docker/container.sh start
./docker/container.sh enter
```

Build the ROS 2 workspace inside the container:

```bash
cb
```

The repository is mounted at `/root/ros2_ws/src/cyclo_solution`. The image
also builds the `feature-cumotion` branch of
[`robotis_interfaces`](https://github.com/ROBOTIS-GIT/robotis_interfaces).

For Dockerfile or system dependency changes, build and start a local image:

```bash
docker build -f docker/Dockerfile.amd64 \
  -t robotis/cyclo-solution:0.1.0-local .

CYCLO_IMAGE=robotis/cyclo-solution:0.1.0-local \
CYCLO_SKIP_PULL=1 \
  ./docker/container.sh start
```

## Run

Start the robot bringup first, then launch cuMotion:

```bash
ros2 launch cyclo_cumotion_bringup cumotion_moveit.launch.py
```

Available planning groups are `arm_l`, `arm_r`, `both_arms`, and `wholebody`.
RViz starts by default.

Common launch arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `read_esdf_world` | `false` | Enable nvblox collision checking. |
| `nvblox_camera_set` | `all` | Select `all`, `head`, or `wrists`. |
| `enable_object_attachment` | `false` | Enable predefined object attachment. |
| `enable_static_scene` | `false` | Load the configured static collision scene. |
| `start_rviz` | `true` | Start RViz with the planning configuration. |

Example with depth-based collision checking and object attachment:

```bash
ros2 launch cyclo_cumotion_bringup cumotion_moveit.launch.py \
  read_esdf_world:=true \
  enable_object_attachment:=true
```

Planned robot motion and end-effector paths are visualized in RViz.

### Object Attachment

The default catalog contains `table_box`. An empty `attachment_frame` uses its
configured default, `end_effector_r_link`.

```bash
ros2 service call \
  /cyclo_cumotion/object_attachment/attach_by_name \
  robotis_interfaces/srv/AttachObjectByName \
  "{object_name: table_box, attachment_frame: ''}"

ros2 service call \
  /cyclo_cumotion/object_attachment/detach \
  std_srvs/srv/Trigger '{}'
```

## Zenoh

The repository does not set a remote Zenoh endpoint. Configure
`ZENOH_CONFIG_OVERRIDE` in the container user's `.bashrc`, or start a local
router with the `zenohd` alias.

## References

- [Isaac ROS cuMotion MoveIt](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html)
- [Isaac ROS Object Attachment](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_object_attachment/index.html)
- [ROBOTIS Docker Images](https://hub.docker.com/u/robotis)

## License

This project is licensed under Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
