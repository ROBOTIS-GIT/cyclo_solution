# Cyclo Solution

Cyclo Solution provides MoveIt 2 and Isaac ROS cuMotion integration for
ROBOTIS FFW robots. Related packages are grouped under `cyclo_cumotion/`.

## Packages

- `cyclo_cumotion_bringup`: integrated launch files
- `cyclo_cumotion_description`: FFW planning model
- `cyclo_cumotion_moveit_config`: MoveIt and cuMotion configuration
- `cyclo_cumotion_nvblox`: multi-camera nvblox integration
- `cyclo_cumotion_object_attachment`: predefined object attachment
- `cyclo_cumotion_robot_segmenter`: robot removal from depth images

The repository is mounted in the container at
`/root/ros2_ws/src/cyclo_solution`. The Docker image also builds the
`feature-cumotion` branch of
[`robotis_interfaces`](https://github.com/ROBOTIS-GIT/robotis_interfaces).

## Container

```bash
cd cyclo_solution
./docker/container.sh start
./docker/container.sh enter
```

`start` pulls `robotis/cyclo-solution:0.1.0` and starts a container that is
automatically restarted after a host reboot. Build source changes inside the
container with:

```bash
cb
```

Rebuild the image only after changing the Dockerfile or system dependencies:

```bash
docker build -f docker/Dockerfile.amd64 \
  -t robotis/cyclo-solution:0.1.0-local .

CYCLO_IMAGE=robotis/cyclo-solution:0.1.0-local \
CYCLO_SKIP_PULL=1 \
  ./docker/container.sh start
```

## Run cuMotion

The robot bringup must already be running. Start cuMotion with:

```bash
ros2 launch cyclo_cumotion_bringup cumotion_moveit.launch.py
```

Planning groups are `arm_l`, `arm_r`, `both_arms`, and `wholebody`. RViz is
started by default.

Common options:

| Option | Description |
| --- | --- |
| `read_esdf_world:=true` | Enable nvblox collision checking |
| `nvblox_camera_set:=all` | Use the ZED and both wrist depth cameras (default) |
| `nvblox_camera_set:=head` | Use only the ZED |
| `nvblox_camera_set:=right` | Use both wrist depth cameras (legacy option name) |
| `enable_object_attachment:=true` | Enable predefined object attachment |
| `enable_static_scene:=true` | Load the configured static collision scene |
| `start_rviz:=false` | Run without RViz |

Options can be combined. For example:

```bash
ros2 launch cyclo_cumotion_bringup cumotion_moveit.launch.py \
  read_esdf_world:=true \
  enable_object_attachment:=true
```

Planned robot motion and end-effector paths are shown in RViz. The
end-effector line is removed after playback by default. Keep it with
`planned_end_effector_path_auto_clear:=false`.

## Object attachment

Launch with `enable_object_attachment:=true`, then attach the catalog's
`table_box`. Leaving `attachment_frame` empty uses its configured default,
`end_effector_r_link`.

```bash
ros2 service call \
  /cyclo_cumotion/object_attachment/attach_by_name \
  robotis_interfaces/srv/AttachObjectByName \
  "{object_name: table_box, attachment_frame: ''}"

ros2 service call \
  /cyclo_cumotion/object_attachment/detach \
  std_srvs/srv/Trigger '{}'
```

Objects are configured in
`cyclo_cumotion/cyclo_cumotion_object_attachment/config/predefined_objects.yaml`.

## Zenoh

The repository does not set a remote Zenoh endpoint. Configure
`ZENOH_CONFIG_OVERRIDE` in the container user's `.bashrc`, or run the local
router with the `zenohd` alias.

Stop and remove the container with `./docker/container.sh stop`.

## References

- [Isaac ROS cuMotion MoveIt](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html)
- [Isaac ROS Object Attachment](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_object_attachment/index.html)

## License

This project is licensed under Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
