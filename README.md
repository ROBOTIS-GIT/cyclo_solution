# Cyclo Solution

Cyclo Solution contains modular Isaac ROS applications for ROBOTIS robots. The
current `cyclo_cumotion` module integrates MoveIt 2, Isaac ROS cuMotion, nvblox,
and moving-camera robot segmentation for FFW robots.

The repository is bind-mounted at `/root/ros2_ws/src/cyclo_solution` so source
changes are immediately available in the development container.

## Packages

- `cyclo_cumotion`: module bringup and feature meta-package
- `cyclo_cumotion_description`: common FFW planning robot description
- `cyclo_cumotion_moveit_config`: MoveIt, cuMotion, and XRDF configuration
- `cyclo_cumotion_nvblox`: robot-segmented nvblox integration
- `cyclo_cumotion_object_attachment`: predefined-object attachment integration
- `cyclo_cumotion_robot_segmenter`: moving-camera robot segmenter

The cuMotion-specific message and service definitions are provided by the
`feature-cumotion` branch of
[`robotis_interfaces`](https://github.com/ROBOTIS-GIT/robotis_interfaces).
The development image clones and builds that branch together with this
repository.

## Development container

```bash
cd cyclo_solution/docker
./container.sh start
./container.sh enter
```

The image uses ROS 2 Jazzy and Isaac ROS 4.6. The single amd64 Dockerfile can
reuse `robotis/cyclo-solution:4.6-local` as a prebuilt dependency base when it
is available, so CUDA and Isaac ROS packages are not rebuilt unnecessarily.

Build the workspace inside the container:

```bash
cb
```

Only Dockerfile or system dependency changes require an image rebuild.

## cuMotion planning

Launch the common FFW planning model:

```bash
ros2 launch cyclo_cumotion cumotion_moveit.launch.py
```

The model shares the lift, arms, grippers, planning groups, and collision
spheres across FFW variants. The common URDF also contains the SG2 swerve-wheel
joints and visual links so full SG2 joint-state messages are valid. On BG2,
those joints remain at their zero defaults and are outside every planning
group and the cuMotion collision-sphere model.

Available MoveIt planning groups are `arm_l`, `arm_r`, `both_arms`, and
`wholebody`. A single cuMotion backend is reconfigured by the group router when
the selected group changes.

Depth-based world collision checking is disabled by default. Enable it with:

```bash
ros2 launch cyclo_cumotion cumotion_moveit.launch.py \
  read_esdf_world:=true
```

A static MoveIt scene can be enabled independently:

```bash
ros2 launch cyclo_cumotion cumotion_moveit.launch.py \
  enable_static_scene:=true
```

Object attachment is disabled by default. Enable the NVIDIA attachment node
and predefined-object catalog together with cuMotion using:

```bash
ros2 launch cyclo_cumotion cumotion_moveit.launch.py \
  enable_object_attachment:=true
```

Attach the catalog's `small_box` to either gripper. An empty
`attachment_frame` uses the object's configured default:

```bash
ros2 service call \
  /cyclo_cumotion/object_attachment/attach_by_name \
  robotis_interfaces/srv/AttachObjectByName \
  "{object_name: small_box, attachment_frame: end_effector_l_link}"
ros2 service call \
  /cyclo_cumotion/object_attachment/detach std_srvs/srv/Trigger '{}'
```

Objects are defined in
`cyclo_cumotion_object_attachment/config/predefined_objects.yaml`. Attached
collision spheres and the selected left/right attachment frame are preserved
when switching between `arm_l`, `arm_r`, `both_arms`, and `wholebody`.

No perception topic or CenterPose gateway is started. The
`robotis_interfaces/AttachmentGeometry` message remains as the neutral
geometry contract for a future perception adapter; the current runtime only
accepts catalog object names.

## Zenoh

Docker Compose does not start a Zenoh router. Start one explicitly with the
`zenohd` alias when a local router is required. No `ZENOH_CONFIG_OVERRIDE` is
set by the Dockerfile or Compose configuration.

When a remote router is required, set `ZENOH_CONFIG_OVERRIDE` explicitly in the
container user's `.bashrc`. The endpoint is intentionally not stored in this
repository.

Stop and remove the module containers with:

```bash
./container.sh stop
```

## References

- [Isaac ROS Getting Started](https://nvidia-isaac-ros.github.io/getting_started/index.html)
- [Isaac ROS cuMotion MoveIt](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_moveit/index.html)
- [Isaac ROS cuMotion Object Attachment](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion_object_attachment/index.html)

## License and attribution

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
`cyclo_cumotion_robot_segmenter` contains modifications derived from NVIDIA
Isaac ROS cuMotion. See [NOTICE](NOTICE) for attribution.
