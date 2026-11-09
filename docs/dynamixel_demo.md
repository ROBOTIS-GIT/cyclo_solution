# DYNAMIXEL pick → handover → cup demo

TaskManager owns the full state machine. There is no separate DemoSequence runner.
The existing per-motion actions remain available for debugging.

## Software layout

- `src/tasks/demo_actions.cpp`: `/run_demo`, stage progression, bounded retries, cancellation and checkpoints.
- `include/demo_1f/tasks/demo_recovery.hpp`: retry policy.
- `modules/dynamixel_pick/ffw_vision`: upstream YOLO11 segmentation + aligned depth measurement.
- `modules/dynamixel_pick/ffw_grasp`: upstream multi-frame fusion, basket geometry and grasp candidates.
- `modules/dynamixel_pick/ffw_pick`: upstream analytic IK/path search and execution; `action_server.py` exposes a cancellable backend.
- `modules/dynamixel_pick/ffw_vision_msgs`: upstream perception interfaces plus internal `PickDynamixel` action.

Python is retained for the existing pick engine. TaskManager talks to its internal
`/dynamixel_pick/execute` action and waits for its terminal result. Pick uses its existing
MoveIt execution client; handover/cup motions use the C++ TrajectoryExecutor. TaskManager
serializes these two execution paths. Do not invoke the internal backend or old `pick_moveit`
CLI concurrently with demo motions. Both use the SAME move_group and robot controllers.

The original checkout under `ai_worker/` is not changed or used to build the imported modules;
`modules/dynamixel_pick/UPSTREAM.md` records their source commit. `ai_worker/COLCON_IGNORE`
prevents a duplicate bringup build. Camera drivers and robot bringup remain external.

## Build in an existing container

Copy/mount this folder at `/root/ros2_ws/src/demo_1f`, then:

```bash
cd /root/ros2_ws/src/demo_1f
./scripts/build_workspace.sh
source /root/ros2_ws/install/setup.bash
```

The explicit package paths in this script are required: colcon normally stops descending
when it finds the parent `demo_1f/package.xml`. This builds the four imported packages too.
No Docker image rebuild is necessary. The Dockerfile has also been updated for future builds.

The DYNAMIXEL detector uses PyTorch/Ultralytics, independently of the existing cup detector's
TensorRT/OpenCV backend. Check dependencies in the Python interpreter used for pick nodes:

```bash
/opt/demo-pick-venv/bin/python3 scripts/check_pick_dependencies.py
```

If torch/Ultralytics are absent on an x86 PC, create an isolated CPU inference environment:

```bash
./scripts/setup_pick_inference.sh
# Created /opt/demo-pick-venv; existing ROS/OpenCV installation is preserved.
```

On Jetson, use a Python environment with torch/torchvision compatible with that JetPack.
The setup script reuses a working installation and does not replace it with desktop wheels.
It stops if compatible torch/torchvision are missing on ARM. Use `pick_python:=` below to
select that interpreter. It must also have access to ROS Python packages and the built overlay.

The Dockerfile prepares `/opt/demo-pick-venv` during image build. The first build
needs internet access for Python packages, using host networking; ROS compilation
runs offline. The setup script reuses the base image's GPU torch/torchvision.
Container startup performs no package installation and does not mount over the
image's Python environment. Compose uses NVIDIA runtime, host networking/IPC and
`restart: unless-stopped`. The idle container returns after a host reboot unless
it was explicitly stopped; robot motion still requires an action request.

## Start

Bringup and head/right/left wrist camera topics must already be available. These launches
never open camera devices or start a second hardware bringup.

```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
source /root/ros2_ws/install/setup.bash
ros2 launch demo_1f full_demo.launch.py use_sim_time:=false
```

The image's inference environment is selected by default. To use another one, append:

```text
pick_python:=/path/to/python3
```

If the cup viewer/perception is already running, append `start_cup_perception:=false`.
If move_group is already running, append `start_moveit:=false`. Do not keep a previous
handover_demo_node running at the same time. Cup viewer uses port 8765; the new DYNAMIXEL
viewer uses 8767. Original camera topics are in ffw_vision/config/cameras.yaml; the right
wrist needs RGB, camera_info and `aligned_depth_to_color/image_raw` containing metric depth.

Nothing moves on launch. In another terminal with the same environment:

```bash
ros2 run demo_1f action_runner.py
```

- `d`: full demo.
- `1`: initial pose; `2`: right-hand cell pick from the current pose.
- `3`: right-to-left handover; `4`: left cup approach.
- `5`: left wrist refinement and release; `6`: both-hand withdrawal.
- `r`: resume a recoverable checkpoint; after `2`, continue with handover.
- `c`: cancel the goal sent by this runner and wait for the final result.
- `p`: status; `q`: exit, waiting for cancellation if necessary.

There are no extra menus, direction selectors, previews, cup-pick entries or server launch controls.
Start `full_demo.launch.py` separately. Exiting the runner does not stop that server.

Equivalent direct action:

```bash
ros2 action send_goal /run_demo demo_1f/action/RunDemo \
  '{command: start, cup_id: cup}' --feedback
# {command: pick, cup_id: cup} or {command: resume, cup_id: cup}
```

## Sequence and recovery

1. INITIALIZE_ROBOT: open the grippers, then move both arms, lift (-0.1 m) and head
   ([0.69, 0] rad) together in one MoveIt plan to the basket observation pose.
   There is no folded-arm waypoint or intermediate lift height.
   Targets in `config/demo.yaml` follow the ai_worker custom initial positions;
   the original head target 0.76 rad is replaced with 0.69 rad within the SG2 URDF limit.
   All movements use MoveIt and wait for completion. Launching the server performs no motion.
2. PICK_DYNAMIXEL: acquire fresh
   images, choose a collision-free grasp, open, approach, close, lift and test gripper position.
3. HANDOVER: right_to_left sequence, once. After release, move each TCP outward
   5 cm from its own transfer position, preserving height and orientation.
   On completion, continue directly to cup approach using the saved head track.
4. APPROACH_CUP: snapshot the latest saved head BODY location → left TCP observation pose.
   RobotContext tracks the head cup continuously, including before handover. Three valid,
   transformed samples confirm each update (under 1 s apart and 2 cm position spread).
   Detection loss keeps the last confirmed location and its real timestamps for at most
   `approach.max_age` (30 s). No post-request head observation is required. Wrist RIM data
   remains separate. Input freshness/TF limits are unchanged. Each approach freezes its
   own goal and collision model while background tracking continues for future requests.
   This memory is reset with the node and assumes a stationary robot base.
5. REFINE_CUP: wrist rim observation, pitch search, correction/lowering and release.
   Pitch speed is 0.06. After at least three fresh unconfirmed frames and one second
   without stability progress, increase pitch by 5 degrees up to 75 degrees.
   Advancing stability keeps the current angle. The overall 15-second search budget
   includes observation and pitch planning/execution; an active pitch is cancelled
   and its stop confirmed on expiry. Missing camera/TF input never triggers a tilt.
6. WITHDRAW_CUP: run left then right sequentially (`arm: both`). For each hand,
   lift its measured TCP 10 cm in world Z and retreat 8 cm along +TCP Z
   (opposite hand-forward), preserving orientation. Each withdrawal uses collision-checked LIN;
   failure/cancellation stops the sequence. A downward retreat is rejected.
7. RETURN_INITIAL: reuse the initial-pose module after successful withdrawal; do not repeat pick.

Menu `5` / `/refine_cup` stops after release. Menu `6` / `/withdraw_cup` only withdraws
both hands sequentially, with no perception or gripper commands.
The action also accepts `arm: left` or `arm: right`; preview each hand separately. Manual sequence: `5` → `6` → `1`.
The full demo calls withdrawal and then returns to initial pose automatically.
The full cup-side tabletop collision model is not added by this withdrawal sequence.

`initial` runs initialization only; `start` initializes before the full demo.
`pick` starts directly from the current pose. Retrying pick or resuming a later
stage does not repeat initialization; the pick backend skips its old right-arm-only HOME move.

`config/demo.yaml` defaults to two additional attempts, one second apart. Pick perception /
pre-grasp planning failures retry only if no grasp execution started. Cup observation failures
retry their own stage, never restart pick or handover. The existing wrist module performs
its pitch-search recovery. Arbitrary controller/execution faults do not automatically replay
motions, especially gripper operations. Unknown stop/scene results latch a fault and block
further goals.

After retry exhaustion, a result with `can_resume: true` retains the failed stage. `r` repeats
that stage after the observation issue has been resolved. Resume checkpoints are in memory;
restarting the server clears them. A manual motion invalidates the saved checkpoint.
After an unrecoverable failure, inspect/recover the robot using the individual motions.
Once the object is removed and both grippers are ready for a new pick, explicitly clear
only the software record:

```bash
ros2 service call /handover_demo/reset_demo std_srvs/srv/Trigger '{}'
```

This service commands no motion, does not clear an unknown-stop fault and does not assert
that the grippers are empty. Restart after resolving a latched stop fault.

## Geometry and verification limits

The demo builds `urdf/sg2_extended_grippers.urdf` from the installed SG2 description.
Both hands use `gripper_l_tcp` / `gripper_r_tcp`, fixed 45 mm along the original
end-effector frame's local -Z. The handover launch publishes only these two added
static transforms; existing bringup/camera transforms are preserved. All demo
motions and both MoveIt/pick IK use this shared model. Distal finger collision
geometry includes an approximate layered taper, following the actual finger joints.
The original handover Cartesian target positions/orientations are preserved by
re-solving its joint targets for the extended TCPs. Timing/gripper commands stay the same.

Pick's `tip_below_end_effector` is now zero: the new TCP already includes the
45 mm extension. Its legacy attached finger boxes are removed, because the URDF
owns finger collisions for both hands throughout the demo. Pick reads the same
model file as MoveIt rather than the unextended bringup description topic.
Original camera/arm bias values remain robot-specific; override
`dynamixel_calibration:=/path/to/robot_calibration.yaml` if they differ.

Basket/table objects remain in MoveIt's scene; cup scene updates preserve these
obstacles rather than erase them. Printed-finger shape is approximate, not a measured CAD mesh.

A pick grip is inferred from fresh gripper position after closing and lifting. The existing
handover has no independent transfer sensor (`LEFT_ASSUMED`). Release success means the
motion and opening completed (`RELEASED_UNVERIFIED`); object landing is not verified.
