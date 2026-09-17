# FFW cuMotion models

These XRDF files define the cuMotion c-space and collision spheres for the FFW
lift, arms, grippers, and base. Wheel joints may exist in the URDF, but are not
part of the XRDF c-space or collision model.

## Planning models

The group router selects the model for `arm_l`, `arm_r`, `both_arms`, or
`wholebody`. Joints outside the selected group are fixed at their latest
positions; XRDF defaults are used only when a live value is unavailable.

- `ffw_left_arm.xrdf`: left-arm joints
- `ffw_right_arm.xrdf`: right-arm joints
- `ffw_bimanual.xrdf`: both arms
- `ffw_wholebody.xrdf`: lift and both arms

Only the joints in each model's c-space are planning variables. For example,
the head and grippers must not be added to a planning XRDF merely to improve
robot segmentation, because cuMotion would then treat them as part of the
planning problem even though they are not in the selected MoveIt group.

## Segmentation model

`ffw_segmentation.xrdf` is used only by the robot segmenters that remove the
robot from camera depth images before the images are integrated by Nvblox. It
is independent of the currently selected planning group and includes every
moving joint needed to reproduce the visible robot pose:

- lift
- both arms
- both head joints
- both gripper master joints

In this model, the c-space joint list identifies the current positions that the
segmenter reads from `/joint_states`; these joints are not being planned or
optimized. Including the head and gripper joints keeps their collision spheres
aligned with the physical robot as the head moves or the grippers open and
close. If they were omitted, their default XRDF positions could be used instead,
causing the depth mask to disagree with the real robot. Robot pixels could then
remain in the Nvblox map as false obstacles, or nearby scene geometry could be
removed incorrectly.

Compared with `ffw_wholebody.xrdf`, `ffw_segmentation.xrdf` uses the same
collision spheres, default joint positions, and self-collision settings. Its
important difference is the additional head and gripper joints in the c-space.

When an object is attached, the group router also overlays its collision
spheres onto the segmentation model. This allows the carried object to be
removed from the depth stream instead of being integrated into the Nvblox map
as an obstacle.

The collision spheres are derived from the supplied `ai_worker.xrdf`.
