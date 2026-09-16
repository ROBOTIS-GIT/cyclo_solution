# FFW cuMotion models

These XRDF files define the cuMotion c-space and collision spheres for the FFW
lift, arms, grippers, and base. Wheel joints may exist in the URDF, but are not
part of the XRDF c-space or collision model.

The group router selects the model for `arm_l`, `arm_r`, `both_arms`, or
`wholebody`. Joints outside the selected group are fixed at their latest
positions; XRDF defaults are used only when a live value is unavailable.

The collision spheres are derived from the supplied `ai_worker.xrdf`.
