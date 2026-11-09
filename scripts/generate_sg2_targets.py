#!/usr/bin/env python3
"""Generate reproducible empty-hand targets from the SG2 URDF, without ROS motion.

Needs numpy/scipy/PyYAML (available in the AI Worker image). Coordinates are in
arm_base_link, so the joint targets remain usable at different lift heights.
Collision/path checks are performed separately with MoveIt's full robot model.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

RETREAT_DISTANCE = 0.05  # Each TCP moves outward from its actual transfer pose.


class RosDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True  # ROS parameter files do not support YAML aliases.


class Arm:
    def __init__(self, root, side, tcp_joint=None):
        self.side = side
        suffix = "l" if side == "left" else "r"
        if tcp_joint is None and root.find(f"joint[@name='gripper_{suffix}_tcp_joint']") is not None:
            tcp_joint = f'gripper_{suffix}_tcp_joint'
        self.names = [f"arm_{suffix}_joint{i}" for i in range(1, 8)]
        self.joints = []
        self.bounds = []
        fixed = [f"end_effector_{suffix}_joint"] + ([tcp_joint] if tcp_joint else [])
        for name in self.names + fixed:
            joint = root.find(f"joint[@name='{name}']")
            origin = joint.find("origin")
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
            matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            axis = joint.find("axis")
            self.joints.append(
                (matrix, None if axis is None else np.fromstring(axis.get("xyz"), sep=" "))
            )
            limit = joint.find("limit")
            if limit is not None:
                self.bounds.append(
                    (float(limit.get("lower")) + 0.025, float(limit.get("upper")) - 0.025)
                )
        self.lower, self.upper = np.array(self.bounds).T

    def fk(self, q, count=None):
        transform = np.eye(4)
        for i, (origin, axis) in enumerate(self.joints[:count]):
            transform = transform @ origin
            if axis is not None:
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(axis * q[i]).as_matrix()
                transform = transform @ rotation
        return transform

    def solve(self, position, rotation, seed):
        seed = np.clip(seed, self.lower + 0.001, self.upper - 0.001)

        def residual(q):
            transform = self.fk(q)
            return np.r_[
                10 * (transform[:3, 3] - position),
                Rotation.from_matrix(rotation.T @ transform[:3, :3]).as_rotvec(),
                0.0001 * (q - seed),
            ]

        solution = least_squares(
            residual,
            seed,
            bounds=(self.lower, self.upper),
            max_nfev=1000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        q = solution.x
        if np.linalg.norm(residual(q)[:3]) > 0.001 or np.linalg.norm(residual(q)[3:6]) > 0.001:
            raise RuntimeError(
                f"{self.side}: target not reachable: {position}, error={residual(q)[:6]}"
            )
        return np.round(q, 8).tolist()


def generate(urdf, output, report):
    raw = Path(urdf).read_bytes()
    root = ET.fromstring(raw)
    arms = {side: Arm(root, side) for side in ("left", "right")}
    # Neutral elbow (joint4) level: -0.300 m below arm_base_link in this URDF.
    elbow_z = float(arms["left"].fk(np.zeros(7), 4)[2, 3])
    ready, ready_cross, approach, present, retreat, giver_retreat, poses = {}, {}, {}, {}, {}, {}, {}
    for side, arm in arms.items():
        sign = 1 if side == "left" else -1
        # The TCP extends along local -Z. Aim that axis toward the opposite hand.
        rotation = Rotation.from_euler("x", -sign * np.pi / 2).as_matrix()
        position = np.array([0.33, sign * 0.14, elbow_z])
        seed = np.array([-0.35, sign * 0.35, sign * 0.2, -1.65, sign * 1.25, 0.2, -sign * 0.6])
        ready[side] = arm.solve(position, rotation, seed)
        # The receiver advances 20 mm past the transfer point. Only the receiver turns
        # its jaw plane by 90 degrees; the giver keeps its presenting orientation.
        cross_rotation = rotation @ Rotation.from_euler("z", -sign * np.pi / 2).as_matrix()
        cross_position = position.copy()
        cross_position[1] = 0.0
        present[side] = arm.solve(cross_position, rotation, ready[side])
        receiver_position = cross_position - 0.02 * cross_rotation[:, 2]
        approach[side] = arm.solve(receiver_position, cross_rotation, ready[side])
        # Pre-rotate the receiver while the TCPs are still 0.28 m apart. Seed
        # from its transfer pose to keep the same arm configuration on approach.
        ready_cross[side] = arm.solve(position, cross_rotation, approach[side])
        # Both arms move outward; the receiver preserves its crossed orientation.
        outward = receiver_position.copy()
        outward[1] += sign * RETREAT_DISTANCE
        retreat[side] = arm.solve(outward, cross_rotation, approach[side])
        giver_outward = cross_position.copy()
        giver_outward[1] += sign * RETREAT_DISTANCE
        giver_retreat[side] = arm.solve(giver_outward, rotation, present[side])
        poses[side] = {}
        for name, q in [
            ("ready", ready[side]),
            ("ready_cross", ready_cross[side]),
            ("present", present[side]),
            ("approach", approach[side]),
            ("retreat", retreat[side]),
            ("giver_retreat", giver_retreat[side]),
        ]:
            t = arm.fk(q)
            poses[side][name] = {
                "joints": q,
                "position": t[:3, 3].tolist(),
                "quaternion_xyzw": Rotation.from_matrix(t[:3, :3]).as_quat().tolist(),
                "tcp_forward": (-t[:3, 2]).tolist(),
            }

    config = {
        "arms": {
            side: {"tcp": f"gripper_{suffix}_tcp" if root.find(f"joint[@name='gripper_{suffix}_tcp_joint']") is not None
                   else f"end_effector_{suffix}_link"}
            for side, suffix in (("left", "l"), ("right", "r"))
        },
        "planning_frame": "base_link",
        "joint_states_topic": "/joint_states",
        "move_group_action": "/move_action",
        "handover_action": "/handover",
        "planning_pipeline": "pilz_industrial_motion_planner",
        "coordinated_group": "handover_arms",
        "coordinated_pipeline": "ompl",
        "coordinated_planner": "RRTConnectkConfigDefault",
        "gripper_transport": "action",
        "velocity_scale": 0.15,
        "acceleration_scale": 0.15,
        "planning_time": 5.0,
        "ompl_planning_attempts": 3,
        "state_max_age": 1.0,
        "tf_max_age": 1.0,
        "require_tcp_tf": True,
        "settle_time": 0.25,
        "stopped_velocity": 0.03,
        "cancel_timeout": 5.0,
        "gripper_stop_mode": "position_window",
        "require_passive_gripper_stationary": False,
        "gripper_stop_window": 0.5,
        "gripper_completion_window": 0.25,
        "gripper_settle_time": 0.1,
        "gripper_stop_position_tolerance": 0.004,
        "gripper_stop_max_gap": 0.1,
        "gripper_goal_tolerance": 0.004,
        "smooth_motion": True,
        "path_simplification_tolerance": 0.002,
        "trajectory_sample_period": 0.01,
        "continuous_start_tolerance": 0.01,
        "motion_sequences": {},
    }
    # URDF gripper fingers spread at 0 rad and close toward the center at ~1 rad.
    # At coincident TCPs, closing BOTH crossed grippers to 0.95 rad makes their
    # fingertips collide. Use 0.8 rad for the empty-hand transfer demonstration.
    for giver, receiver in [("left", "right"), ("right", "left")]:
        sequence = {
            "steps": [
                "face_each_other",
                "open_approach",
                "receiver_close",
                "giver_release",
                "retreat_together",
            ],
            "face_each_other": {
                "type": "dual_joint",
                "blend_with_next": True,
                "velocity_scale": 0.40,
                "acceleration_scale": 0.40,
                "left_target": ready_cross["left"] if receiver == "left" else ready["left"],
                "right_target": ready_cross["right"] if receiver == "right" else ready["right"],
                "timeout": 45.0,
            },
            "open_approach": {
                "type": "dual_joint_gripper",
                "velocity_scale": 0.40,
                "acceleration_scale": 0.40,
                "arm": "receiver",
                "left_target": approach["left"] if receiver == "left" else present["left"],
                "right_target": approach["right"] if receiver == "right" else present["right"],
                "gripper_target": [0.0],
                "tolerance": float(np.deg2rad(1.0)),
                "planning_tolerance": 0.0005,
                "timeout": 45.0,
            },
            "receiver_close": {
                "type": "gripper",
                "gripper_wait_only": True,
                "arm": "receiver",
                "target": [0.8],
                "duration": 0.6,
                "velocity_scale": 1.0,
                "acceleration_scale": 1.0,
                "timeout": 12.0,
            },
            "giver_release": {
                "type": "gripper",
                "arm": "giver",
                "target": [0.0],
                "duration": 0.6,
                "velocity_scale": 1.0,
                "acceleration_scale": 1.0,
                "timeout": 12.0,
            },
            "retreat_together": {
                "type": "dual_joint",
                "timeout": 45.0,
                "velocity_scale": 0.40,
                "acceleration_scale": 0.40,
            },
        }
        for side in arms:
            q = retreat[side] if side == receiver else giver_retreat[side]
            sequence["retreat_together"][side + "_target"] = q
        config["motion_sequences"][giver + "_to_" + receiver] = sequence
    Path(output).write_text(
        "# Generated from SG2 URDF by scripts/generate_sg2_targets.py.\n"
        "# Empty-hand simulation preset. TCP coordinates are referenced to elbow height.\n"
        "# joint arrays follow arm_[lr]_joint1 ... joint7; angles are radians.\n"
        + yaml.dump(
            {"handover_demo": {"ros__parameters": config}}, Dumper=RosDumper, sort_keys=False
        )
    )
    Path(report).write_text(
        json.dumps(
            {
                "urdf_sha256": hashlib.sha256(raw).hexdigest(),
                "frame": "arm_base_link",
                "elbow_height": elbow_z,
                "ready_tcp_gap": 0.28,
                "approach_tcp_gap": 0.02,
                "receiver_extra_advance_m": 0.02,
                "retreat_distance_per_arm": RETREAT_DISTANCE,
                "transfer_point": [0.33, 0.0, elbow_z],
                "poses": poses,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "elbow_height": elbow_z,
                "ready": ready,
                "ready_cross": ready_cross,
                "approach": approach,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    generate(args.urdf, args.output, args.report)
