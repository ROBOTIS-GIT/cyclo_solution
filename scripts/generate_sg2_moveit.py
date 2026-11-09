#!/usr/bin/env python3
"""Build the demo-owned coordinated group without changing robot bringup files."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def generate(robot_source, output):
    source, output = Path(robot_source), Path(output)
    robot = ET.parse(
        source / "ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
    ).getroot()
    semantic = ET.parse(source / "ffw_moveit_config/config/ffw.srdf").getroot()
    semantic.set("name", robot.get("name"))
    links = {link.get("name") for link in robot.findall("link")}

    def arm_of(link):
        if link.startswith(("arm_l_", "gripper_l_", "end_effector_l", "camera_left")):
            return "left"
        if link.startswith(("arm_r_", "gripper_r_", "end_effector_r", "camera_right")):
            return "right"
        return None

    removed = 0
    for pair in list(semantic.findall("disable_collisions")):
        first, second = pair.get("link1"), pair.get("link2")
        # The upstream SRDF excludes many opposite-arm pairs as "Never".
        # Handover explicitly brings those arms together: restore those checks.
        if (
            first not in links
            or second not in links
            or (arm_of(first) and arm_of(second) and arm_of(first) != arm_of(second))
        ):
            semantic.remove(pair)
            removed += 1
    for end_effector in list(semantic.findall("end_effector")):
        semantic.remove(end_effector)  # upstream points each arm back at itself
    for side in ('l', 'r'):
        ET.SubElement(semantic.find(f"group[@name='arm_{side}']"), 'joint',
                      name=f'gripper_{side}_tcp_joint')
    group = ET.SubElement(semantic, "group", name="handover_arms")
    for side in ["l", "r"]:
        ET.SubElement(group, "group", name="arm_" + side)
        # Group-scoped state updates only update mimic joints inside the group.
        # Include passive fingers so OMPL checks their planned opening, rather
        # than leaving them at the measured start-state angle.
        for index in range(1, 5):
            ET.SubElement(group, "joint", name=f"gripper_{side}_joint{index}")
    initial = ET.SubElement(semantic, "group", name="initial_pose")
    for name in ("arm_l", "arm_r", "head", "lift"):
        ET.SubElement(initial, "group", name=name)
    ET.indent(semantic)
    (output / "sg2_handover.srdf").write_text(ET.tostring(semantic, encoding="unicode") + "\n")
    limits = {"joint_limits": {}}
    controllers = {
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": ["arm_l_controller", "arm_r_controller"]
        },
    }
    for group, joints, velocity, acceleration in (
        ("head", ["head_joint1", "head_joint2"], 0.5, 0.5),
        ("lift", ["lift_joint"], 0.05, 0.1),
    ):
        controller = f"{group}_controller"
        controllers["moveit_simple_controller_manager"]["controller_names"].append(controller)
        controllers["moveit_simple_controller_manager"][controller] = {
            "type": "FollowJointTrajectory", "action_ns": "follow_joint_trajectory",
            "default": True, "joints": joints,
        }
        for name in joints:
            limits["joint_limits"][name] = {
                "has_velocity_limits": True, "max_velocity": velocity,
                "has_acceleration_limits": True, "max_acceleration": acceleration,
            }
    for side in ["l", "r"]:
        joints = [f"arm_{side}_joint{i}" for i in range(1, 8)] + [f"gripper_{side}_joint1"]
        controllers["moveit_simple_controller_manager"][f"arm_{side}_controller"] = {
            "type": "FollowJointTrajectory",
            "action_ns": "follow_joint_trajectory",
            "default": True,
            "joints": joints,
        }
        for name in joints:
            limits["joint_limits"][name] = {
                "has_velocity_limits": True,
                "max_velocity": 2.0,
                "has_acceleration_limits": True,
                "max_acceleration": 2.0,
                "has_deceleration_limits": True,
                "max_deceleration": -2.0,
                "has_jerk_limits": True,
                "max_jerk": 20.0 if name.startswith("gripper_") else 4.0,
            }
    (output / "sg2_joint_limits.yaml").write_text(yaml.safe_dump(limits, sort_keys=False))
    (output / "sg2_moveit_controllers.yaml").write_text(
        yaml.safe_dump(controllers, sort_keys=False)
    )
    print(
        f"Created handover_arms (16 active joints + 6 mimic joints); restored/removed {removed} collision exclusions."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate(args.robot_source, args.output)
