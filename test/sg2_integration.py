"""Real MoveIt + real JTCs on GenericSystem. Run ONLY in the isolated test container.

This is a controller/plan integration test, not Gazebo physics or grasp validation.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from control_msgs.msg import JointTrajectoryControllerState
from demo_1f.action import Handover
from demo_1f.msg import DemoStatus
from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPlannerParams
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory


def main(urdf_path, legacy=False, report=None, approach=False, approach_only=False):
    if os.environ.get("DEMO_ISOLATED_TEST") != "1":
        raise RuntimeError(
            "Use --network none and DEMO_ISOLATED_TEST=1; this test creates mock controllers."
        )
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        initial_pose = {}
        if approach_only:
            preset = yaml.safe_load(
                (Path(get_package_share_directory("demo_1f")) / "config/handover.yaml").read_text()
            )["handover_demo"]["ros__parameters"]
            retreat = preset["motion_sequences"]["right_to_left"]["retreat_together"]
            for side, key in [("l", "left"), ("r", "right")]:
                initial_pose.update(
                    {
                        f"arm_{side}_joint{i}": value
                        for i, value in enumerate(retreat[key + "_target"], 1)
                    }
                )
            initial_pose.update({"gripper_l_joint1": 0.8, "gripper_r_joint1": 0.0})
        root = ET.parse(urdf_path).getroot()
        for tag in ["ros2_control", "gazebo"]:
            for element in list(root.findall(tag)):
                root.remove(element)
        system = ET.SubElement(root, "ros2_control", name="DemoMock", type="system")
        hardware = ET.SubElement(system, "hardware")
        ET.SubElement(hardware, "plugin").text = "mock_components/GenericSystem"
        for joint in root.findall("joint"):
            if joint.get("type") not in ["revolute", "prismatic", "continuous"]:
                continue
            if joint.find("mimic") is not None:
                continue
            entry = ET.SubElement(system, "joint", name=joint.get("name"))
            ET.SubElement(entry, "command_interface", name="position")
            for interface in ["position", "velocity"]:
                state = ET.SubElement(entry, "state_interface", name=interface)
                initial = (
                    0.95
                    if interface == "position" and joint.get("name").startswith("gripper_")
                    else 0.0
                )
                if interface == "position" and joint.get("name") == "arm_l_joint2":
                    initial = -6.77264e-14
                if interface == "position" and joint.get("name") == "arm_r_joint2":
                    initial = 7.24182e-14
                if interface == "position" and joint.get("name") in initial_pose:
                    initial = initial_pose[joint.get("name")]
                ET.SubElement(state, "param", name="initial_value").text = str(initial)
        description = ET.tostring(root, encoding="unicode")
        parameters = work / "robot.yaml"
        parameters.write_text(
            yaml.safe_dump(
                {
                    "/**": {
                        "ros__parameters": {"robot_description": description, "use_sim_time": False}
                    }
                }
            )
        )
        controllers = {
            "controller_manager": {
                "ros__parameters": {
                    "update_rate": 100,
                    "use_sim_time": False,
                    "joint_state_broadcaster": {
                        "type": "joint_state_broadcaster/JointStateBroadcaster"
                    },
                }
            }
        }
        for side in ["l", "r"]:
            name = f"arm_{side}_controller"
            controllers["controller_manager"]["ros__parameters"][name] = {
                "type": "joint_trajectory_controller/JointTrajectoryController"
            }
            controllers[name] = {
                "ros__parameters": {
                    "joints": [f"arm_{side}_joint{i}" for i in range(1, 8)]
                    + [f"gripper_{side}_joint1"],
                    "command_interfaces": ["position"],
                    "state_interfaces": ["position", "velocity"],
                    "allow_partial_joints_goal": True,
                }
            }
        controller_file = work / "controllers.yaml"
        controller_file.write_text(yaml.safe_dump(controllers))
        processes, logs = [], []

        def start(name, command):
            log = (work / (name + ".log")).open("w+")
            logs.append((name, log))
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            processes.append(process)
            return process

        rclpy.init()
        node = rclpy.create_node("sg2_integration_test")
        tf_buffer = Buffer()
        # Retain the listener for the duration of the test.
        _tf_listener = TransformListener(tf_buffer, node)
        statuses = []
        joint_states = []
        planned_paths = []
        controller_states = {side: [] for side in ["l", "r"]}
        node.create_subscription(
            JointTrajectory, "/handover_demo/planned_trajectory", planned_paths.append, 10
        )
        for side in ["l", "r"]:
            node.create_subscription(
                JointTrajectoryControllerState,
                f"/arm_{side}_controller/controller_state",
                lambda message, side=side: controller_states[side].append(message),
                10,
            )
        preset = yaml.safe_load(
            (Path(get_package_share_directory("demo_1f")) / "config/handover.yaml").read_text()
        )["handover_demo"]["ros__parameters"]
        if legacy:
            preset["smooth_motion"] = False
            for sequence in preset["motion_sequences"].values():
                for step in sequence["steps"]:
                    options = sequence[step]
                    options["blend_with_next"] = False
                    options["velocity_scale"] = 0.0
                    options["acceleration_scale"] = 0.0
                    if options["type"] == "gripper":
                        options["duration"] = 2.0
        demo_config = work / "handover.yaml"
        demo_config.write_text(yaml.safe_dump({"handover_demo": {"ros__parameters": preset}}))
        node.create_subscription(JointState, "/joint_states", joint_states.append, 10)
        node.create_subscription(
            DemoStatus,
            "/handover_demo/status",
            statuses.append,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        client = ActionClient(node, Handover, "/handover")
        move = ActionClient(node, MoveGroup, "/move_action")
        planner = node.create_client(GetPlannerParams, "/get_planner_params")

        def wait(condition, timeout=30):
            deadline = time.monotonic() + timeout
            while not condition() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            assert condition(), (
                "Timed out: handover_ready="
                + str(client.server_is_ready())
                + ", move_ready="
                + str(move.server_is_ready())
                + "\n"
                + str(statuses[-1] if statuses else "no status")
                + "\n"
                + str(joint_states[-1] if joint_states else "no joint states")
            )

        success = False
        try:
            start(
                "rsp",
                [
                    "ros2",
                    "run",
                    "robot_state_publisher",
                    "robot_state_publisher",
                    "--ros-args",
                    "--params-file",
                    str(parameters),
                ],
            )
            start(
                "controllers",
                [
                    "ros2",
                    "run",
                    "controller_manager",
                    "ros2_control_node",
                    "--ros-args",
                    "--params-file",
                    str(parameters),
                    "--params-file",
                    str(controller_file),
                ],
            )
            spawner = start(
                "spawner",
                [
                    "ros2",
                    "run",
                    "controller_manager",
                    "spawner",
                    "joint_state_broadcaster",
                    "arm_l_controller",
                    "arm_r_controller",
                    "--controller-manager-timeout",
                    "30",
                ],
            )
            wait(lambda: spawner.poll() is not None, 40)
            assert spawner.returncode == 0, "Controller spawn failed"
            start(
                "demo",
                [
                    "ros2",
                    "launch",
                    "demo_1f",
                    "handover.launch.py",
                    "use_sim_time:=false",
                    "start_moveit:=true",
                    "description_file:=" + str(urdf_path),
                    "config:=" + str(demo_config),
                ],
            )
            wait(
                lambda: client.server_is_ready()
                and move.server_is_ready()
                and bool(statuses)
                and statuses[-1].robot_state_fresh
            )
            # Successful motion alone is insufficient: OMPL can fall back when
            # a requested config is missing. Check its actual registered config.
            wait(planner.service_is_ready)
            request = GetPlannerParams.Request()
            request.pipeline_id = "ompl"
            request.planner_config = "RRTConnectkConfigDefault"
            request.group = "handover_arms"
            future = planner.call_async(request)
            wait(future.done)
            parameters = future.result().params
            registered = dict(zip(parameters.keys, parameters.values))
            assert registered.get("type") == "geometric::RRTConnect", registered
            # Let Zenoh action endpoint discovery settle before timing either preset.
            ready_until = time.monotonic() + 1.0
            while time.monotonic() < ready_until:
                rclpy.spin_once(node, timeout_sec=0.05)
            print(
                "PASS: handover_arms/RRTConnectkConfigDefault registered as geometric::RRTConnect",
                flush=True,
            )
            outcomes = []
            for direction in [] if approach_only else ["left_to_right", "right_to_left"]:
                path_start = len(planned_paths)
                reference_starts = {side: len(controller_states[side]) for side in ["l", "r"]}
                run_started = time.monotonic()
                phases = []
                approach_checks = []
                ready_joints = {}

                def feedback(message):
                    phase = message.feedback.phase
                    if phase not in phases:
                        phases.append(phase)
                        print(direction + ": " + phase, flush=True)
                    if phase == "open_approach" and not ready_joints:
                        ready_joints.update(zip(joint_states[-1].name, joint_states[-1].position))
                    if phase == "receiver_close" and not approach_checks:
                        for side, key in [("l", "left"), ("r", "right")]:
                            target = preset["motion_sequences"][direction]["face_each_other"][
                                key + "_target"
                            ]
                            samples = controller_states[side][reference_starts[side] :]
                            names = [f"arm_{side}_joint{i}" for i in range(1, 8)]

                            def ready_error(sample):
                                measured = dict(zip(sample.joint_names, sample.reference.positions))
                                return max(
                                    abs(measured[name] - target[i]) for i, name in enumerate(names)
                                )

                            assert samples, "No controller reference feedback"
                            nearest = min(samples, key=ready_error)
                            assert ready_error(nearest) < 0.03, (side, ready_error(nearest))
                            if not legacy:
                                assert (
                                    max(abs(value) for value in nearest.reference.velocities)
                                    > 0.005
                                ), ("Stopped at ready waypoint", side)
                        transforms = [
                            tf_buffer.lookup_transform(
                                "arm_base_link", f"gripper_{side}_tcp", rclpy.time.Time()
                            )
                            for side in ["l", "r"]
                        ]
                        tcp = [
                            [getattr(t.transform.translation, axis) for axis in ["x", "y", "z"]]
                            for t in transforms
                        ]
                        gap = math.dist(*tcp)
                        assert gap < 0.005, ("TCPs did not meet", tcp, gap)
                        measured = dict(zip(joint_states[-1].name, joint_states[-1].position))
                        assert ready_joints, "Missing ready-state snapshot"
                        for side in ["l", "r"]:
                            names = [f"arm_{side}_joint{i}" for i in range(1, 8)]
                            assert max(abs(measured[n] - ready_joints[n]) for n in names) > 0.01, (
                                "Arm did not move during approach",
                                side,
                            )
                        approach_checks.append(gap)
                        print(
                            f"PASS: {direction} both arms approached; measured TCP gap {gap:.6f} m",
                            flush=True,
                        )

                goal = Handover.Goal()
                goal.direction = direction
                future = client.send_goal_async(goal, feedback_callback=feedback)
                wait(future.done)
                handle = future.result()
                assert handle.accepted, "Handover rejected"
                result = handle.get_result_async()
                wait(result.done, 200)
                outcome = result.result().result
                assert outcome.sequence_completed, outcome.error_code + ": " + outcome.message
                assert not outcome.handover_confirmed
                assert outcome.last_completed_step == "retreat_together"
                motion_phases = [
                    phase for phase in phases if phase not in ("CHECK_READY", "BETWEEN_STEPS")
                ]
                assert motion_phases == [
                    "face_each_other",
                    "open_approach",
                    "receiver_close",
                    "giver_release",
                    "retreat_together",
                ], phases
                assert approach_checks, "Approach geometry was not observed"
                paths = planned_paths[path_start:]
                assert len(paths) == (0 if legacy else 2), (
                    "Expected one continuous approach and one retreat",
                    len(paths),
                )
                for path in paths:
                    assert len(path.joint_names) == 16
                    assert all(abs(value) < 1e-8 for value in path.points[0].velocities)
                    assert all(abs(value) < 1e-8 for value in path.points[-1].velocities)
                    assert all(abs(value) < 1e-8 for value in path.points[-1].accelerations)
                    times = [
                        point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                        for point in path.points
                    ]
                    assert all(before < after for before, after in zip(times, times[1:]))
                elapsed = time.monotonic() - run_started
                print(
                    f"PASS: {direction} profile={'legacy' if legacy else 'smooth'}, transfer stop, {len(paths)} execution paths, cycle {elapsed:.3f} s",
                    flush=True,
                )
                outcomes.append(
                    {
                        "direction": direction,
                        "sequence_completed": True,
                        "phases": phases,
                        "cycle_seconds": elapsed,
                    }
                )
            approach_outcome = None
            if approach or approach_only:
                from approach_integration import run_approach

                approach_outcome = run_approach(
                    node, tf_buffer, root, work, processes, start, wait, urdf_path, demo_config
                )
            for name, log in logs:
                if name == "demo":
                    log.seek(0)
                    demo_log = log.read()
                    assert "Could not find the planner configuration" not in demo_log
                    if not approach_only:
                        assert "Snapped 2 start-state values to model bounds" in demo_log
            print("PASS: " + json.dumps(outcomes), flush=True)
            if report:
                Path(report).write_text(
                    json.dumps(
                        {
                            "profile": "legacy" if legacy else "smooth",
                            "outcomes": outcomes,
                            "approach": approach_outcome,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            success = True
        finally:
            client.destroy()
            move.destroy()
            node.destroy_node()
            rclpy.shutdown()
            for process in reversed(processes):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for name, log in logs:
                if not success:
                    log.seek(0)
                    print(name + ":\n" + log.read()[-10000:], flush=True)
                log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument(
        "--legacy", action="store_true", help="Benchmark the previous stop-at-each-step preset"
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--approach", action="store_true")
    parser.add_argument(
        "--approach-only",
        action="store_true",
        help="Initialize mock hardware at the fixed post-handover pose and test cup approach independently",
    )
    args = parser.parse_args()
    main(args.urdf, args.legacy, args.report, args.approach, args.approach_only)
