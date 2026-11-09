"""Check startup without robot feedback in an isolated ROS domain/container."""

import os
import signal
import subprocess
import tempfile
import time

import rclpy
from demo_1f.action import ApproachCup, Handover
from demo_1f.msg import DemoStatus
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory


def main():
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(
            [
                "ros2",
                "launch",
                "demo_1f",
                "approach_cup.launch.py",
                "start_moveit:=false",
                "cup_topic:=/perception_test_head/cup",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        rclpy.init()
        node = rclpy.create_node("demo_smoke_inspector")
        statuses, commands = [], []
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(DemoStatus, "/handover_demo/status", statuses.append, qos)
        for side in ["left", "right"]:
            node.create_subscription(
                JointTrajectory,
                f"/leader/joint_trajectory_command_broadcaster_{side}/joint_trajectory",
                commands.append,
                10,
            )
        client = ActionClient(node, Handover, "/handover")
        approach_client = ActionClient(node, ApproachCup, "/approach_cup")
        refine_client = ActionClient(node, ApproachCup, "/refine_cup")
        pick_client = ActionClient(node, ApproachCup, "/pick_cup")
        pick_refine_client = ActionClient(node, ApproachCup, "/pick_cup_refine")
        capture = node.create_client(Trigger, "/handover_demo/capture_targets")

        def spin_until(condition, timeout=12):
            end = time.monotonic() + timeout
            while not condition() and time.monotonic() < end:
                assert process.poll() is None, "Launch exited prematurely"
                rclpy.spin_once(node, timeout_sec=0.05)
            assert condition(), "Timed out waiting for launch smoke-test condition"

        try:
            spin_until(
                lambda: client.server_is_ready()
                and approach_client.server_is_ready()
                and refine_client.server_is_ready()
                and pick_client.server_is_ready()
                and capture.service_is_ready()
                and bool(statuses)
            )
            assert not pick_refine_client.server_is_ready(), (
                "Removed pickup refinement server still exposed"
            )
            names = sorted(node.get_node_names())
            assert names == ["demo_smoke_inspector", "handover_demo"], names
            for direction in ["left_to_right", "right_to_left"]:
                goal = Handover.Goal()
                goal.direction = direction
                future = client.send_goal_async(goal)
                spin_until(future.done)
                assert not future.result().accepted, "Goal without robot feedback was accepted"
            subscriptions = node.get_subscriber_names_and_types_by_node("handover_demo", "/")
            assert "/perception_test_head/cup" in dict(subscriptions), subscriptions
            assert "/perception/cup" not in dict(subscriptions), subscriptions
            for execute in (False, True):
                request = ApproachCup.Goal(arm="right", cup_id="cup", execute=execute)
                future = approach_client.send_goal_async(request)
                spin_until(future.done)
                handle = future.result()
                if execute:
                    assert not handle.accepted, "Execution accepted without robot feedback"
                else:
                    assert handle.accepted, "Position preview unnecessarily required wrist TF"
                    result = handle.get_result_async()
                    spin_until(result.done)
                    assert not result.result().result.success
                    assert result.result().result.error_code == "CUP_NOT_FOUND"
            future = pick_client.send_goal_async(
                ApproachCup.Goal(arm="left", cup_id="cup", execute=True)
            )
            spin_until(future.done)
            assert not future.result().accepted, "Pick accepted without robot feedback"
            future = capture.call_async(Trigger.Request())
            spin_until(future.done)
            assert not future.result().success, "Capture succeeded without robot state"
            assert not commands, "Unexpected trajectory command at startup"
            print(
                "PASS: default launch; one demo node; both directions without robot feedback "
                "rejected; capture rejects absent state; no startup commands."
            )
        finally:
            client.destroy()
            approach_client.destroy()
            node.destroy_node()
            rclpy.shutdown()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise AssertionError("Launch did not shut down cleanly")
            finally:
                log.seek(0)
                print(log.read())
            assert process.returncode == 0, process.returncode


if __name__ == "__main__":
    main()
