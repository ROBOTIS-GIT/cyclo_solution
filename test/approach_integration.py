"""Cup approach extension for sg2_integration: synthetic cup, actual MoveIt/JTC/GenericSystem."""

import math
import os
import signal

import rclpy
import yaml
from demo_1f.action import ApproachCup
from demo_1f.msg import CupObservation
from geometry_msgs.msg import TransformStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.action import ActionClient
from tf2_ros import TransformBroadcaster


def run_approach(node, tf_buffer, root, work, processes, start, wait, urdf_path, demo_config):
    assert os.environ.get("DEMO_ISOLATED_TEST") == "1"
    tcp = tf_buffer.lookup_transform("base_link", "end_effector_l_link", rclpy.time.Time())
    position = tcp.transform.translation
    radius, cup_height = 0.045, 0.165
    body_z = position.z - 0.235 - cup_height / 2
    config = {
        "inner_radius": radius,
        "cup_height": cup_height,
        "observation_timeout": 3.0,
        "tcp_roll_limit_deg": 60.0,
    }
    approach_config = work / "approach.yaml"
    approach_config.write_text(
        yaml.safe_dump({"handover_demo": {"ros__parameters": {"approach": config}}})
    )
    # Restart only the isolated demo + MoveIt, leaving mock hardware active.
    demo = processes[-1]
    os.killpg(demo.pid, signal.SIGINT)
    demo.wait(timeout=15)
    start(
        "approach_demo",
        [
            "ros2",
            "launch",
            "demo_1f",
            "handover.launch.py",
            "use_sim_time:=false",
            "start_moveit:=true",
            "description_file:=" + str(urdf_path),
            "config:=" + str(demo_config),
            "approach_config:=" + str(approach_config),
        ],
    )
    publisher = node.create_publisher(CupObservation, "/perception/cup", 10)
    delayed_tf = TransformBroadcaster(node)

    roll_samples = []
    current_phase = ""

    def feedback(message):
        nonlocal current_phase
        current_phase = message.feedback.phase

    def observe():
        now = node.get_clock().now()
        try:
            q_now = tf_buffer.lookup_transform(
                "base_link", "end_effector_l_link", rclpy.time.Time()
            ).transform.rotation
            roll = math.atan2(
                2 * (q_now.y * q_now.z + q_now.w * q_now.x),
                2 * (q_now.x * q_now.z - q_now.w * q_now.y),
            )
            roll_samples.append((current_phase, math.degrees(roll)))
        except Exception:
            pass
        if delayed_tf:
            transform = TransformStamped()
            transform.header.frame_id = "base_link"
            transform.child_frame_id = "cup_test_head_camera"
            transform.header.stamp = (now - rclpy.duration.Duration(seconds=0.25)).to_msg()
            transform.transform.rotation.w = 1.0
            delayed_tf.sendTransform(transform)
        observation = CupObservation()
        observation.observation_type = CupObservation.BODY
        observation.header.frame_id = "cup_test_head_camera"
        observation.header.stamp = (now - rclpy.duration.Duration(seconds=0.05)).to_msg()
        observation.source, observation.cup_id = "head", "cup"
        observation.valid, observation.inner_radius = True, radius
        observation.body_position.x = position.x
        observation.body_position.y = position.y
        observation.body_position.z = body_z
        observation.position_error, observation.confidence = 0.09, 1.0
        publisher.publish(observation)

    timer = node.create_timer(0.05, observe)
    approach_client = ActionClient(node, ApproachCup, "/approach_cup")
    try:
        wait(lambda: approach_client.server_is_ready())
        # Native ARM startup varies; wait for actual scene/action endpoints, not a fixed sleep.
        query = node.create_client(GetPlanningScene, "/get_planning_scene")
        apply = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        move = ActionClient(node, MoveGroup, "/move_action")
        execute = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")
        wait(
            lambda: query.service_is_ready()
            and apply.service_is_ready()
            and move.server_is_ready()
            and execute.server_is_ready(),
            60,
        )
        request = GetPlanningScene.Request()
        request.components.components = 1023
        scene = query.call_async(request)
        wait(scene.done)
        assert scene.result() is not None
        # Give the demo node its own graph discovery callback after all services appear.
        ready = node.get_clock().now().nanoseconds + 1_000_000_000
        wait(lambda: node.get_clock().now().nanoseconds >= ready)
        initial_joints = scene.result().scene.robot_state.joint_state
        grippers_before = {
            name: value
            for name, value in zip(initial_joints.name, initial_joints.position, strict=True)
            if name in ("gripper_l_joint1", "gripper_r_joint1")
        }
        assert len(grippers_before) == 2
        for execute_approach in (False, True):
            approach = ApproachCup.Goal(arm="left", cup_id="cup", execute=execute_approach)
            sent = approach_client.send_goal_async(approach, feedback_callback=feedback)
            wait(sent.done)
            approach_handle = sent.result()
            assert approach_handle is not None and approach_handle.accepted, "Approach rejected"
            completed = approach_handle.get_result_async()
            wait(completed.done, 90)
            approach_outcome = completed.result().result
            assert approach_outcome.success, str(approach_outcome)
            assert approach_outcome.reached_observation_pose == execute_approach
            assert approach_outcome.target_pose.header.frame_id == "base_link"
            assert abs(approach_outcome.target_pose.pose.position.z - (body_z + cup_height)) < 1e-6
        approach_rolls = [abs(value) for phase, value in roll_samples if phase == "MOVE_TO_OBSERVE"]
        assert approach_rolls, "No roll samples during approach"
        # Transit orientation is free; only measured arrival must satisfy the tilt bound.
        assert abs(roll_samples[-1][1]) <= 60, roll_samples[-1]
        print(
            f"ROLL: max_approach={max(approach_rolls):.3f} final={roll_samples[-1][1]:.3f}",
            flush=True,
        )
        after = query.call_async(request)
        wait(after.done)
        final_joints = after.result().scene.robot_state.joint_state
        grippers_after = dict(zip(final_joints.name, final_joints.position, strict=True))
        for name, value in grippers_before.items():
            assert abs(grippers_after[name] - value) < 1e-5, "Approach changed gripper position"
        assert not after.result().scene.robot_state.attached_collision_objects, (
            "Invented grasp geometry"
        )
        from rclpy.action import get_action_names_and_types

        assert "/place_in_cup" not in dict(get_action_names_and_types(node))
        print(
            "PASS: BODY approach, delayed TF, arrival roll <=60, target Z=BODY+height, unchanged grippers, no placement action",
            flush=True,
        )
        node.destroy_client(query)
        node.destroy_client(apply)
        move.destroy()
        execute.destroy()
        return {"tcp_approach_completed": True}
    finally:
        node.destroy_timer(timer)
        node.destroy_publisher(publisher)
        approach_client.destroy()
