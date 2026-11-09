"""One-camera perception + browser viewer only; no MoveIt or motion action servers."""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def nodes(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    source = "head"
    profile = yaml.safe_load(Path(arg("profiles")).read_text())[source]
    prefix = f"/perception_test_{source}"
    port = int(arg("port") or profile["port"])
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    roi = yaml.safe_load(arg("roi"))
    if not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi):
        raise ValueError("roi must contain four integers: [x,y,width,height]")
    if any(v < 0 for v in roi) or (roi != [0, 0, 0, 0] and (roi[2] == 0 or roi[3] == 0)):
        raise ValueError("roi must be [0,0,0,0] for full frame or a positive-size rectangle")
    # Profiles are expected names; this launch never changes camera/bringup settings.
    topics = {
        name: arg(name) or profile[name] for name in ("rgb_topic", "depth_topic", "info_topic")
    }
    if any(not topic.strip() for topic in topics.values()):
        raise ValueError("All three camera topic names are required")
    parameters = {
        "cup.id": "cup",
        "cup.inner_radius": float(arg("inner_radius")),
        "cup.wall_thickness": 0.005,
        "debug.enabled": True,
        "debug.allow_full_frame_roi": True,
        f"cameras.{source}.roi": roi,
        f"cameras.{source}.require_zero_distortion": profile.get("require_zero_distortion", False),
        "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
    }
    parameters.update(
        {"ai.backend": arg("ai_backend"), "ai.score_threshold": float(arg("ai_score"))}
    )
    for key in ("model", "engine"):
        if arg("ai_" + key):
            parameters["ai." + key] = arg("ai_" + key)
    parameters.update({f"cameras.{source}.{key}": value for key, value in topics.items()})
    return [
        Node(
            package="demo_1f",
            executable="wrist_perception.py",
            name="wrist_cup_right",
            output="screen",
            parameters=[
                {
                    "arm": "right",
                    "use_sim_time": parameters["use_sim_time"],
                    "inner_radius": float(arg("inner_radius")),
                }
            ],
            remappings=[
                ("/wrist_cup/image", "/wrist_cup_right/image"),
                ("/wrist_cup/status", "/wrist_cup_right/status"),
            ],
        ),
        Node(
            package="demo_1f",
            executable="wrist_perception.py",
            name="wrist_cup",
            output="screen",
            parameters=[
                {
                    "use_sim_time": parameters["use_sim_time"],
                    "inner_radius": float(arg("inner_radius")),
                }
            ],
        ),
        Node(
            package="demo_1f",
            executable="perception_node",
            name=f"perception_test_{source}",
            output="screen",
            parameters=[parameters],
            # Test detections cannot feed the motion subscription unless explicitly selected.
            remappings=[("/perception/cup", prefix + "/cup")],
        ),
        Node(
            package="demo_1f",
            executable="perception_viewer.py",
            name=f"perception_viewer_{source}",
            output="screen",
            parameters=[
                {
                    "topic_prefix": prefix,
                    "depth_head_topic": topics["depth_topic"],
                    "left_exclude_rect": yaml.safe_load(arg("left_exclude_rect")),
                    "host": arg("host"),
                    "port": port,
                    "use_sim_time": parameters["use_sim_time"],
                }
            ],
        ),
    ]


def generate_launch_description():
    defaults = {
        "ai_backend": "auto",
        "ai_score": "0.35",
        "ai_model": "",
        "ai_engine": "",
        "profiles": str(
            Path(get_package_share_directory("demo_1f")) / "config/perception_test.yaml"
        ),
        "rgb_topic": "",
        "depth_topic": "",
        "info_topic": "",
        "roi": "[0,0,0,0]",
        "inner_radius": "0.045",
        "left_exclude_rect": "[0.0, 0.32, 0.23, 0.72]",
        "host": "127.0.0.1",
        "port": "",
        "use_sim_time": "false",
    }
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()]
        + [OpaqueFunction(function=nodes)]
    )
