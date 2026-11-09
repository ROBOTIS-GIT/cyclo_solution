"""Demo only; hardware bringup remains in its existing container."""

from pathlib import Path
import xml.etree.ElementTree as ET

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def demo_node(context):
    package = Path(get_package_share_directory("demo_1f"))
    model = Path(LaunchConfiguration("description_file").perform(context))
    description = model.read_text()
    robot = ET.fromstring(description)
    frames = []
    for side in ('l', 'r'):
        joint = robot.find(f"joint[@name='gripper_{side}_tcp_joint']")
        if joint is None:
            raise ValueError('description_file must include the demo extended-gripper TCPs')
        xyz = joint.find('origin').get('xyz').split()
        frames.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=f'demo_gripper_{side}_tcp',
            arguments=['--x', xyz[0], '--y', xyz[1], '--z', xyz[2],
                       '--frame-id', joint.find('parent').get('link'),
                       '--child-frame-id', joint.find('child').get('link')],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]))
    return [
        *frames,
        Node(
            package="demo_1f",
            executable="handover_demo_node",
            name="handover_demo",
            output="screen",
            remappings=[("/perception/cup", LaunchConfiguration("cup_topic"))],
            parameters=[
                LaunchConfiguration("config"),
                LaunchConfiguration("demo_config"),
                LaunchConfiguration("approach_config"),
                LaunchConfiguration("pick_config"),
                {
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "robot_description": description,
                    "robot_description_semantic": (
                        package / "config/sg2_handover.srdf"
                    ).read_text(),
                    "robot_description_planning": yaml.safe_load(
                        (package / "config/sg2_joint_limits.yaml").read_text()
                    ),
                },
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("demo_1f"), "config", "handover.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "approach_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("demo_1f"), "config", "approach.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "pick_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("demo_1f"), "config", "pick.yaml"]
                ),
            ),
            DeclareLaunchArgument("demo_config", default_value=PathJoinSubstitution(
                [FindPackageShare("demo_1f"), "config", "demo.yaml"])),
            DeclareLaunchArgument("cup_topic", default_value="/perception/cup"),
            DeclareLaunchArgument("start_moveit", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "description_file",
                default_value=str(
                    Path(get_package_share_directory("demo_1f"))
                    / "urdf/sg2_extended_grippers.urdf"
                ),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("demo_1f"), "launch", "sg2_moveit.launch.py"]
                    )
                ),
                condition=IfCondition(LaunchConfiguration("start_moveit")),
                launch_arguments={
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "description_file": LaunchConfiguration("description_file"),
                }.items(),
            ),
            OpaqueFunction(function=demo_node),
        ]
    )
