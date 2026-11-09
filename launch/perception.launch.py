"""Shared observation producer; uses cameras already running in robot bringup."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("demo_1f"), "config", "perception.yaml"]
                ),
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="demo_1f",
                executable="perception_node",
                name="perception",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
            ),
        ]
    )
