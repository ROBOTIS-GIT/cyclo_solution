"""One launch for the demo. Only /run_demo requests move the robot."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package = Path(get_package_share_directory('demo_1f'))

    def include(name, arguments=None, condition=None):
        options = {'launch_arguments': (arguments or {}).items()}
        if condition is not None:
            options['condition'] = condition
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(package / 'launch' / name)), **options)

    return LaunchDescription([
        DeclareLaunchArgument('pick_python', default_value='/opt/demo-pick-venv/bin/python3'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('start_moveit', default_value='true'),
        DeclareLaunchArgument('start_cup_perception', default_value='true'),
        DeclareLaunchArgument('start_dynamixel_pick', default_value='true'),
        DeclareLaunchArgument('cup_topic', default_value='/perception_test_head/cup'),
        DeclareLaunchArgument('viewer_host', default_value='0.0.0.0'),
        include('handover.launch.py', {
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'start_moveit': LaunchConfiguration('start_moveit'),
            'cup_topic': LaunchConfiguration('cup_topic')}),
        include('perception_test.launch.py', {
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'host': LaunchConfiguration('viewer_host')},
            IfCondition(LaunchConfiguration('start_cup_perception'))),
        include('dynamixel_pick.launch.py', {'pick_python': LaunchConfiguration('pick_python')}, condition=IfCondition(LaunchConfiguration('start_dynamixel_pick'))),
    ])
