import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    vision_config = os.path.join(get_package_share_directory('ffw_vision'), 'config')
    config_dir = os.path.join(get_package_share_directory('ffw_grasp'), 'config')
    camera_id = LaunchConfiguration('camera_id')

    return LaunchDescription([
        DeclareLaunchArgument("python_executable", default_value="/usr/bin/python3"),
        DeclareLaunchArgument('camera_id', default_value='right_wrist',
                              description='head, left_wrist or right_wrist'),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(vision_config, 'cameras.yaml')),
        DeclareLaunchArgument('task_file',
                              default_value=os.path.join(config_dir, 'dynamixel_top_grasp.yaml')),
        DeclareLaunchArgument('calibration_file',
                              default_value=os.path.join(config_dir, 'robot_calibration.yaml'),
                              description='this robot: fingertip length, camera and arm biases'),
        DeclareLaunchArgument('basket_file',
                              default_value=os.path.join(vision_config, 'blue_basket.yaml')),
        # the calibration comes last: it wins over a task file that still has it
        Node(
            prefix=[LaunchConfiguration("python_executable")],
            package='ffw_grasp', executable='grasp_planner', name='grasp_planner',
            namespace=['ffw_grasp/', camera_id], output='screen',
            parameters=[LaunchConfiguration('params_file'), LaunchConfiguration('basket_file'),
                        LaunchConfiguration('task_file'), LaunchConfiguration('calibration_file'),
                        {'camera_id': camera_id}]),
    ])
