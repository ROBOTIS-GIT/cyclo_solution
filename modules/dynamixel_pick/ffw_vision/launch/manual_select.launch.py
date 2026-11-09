import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('ffw_vision'), 'config', 'cameras.yaml')
    camera_id = LaunchConfiguration('camera_id')
    params_file = LaunchConfiguration('params_file')
    namespace = ['ffw_vision/', camera_id]
    overrides = {'camera_id': camera_id}

    return LaunchDescription([
        DeclareLaunchArgument('camera_id', default_value='head',
                              description='head, left_wrist or right_wrist'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('port', default_value='8765'),
        Node(
            package='ffw_vision', executable='pixel_observer', name='pixel_observer',
            namespace=namespace, parameters=[params_file, overrides], output='screen'),
        Node(
            package='ffw_vision', executable='web_select', name='web_select',
            namespace=namespace, output='screen',
            parameters=[params_file, overrides, {
                'port': ParameterValue(LaunchConfiguration('port'), value_type=int)}]),
    ])
