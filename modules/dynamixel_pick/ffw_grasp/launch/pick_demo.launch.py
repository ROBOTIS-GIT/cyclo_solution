import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    vision_share = get_package_share_directory('ffw_vision')
    grasp_share = get_package_share_directory('ffw_grasp')
    cameras_file = os.path.join(vision_share, 'config', 'cameras.yaml')
    camera_id = LaunchConfiguration('camera_id')

    # Cameras, detection, grasp planning and views in one terminal (Ctrl+C also stops the cameras);
    # the arm controller (cyclo movej) runs on its own
    return LaunchDescription([
        DeclareLaunchArgument('camera_id', default_value='right_wrist'),
        DeclareLaunchArgument(
            'detect_file',
            default_value=os.path.join(vision_share, 'config', 'dynamixel_basket_hybrid.yaml')),
        DeclareLaunchArgument(
            'grasp_file',
            default_value=os.path.join(grasp_share, 'config', 'dynamixel_top_grasp.yaml')),
        DeclareLaunchArgument('detect_port', default_value='8765'),
        DeclareLaunchArgument('grasp_port', default_value='8766'),
        SetEnvironmentVariable('YOLO_AUTOINSTALL', 'false'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(vision_share, 'launch', 'wrist_cameras.launch.py'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(vision_share, 'launch', 'auto_detect.launch.py')),
            launch_arguments={'camera_id': camera_id,
                              'task_file': LaunchConfiguration('detect_file'),
                              'port': LaunchConfiguration('detect_port')}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(grasp_share, 'launch', 'grasp_planner.launch.py')),
            launch_arguments={'camera_id': camera_id,
                              'task_file': LaunchConfiguration('grasp_file')}.items()),
        Node(
            package='ffw_vision', executable='web_select', name='grasp_view',
            namespace=['ffw_grasp/', camera_id], output='screen',
            parameters=[cameras_file, {
                'camera_id': camera_id, 'display_topic': 'debug_image',
                'port': ParameterValue(LaunchConfiguration('grasp_port'), value_type=int)}]),
    ])
