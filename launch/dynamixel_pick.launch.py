"""DYNAMIXEL perception/planning/backend; consumes existing camera topics."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    vision = Path(get_package_share_directory('ffw_vision'))
    grasp = Path(get_package_share_directory('ffw_grasp'))
    pick = Path(get_package_share_directory('ffw_pick'))
    demo = Path(get_package_share_directory('demo_1f'))
    return LaunchDescription([
        DeclareLaunchArgument('description_file', default_value=str(demo / 'urdf/sg2_extended_grippers.urdf')),
        DeclareLaunchArgument('pick_python', default_value='/opt/demo-pick-venv/bin/python3'),
        DeclareLaunchArgument('dynamixel_port', default_value='8767'),
        DeclareLaunchArgument('dynamixel_calibration', default_value=str(grasp / 'config/robot_calibration.yaml')),
        DeclareLaunchArgument('dynamixel_arm', default_value=str(pick / 'config/arm_r.yaml')),
        DeclareLaunchArgument('dynamixel_config', default_value=str(pick / 'config/pick.yaml')),
        DeclareLaunchArgument('dynamixel_task', default_value=str(grasp / 'config/dynamixel_top_grasp.yaml')),
        SetEnvironmentVariable('YOLO_AUTOINSTALL', 'false'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(vision / 'launch/auto_detect.launch.py')),
            launch_arguments={'python_executable': LaunchConfiguration('pick_python'), 'camera_id': 'right_wrist',
                              'task_file': str(vision / 'config/dynamixel_basket_hybrid.yaml'),
                              'bind_address': '0.0.0.0', 'port': LaunchConfiguration('dynamixel_port')}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(grasp / 'launch/grasp_planner.launch.py')),
            launch_arguments={'camera_id': 'right_wrist',
                              'python_executable': LaunchConfiguration('pick_python'),
                              'task_file': LaunchConfiguration('dynamixel_task'),
                              'calibration_file': LaunchConfiguration('dynamixel_calibration')}.items()),
        Node(prefix=[LaunchConfiguration('pick_python')], package='ffw_pick', executable='pick_action_server', output='screen',
             parameters=[{'arm_file': LaunchConfiguration('dynamixel_arm'),
                          'config_file': LaunchConfiguration('dynamixel_config'),
                          'task_file': LaunchConfiguration('dynamixel_task'),
                          'calibration_file': LaunchConfiguration('dynamixel_calibration'),
                          'urdf_file': LaunchConfiguration('description_file')}]),
    ])
