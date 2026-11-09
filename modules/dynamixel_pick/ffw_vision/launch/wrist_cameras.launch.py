import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    realsense_launch = os.path.join(
        get_package_share_directory('ffw_bringup'), 'launch', 'camera_realsense.launch.py')
    # Metric depth aligned to color; bringup defaults stay unchanged
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch),
            launch_arguments={
                'colorizer.enable1': 'false',
                'colorizer.enable2': 'false',
                'align_depth.enable1': 'true',
                'align_depth.enable2': 'true',
            }.items()),
    ])
