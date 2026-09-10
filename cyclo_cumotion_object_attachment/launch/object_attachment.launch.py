#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yeonguk Kim

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory(
        'cyclo_cumotion_object_attachment')
    nvidia_share = get_package_share_directory(
        'isaac_ros_cumotion_object_attachment')

    nvidia_attachment = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nvidia_share, 'launch', 'object_attachment.launch.py')
        ),
        launch_arguments={
            'object_attachment.container_name': '',
            'object_attachment.max_overshoot': LaunchConfiguration(
                'max_overshoot'),
            'object_attachment.clear_esdf_on_attach': LaunchConfiguration(
                'clear_esdf_on_attach'),
            'object_attachment.esdf_reference_frame': 'base_link',
            'object_attachment.get_robot_description_service':
                '/cyclo_cumotion/attachment/get_robot_description',
            'object_attachment.set_robot_description_service':
                '/cyclo_cumotion/attachment/set_robot_description',
        }.items(),
        condition=IfCondition(LaunchConfiguration('enabled')),
    )

    predefined_objects = Node(
        package='cyclo_cumotion_object_attachment',
        executable='predefined_object_attachment.py',
        name='predefined_object_attachment',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enabled')),
        parameters=[
            {
                'catalog_file': LaunchConfiguration('object_catalog_file'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'enabled', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument(
            'object_catalog_file',
            default_value=os.path.join(
                package_share, 'config', 'predefined_objects.yaml')),
        DeclareLaunchArgument(
            'clear_esdf_on_attach', default_value='false',
            choices=['true', 'false']),
        DeclareLaunchArgument('max_overshoot', default_value='0.01'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false', choices=['true', 'false']),
        nvidia_attachment,
        predefined_objects,
    ])
