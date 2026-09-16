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

import fcntl
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import (
    ComposableNodeContainer,
    LoadComposableNodes,
    Node,
)
from launch_ros.descriptions import ComposableNode


_nvblox_launch_lock = None


def acquire_nvblox_launch_lock():
    """Prevent component loads from leaking into an older named container."""
    global _nvblox_launch_lock
    lock_path = '/tmp/cyclo_cumotion_nvblox.lock'
    lock = open(lock_path, 'a+', encoding='utf-8')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError(
            'Another cyclo_cumotion nvblox launch is already running. '
            'Stop it before starting a new nvblox or cumotion launch.'
        ) from error
    lock.seek(0)
    lock.truncate()
    lock.write(f'{os.getpid()}\n')
    lock.flush()
    _nvblox_launch_lock = lock


def launch_setup(context):
    acquire_nvblox_launch_lock()
    description_share = get_package_share_directory('cyclo_cumotion_description')
    moveit_share = get_package_share_directory('cyclo_cumotion_moveit_config')
    package_share = get_package_share_directory('cyclo_cumotion_nvblox')

    urdf_path = os.path.join(description_share, 'urdf', 'ffw.urdf')
    xrdf_path = os.path.join(
        moveit_share, 'config', 'xrdf', 'ffw_segmentation.xrdf')
    nvblox_params = os.path.join(package_share, 'config', 'nvblox.yaml')

    use_sim_time = (
        LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    )
    joint_states_topic = LaunchConfiguration('joint_states_topic').perform(context)
    robot_description_service_name = LaunchConfiguration(
        'robot_description_service_name').perform(context)
    reload_robot_description_topic = LaunchConfiguration(
        'reload_robot_description_topic').perform(context)
    reload_robot_description_ack_topic = LaunchConfiguration(
        'reload_robot_description_ack_topic').perform(context)

    camera_topics_by_name = {
        'head': (
            '/zed/zed_node/depth/depth_registered',
            '/zed/zed_node/depth/camera_info',
            '/cyclo_cumotion/nvblox/head/robot_mask',
            '/cyclo_cumotion/nvblox/head/robot_world_depth',
        ),
        'left': (
            '/camera_left/camera_left/depth/image_rect_raw',
            '/camera_left/camera_left/depth/camera_info',
            '/cyclo_cumotion/nvblox/left/robot_mask',
            '/cyclo_cumotion/nvblox/left/robot_world_depth',
        ),
        'right': (
            '/camera_right/camera_right/depth/image_rect_raw',
            '/camera_right/camera_right/depth/camera_info',
            '/cyclo_cumotion/nvblox/right/robot_mask',
            '/cyclo_cumotion/nvblox/right/robot_world_depth',
        ),
    }
    camera_set = LaunchConfiguration('camera_set').perform(context)
    selected_names = {
        'all': ['head', 'left', 'right'],
        'head': ['head'],
        'right': ['left', 'right'],
    }[camera_set]
    camera_topics = [camera_topics_by_name[name] for name in selected_names]

    def make_segmenters():
        segmenters = []
        for index, topics in enumerate(camera_topics):
            depth, camera_info, robot_mask, robot_world_depth = topics
            segmenters.append(ComposableNode(
                package='cyclo_cumotion_robot_segmenter',
                plugin='nvidia::isaac_ros::manipulator::RobotSegmenter',
                name=f'robot_segmenter_{index}',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'urdf_path': urdf_path,
                    'xrdf_path': xrdf_path,
                    'robot_base_frame': 'base_link',
                    'additional_buffer_distance': 0.05,
                    'input_qos': 'SENSOR_DATA',
                    'output_qos': 'SENSOR_DATA',
                    'input_qos_depth': 10,
                    'output_qos_depth': 10,
                    'tf_lookup_timeout_ms': 200,
                    'tf_poll_period_ms': 5,
                    'pending_depth_queue_size': 10,
                    'joint_limit_tolerance': 0.02,
                    'robot_description_service_name':
                        robot_description_service_name,
                    'reload_robot_description_topic':
                        reload_robot_description_topic,
                    'reload_robot_description_ack_topic':
                        reload_robot_description_ack_topic,
                }],
                remappings=[
                    ('depth_image', depth),
                    ('camera_info_depth', camera_info),
                    ('joint_states', joint_states_topic),
                    ('robot_mask', robot_mask),
                    ('robot_depth', robot_world_depth),
                ],
                extra_arguments=[{'use_intra_process_comms': True}],
            ))
        return segmenters

    nvblox_remappings = []
    for index, topics in enumerate(camera_topics):
        _, depth_camera_info, _, robot_world_depth = topics
        nvblox_remappings.extend([
            (f'camera_{index}/depth/image', robot_world_depth),
            (f'camera_{index}/depth/camera_info', depth_camera_info),
        ])

    nvblox = Node(
        package='nvblox_ros',
        executable='nvblox_node',
        name='nvblox_node',
        output='screen',
        parameters=[nvblox_params, {
            'use_sim_time': use_sim_time,
            'num_cameras': len(camera_topics),
        }],
        remappings=nvblox_remappings,
    )

    tsdf_cube_marker = Node(
        package='cyclo_cumotion_nvblox',
        executable='tsdf_cube_marker.py',
        name='tsdf_cube_marker',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    container = ComposableNodeContainer(
        name='cyclo_cumotion_nvblox_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        # The binary robot segmenter aborts if its remote camera/joint streams
        # disappear during an AI Worker reboot. Restore world_depth without
        # requiring the complete mapping launch to be restarted manually.
        respawn=True,
        respawn_delay=5.0,
    )

    # Load explicitly after remote TF/joint discovery has had time to settle.
    # RobotSegmenter's constructor waits synchronously for camera TF; loading
    # immediately with the container can otherwise pin the load_node request
    # before the external bringup graph is visible.
    initial_segmenter_load = TimerAction(
        period=3.0,
        actions=[LoadComposableNodes(
            target_container=container,
            composable_node_descriptions=make_segmenters(),
        )],
    )

    # A respawn creates an empty container, so reload the segmenter(s) after
    # subsequent process starts as well. Skip the first event because the
    # explicit initial action above owns that load.
    container_start_count = 0

    def load_segmenters(context):
        nonlocal container_start_count
        del context
        container_start_count += 1
        if container_start_count == 1:
            return []
        # LoadComposableNodes and ComposableNode descriptions contain mutable
        # launch state. Recreate both so respawns retain parameters/remaps.
        return [TimerAction(
            period=3.0,
            actions=[LoadComposableNodes(
                target_container=container,
                composable_node_descriptions=make_segmenters(),
            )],
        )]

    load_segmenters_on_container_start = RegisterEventHandler(
        OnProcessStart(
            target_action=container,
            on_start=[OpaqueFunction(function=load_segmenters)],
        )
    )

    return [
        load_segmenters_on_container_start,
        container,
        initial_segmenter_load,
        nvblox,
        tsdf_cube_marker,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('joint_states_topic', default_value='/joint_states'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument(
            'camera_set', default_value='all', choices=['all', 'head', 'right'],
            description=(
                'Use all depth cameras, only the head ZED, or only the right '
                'wrist camera')),
        DeclareLaunchArgument(
            'robot_description_service_name',
            default_value='/cyclo_cumotion/nvblox/segmentation/get_robot_description',
            description=(
                'Dedicated segmenter description service. Keep this separate '
                'from planning-group-specific cuMotion descriptions.')),
        DeclareLaunchArgument(
            'reload_robot_description_topic',
            default_value='/cyclo_cumotion/nvblox/segmentation/reload_robot_description',
            description=(
                'Dedicated segmenter reload topic. Planning-group model reloads '
                'must not replace the full segmentation c-space.')),
        DeclareLaunchArgument(
            'reload_robot_description_ack_topic',
            default_value=(
                '/cyclo_cumotion/nvblox/segmentation/'
                'reload_robot_description_ack'),
            description=(
                'Acknowledgements emitted after each segmenter has installed '
                'the requested robot-description generation.')),
        OpaqueFunction(function=launch_setup),
    ])
