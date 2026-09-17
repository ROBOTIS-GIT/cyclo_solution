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
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


_instance_lock = None


def segmenter_count_for_camera_set(camera_set):
    """Return the number of robot segmenters launched for one camera set."""
    return {
        'all': 3,
        'head': 1,
        'wrists': 2,
    }[camera_set]


def acquire_instance_lock():
    """Prevent two launch trees from publishing the same global endpoints."""
    global _instance_lock
    domain_id = os.environ.get('ROS_DOMAIN_ID', '0')
    lock_path = f'/tmp/cyclo_cumotion-domain-{domain_id}.lock'
    lock_file = open(lock_path, 'w', encoding='utf-8')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError(
            'Another cyclo_cumotion launch is already running in ROS domain '
            f'{domain_id}; stop it before starting a second instance'
        ) from error
    lock_file.write(f'{os.getpid()}\n')
    lock_file.flush()
    _instance_lock = lock_file


def launch_setup(context):
    acquire_instance_lock()
    moveit_share = get_package_share_directory('cyclo_cumotion_moveit_config')
    description_share = get_package_share_directory('cyclo_cumotion_description')
    cumotion_share = get_package_share_directory('isaac_ros_cumotion')
    nvblox_share = get_package_share_directory('cyclo_cumotion_nvblox')
    object_attachment_share = get_package_share_directory(
        'cyclo_cumotion_object_attachment')

    urdf_path = os.path.join(description_share, 'urdf', 'ffw.urdf')
    xrdf_dir = os.path.join(moveit_share, 'config', 'xrdf')
    sanitized_joint_states_topic = '/isaac_ros/joint_states'
    enable_static_scene = (
        LaunchConfiguration('enable_static_scene').perform(context).lower()
        == 'true'
    )
    read_esdf_world = (
        LaunchConfiguration('read_esdf_world').perform(context).lower() == 'true'
    )
    camera_set = LaunchConfiguration('nvblox_camera_set').perform(context)
    segmenter_count = segmenter_count_for_camera_set(camera_set)
    static_scene_file = (
        LaunchConfiguration('moveit_collision_objects_scene_file').perform(context)
        if enable_static_scene
        else ''
    )
    xrdf_by_group = {
        'arm_l': os.path.join(xrdf_dir, 'ffw_left_arm.xrdf'),
        'arm_r': os.path.join(xrdf_dir, 'ffw_right_arm.xrdf'),
        'both_arms': os.path.join(xrdf_dir, 'ffw_bimanual.xrdf'),
        'wholebody': os.path.join(xrdf_dir, 'ffw_wholebody.xrdf'),
    }

    moveit_config = (
        MoveItConfigsBuilder(
            robot_name='cyclo_cumotion_ffw',
            package_name='cyclo_cumotion_moveit_config',
        )
        .robot_description(file_path=urdf_path)
        .robot_description_semantic(Path('config') / 'ffw.srdf')
        .planning_pipelines(
            default_planning_pipeline='isaac_ros_cumotion',
            pipelines=['isaac_ros_cumotion'],
            load_all=False,
        )
        .to_moveit_configs()
    )

    common_parameters = [
        moveit_config.to_dict(),
        {
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'), value_type=bool
            ),
            'allow_trajectory_execution': True,
            # The physical arm can move a few encoder counts between planning and
            # controller hand-off.  Keep this small, but above the observed
            # 0.01-rad rejection, so a valid cuMotion plan is not discarded at
            # execution time.
            'trajectory_execution.allowed_start_tolerance': ParameterValue(
                LaunchConfiguration('trajectory_start_tolerance'), value_type=float
            ),
            'publish_robot_description_semantic': True,
            'publish_planning_scene': True,
            'publish_geometry_updates': True,
            'publish_state_updates': True,
            'publish_transforms_updates': True,
        },
    ]

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=common_parameters,
        remappings=[('/joint_states', sanitized_joint_states_topic)],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_moveit',
        output='log',
        condition=IfCondition(LaunchConfiguration('start_rviz')),
        arguments=['-d', os.path.join(moveit_share, 'config', 'moveit.rviz')],
        parameters=common_parameters,
        remappings=[('/joint_states', sanitized_joint_states_topic)],
    )

    joint_state_sanitizer = Node(
        package='cyclo_cumotion_moveit_config',
        executable='joint_state_sanitizer.py',
        output='screen',
        parameters=[{
            'urdf_path': urdf_path,
            'input_topic': LaunchConfiguration('joint_states_topic'),
            'output_topic': sanitized_joint_states_topic,
            'limit_roundoff_tolerance': 1.0e-4,
            # SG2 publishes these joints. BG2 does not have them, so fill only
            # these optional platform joints with zero to keep MoveIt's common
            # superset model complete without fabricating arm or lift states.
            'default_joint_names': [
                'left_wheel_steer',
                'left_wheel_drive',
                'right_wheel_steer',
                'right_wheel_drive',
                'rear_wheel_steer',
                'rear_wheel_drive',
            ],
            'default_joint_positions': [0.0] * 6,
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'), value_type=bool
            ),
        }],
    )
    static_scene_world_frame = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_scene_world_to_odom',
        output='screen',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--qx', '0',
            '--qy', '0',
            '--qz', '0',
            '--qw', '1',
            '--frame-id', 'world',
            '--child-frame-id', 'odom',
        ],
    )

    cumotion = GroupAction(
        [
            SetRemap(
                src='/cumotion/move_group', dst='/cumotion/backend_move_group'
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(cumotion_share, 'launch', 'isaac_ros_cumotion.launch.py')
                ),
                launch_arguments={
                    'cumotion_action_server.urdf_file_path': urdf_path,
                    'cumotion_action_server.xrdf_file_path': xrdf_by_group[
                        'wholebody'
                    ],
                    'cumotion_action_server.read_esdf_world': LaunchConfiguration(
                        'read_esdf_world'
                    ),
                    'cumotion_action_server.joint_states_topic': sanitized_joint_states_topic,
                    'cumotion_action_server.enable_cumotion_debug_mode': LaunchConfiguration(
                        'enable_cumotion_debug_mode'
                    ),
                    'cumotion_action_server.publish_world_collision_spheres': LaunchConfiguration(
                        'publish_world_collision_spheres'
                    ),
                    'cumotion_action_server.publish_self_collision_spheres': LaunchConfiguration(
                        'publish_self_collision_spheres'
                    ),
                    'cumotion_action_server.moveit_collision_objects_scene_file':
                        static_scene_file,
                }.items(),
            ),
        ]
    )

    group_router = Node(
        package='cyclo_cumotion_moveit_config',
        executable='cumotion_group_router.py',
        output='screen',
        parameters=[{
            'urdf_path': urdf_path,
            'initial_group': 'wholebody',
            'joint_states_topic': sanitized_joint_states_topic,
            'static_scene_file': static_scene_file,
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'), value_type=bool
            ),
            'xrdf.arm_l': xrdf_by_group['arm_l'],
            'xrdf.arm_r': xrdf_by_group['arm_r'],
            'xrdf.both_arms': xrdf_by_group['both_arms'],
            'xrdf.wholebody': xrdf_by_group['wholebody'],
            'segmentation_xrdf': os.path.join(
                xrdf_dir, 'ffw_segmentation.xrdf'),
            'segmentation_reload_required': read_esdf_world,
            'segmentation_node_names': [
                f'/robot_segmenter_{index}'
                for index in range(segmenter_count)
            ],
            'segmentation_reload_timeout_sec': 10.0,
            'planned_end_effector_path_auto_clear': ParameterValue(
                LaunchConfiguration(
                    'planned_end_effector_path_auto_clear'),
                value_type=bool,
            ),
            'planned_end_effector_path_hold_sec': ParameterValue(
                LaunchConfiguration('planned_end_effector_path_hold_sec'),
                value_type=float,
            ),
        }],
    )

    object_attachment = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            object_attachment_share, 'launch', 'object_attachment.launch.py')),
        condition=IfCondition(LaunchConfiguration('enable_object_attachment')),
        launch_arguments={
            'object_catalog_file': LaunchConfiguration('object_catalog_file'),
            'clear_esdf_on_attach': LaunchConfiguration('read_esdf_world'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )

    planning_actions = [joint_state_sanitizer]
    if enable_static_scene:
        planning_actions.append(static_scene_world_frame)
    planning_actions.append(cumotion)
    planning_actions.extend([
        move_group_node,
        group_router,
        object_attachment,
        rviz_node,
    ])

    if not read_esdf_world:
        return planning_actions

    nvblox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nvblox_share, 'launch', 'nvblox.launch.py')
        ),
        launch_arguments={
            'joint_states_topic': LaunchConfiguration('joint_states_topic'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'camera_set': LaunchConfiguration('nvblox_camera_set'),
        }.items(),
    )
    wait_for_esdf = Node(
        package='cyclo_cumotion_nvblox',
        executable='wait_for_esdf.py',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'), value_type=bool
            ),
        }],
    )

    def start_planning_after_esdf(event, _context):
        if event.returncode == 0:
            return planning_actions
        return []

    start_planning = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_esdf,
            on_exit=start_planning_after_esdf,
        )
    )
    return [start_planning, nvblox, wait_for_esdf]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument('joint_states_topic', default_value='/joint_states'),
            DeclareLaunchArgument(
                'read_esdf_world',
                default_value='false',
                choices=['true', 'false'],
                description=(
                    'Launch nvblox, wait for a usable ESDF, then start cuMotion '
                    'with ESDF collision checking'
                ),
            ),
            DeclareLaunchArgument(
                'nvblox_camera_set',
                default_value='all',
                choices=['all', 'head', 'wrists'],
                description='Depth cameras integrated into the nvblox world model',
            ),
            DeclareLaunchArgument(
                'enable_cumotion_debug_mode',
                default_value='false',
                choices=['true', 'false'],
            ),
            DeclareLaunchArgument(
                'publish_world_collision_spheres',
                default_value='true',
                choices=['true', 'false'],
                description=(
                    'Publish the live XRDF world-collision spheres on '
                    '/cumotion/collision_spheres for RViz visualization'
                ),
            ),
            DeclareLaunchArgument(
                'publish_self_collision_spheres',
                default_value='false',
                choices=['true', 'false'],
                description='Publish XRDF self-collision spheres as MarkerArray',
            ),
            DeclareLaunchArgument(
                'enable_static_scene',
                default_value='false',
                choices=['true', 'false'],
                description=(
                    'Load the configured MoveIt scene file into cuMotion and '
                    'publish the world-to-odom transform needed for its world frame'
                ),
            ),
            DeclareLaunchArgument(
                'moveit_collision_objects_scene_file',
                default_value=os.path.join(
                    get_package_share_directory('cyclo_cumotion_moveit_config'),
                    'config',
                    'scene',
                    'two_open_boxes.scene',
                ),
                description=(
                    'MoveIt .scene file loaded when enable_static_scene is true'
                ),
            ),
            DeclareLaunchArgument(
                'enable_object_attachment',
                default_value='false',
                choices=['true', 'false'],
                description=(
                    'Launch predefined-object attachment together with cuMotion'),
            ),
            DeclareLaunchArgument(
                'object_catalog_file',
                default_value=os.path.join(
                    get_package_share_directory(
                        'cyclo_cumotion_object_attachment'),
                    'config',
                    'predefined_objects.yaml',
                ),
                description='YAML catalog of predefined attachment objects',
            ),
            DeclareLaunchArgument(
                'trajectory_start_tolerance',
                default_value='0.03',
                description=(
                    'Maximum joint error in radians allowed between a planned '
                    'trajectory start and the measured state at execution'
                ),
            ),
            DeclareLaunchArgument(
                'planned_end_effector_path_auto_clear',
                default_value='true',
                choices=['true', 'false'],
                description=(
                    'Delete the RViz end-effector trajectory line after one '
                    'planned trajectory playback'
                ),
            ),
            DeclareLaunchArgument(
                'planned_end_effector_path_hold_sec',
                default_value='0.0',
                description=(
                    'Additional seconds to retain the RViz end-effector path '
                    'after trajectory playback'
                ),
            ),
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                choices=['true', 'false'],
                description=(
                    'Use simulation clock. Set true only when /clock is being published.'
                ),
            ),
            DeclareLaunchArgument(
                'start_rviz', default_value='true', choices=['true', 'false']
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
