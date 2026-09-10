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

import copy
import math
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

from isaac_ros_cumotion_interfaces.srv import (
    GetRobotDescription,
    PublishStaticPlanningScene,
    SetRobotDescription,
)
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import ApplyPlanningScene, GetPositionFK
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from robotis_interfaces.srv import SetAttachmentFrame, SetGraspContactMode
from sensor_msgs.msg import JointState
from std_msgs.msg import String, UInt64
from std_srvs.srv import Trigger
from visualization_msgs.msg import (
    InteractiveMarkerUpdate,
    Marker,
    MarkerArray,
)
import yaml


class CumotionGroupRouter(Node):
    """Route MoveIt groups to matching cuMotion c-spaces without relaunching."""

    _supported_groups = ('arm_l', 'arm_r', 'both_arms', 'wholebody')
    # Isaac ROS cuMotion 4.6 keeps the tool-frame name selected at process
    # startup when SetRobotDescription swaps the model. Keep one stable tool
    # name and make it an alias of the requested right tool in the backend-only
    # URDF when needed.
    _backend_tool_frame = 'end_effector_l_link'
    _default_tool_by_group = {
        'arm_l': 'end_effector_l_link',
        'arm_r': 'end_effector_r_link',
        'both_arms': 'end_effector_l_link',
        'wholebody': 'end_effector_l_link',
    }
    _equivalent_tool_frames = {
        'wholebody_control_l_link': 'end_effector_l_link',
        'wholebody_control_r_link': 'end_effector_r_link',
    }
    _lift_joints = ('lift_joint',)
    _left_arm_joints = tuple(f'arm_l_joint{i}' for i in range(1, 8))
    _right_arm_joints = tuple(f'arm_r_joint{i}' for i in range(1, 8))
    _joints_by_group = {
        'arm_l': _left_arm_joints,
        'arm_r': _right_arm_joints,
        'both_arms': _left_arm_joints + _right_arm_joints,
        'wholebody': (
            _lift_joints + _left_arm_joints + _right_arm_joints
        ),
    }
    _attachment_ignore_by_frame = {
        'end_effector_l_link': [
            'arm_l_link6',
            'arm_l_link7',
            'gripper_l_rh_p12_rn_r1',
            'gripper_l_rh_p12_rn_l1',
            'gripper_l_rh_p12_rn_r2',
            'gripper_l_rh_p12_rn_l2',
        ],
        'end_effector_r_link': [
            'arm_r_link6',
            'arm_r_link7',
            'gripper_r_rh_p12_rn_r1',
            'gripper_r_rh_p12_rn_l1',
            'gripper_r_rh_p12_rn_r2',
            'gripper_r_rh_p12_rn_l2',
        ],
    }
    # Only the finger links expected to touch a grasped object are removed
    # from the world/self collision geometry during the one-shot final
    # approach. Arm and wrist links remain collision checked.
    _grasp_contact_links_by_frame = {
        'end_effector_l_link': [
            'gripper_l_rh_p12_rn_r1',
            'gripper_l_rh_p12_rn_l1',
            'gripper_l_rh_p12_rn_r2',
            'gripper_l_rh_p12_rn_l2',
        ],
        'end_effector_r_link': [
            'gripper_r_rh_p12_rn_r1',
            'gripper_r_rh_p12_rn_l1',
            'gripper_r_rh_p12_rn_r2',
            'gripper_r_rh_p12_rn_l2',
        ],
    }

    def __init__(self):
        super().__init__('cumotion_group_router')
        callback_group = ReentrantCallbackGroup()
        self._planning_lock = threading.Lock()
        configured_groups = tuple(self.declare_parameter(
            'enabled_groups', list(self._supported_groups)).value)
        unknown_groups = set(configured_groups) - set(self._supported_groups)
        if unknown_groups:
            raise ValueError(
                f'Unknown enabled cuMotion groups: {sorted(unknown_groups)}')
        self._enabled_groups = configured_groups

        self._urdf = Path(self.declare_parameter('urdf_path', '').value).read_text(
            encoding='utf-8'
        )
        self._position_limits = {}
        mimic_limits = []
        for joint in ET.fromstring(self._urdf).findall('joint'):
            limit = joint.find('limit')
            if limit is not None and 'lower' in limit.attrib and 'upper' in limit.attrib:
                self._position_limits[joint.attrib['name']] = (
                    float(limit.attrib['lower']),
                    float(limit.attrib['upper']),
                )
            mimic = joint.find('mimic')
            if mimic is not None and limit is not None:
                mimic_limits.append((
                    mimic.attrib['joint'],
                    float(mimic.attrib.get('multiplier', 1.0)),
                    float(mimic.attrib.get('offset', 0.0)),
                    float(limit.attrib['lower']),
                    float(limit.attrib['upper']),
                ))
        # cuMotion intersects a driving joint's URDF limit with every mimic
        # child's limit. Mirror that effective auxiliary c-space limit here so
        # a small encoder overshoot cannot abort SetRobotDescription.
        for master, multiplier, offset, child_lower, child_upper in mimic_limits:
            if multiplier == 0.0 or master not in self._position_limits:
                continue
            projected = (
                (child_lower - offset) / multiplier,
                (child_upper - offset) / multiplier,
            )
            mimic_lower, mimic_upper = sorted(projected)
            master_lower, master_upper = self._position_limits[master]
            effective = (
                max(master_lower, mimic_lower),
                min(master_upper, mimic_upper),
            )
            if effective[0] > effective[1]:
                raise ValueError(
                    f'Inconsistent URDF/mimic limits for {master!r}: '
                    f'{effective}'
                )
            self._position_limits[master] = effective
        self._auxiliary_joint_limit_tolerance = float(
            self.declare_parameter(
                'auxiliary_joint_limit_tolerance', 0.02
            ).value
        )
        if self._auxiliary_joint_limit_tolerance < 0.0:
            raise ValueError('auxiliary_joint_limit_tolerance must be non-negative')
        xrdf_text_by_group = {
            group: Path(self.declare_parameter(f'xrdf.{group}', '').value).read_text(
                encoding='utf-8'
            )
            for group in self._supported_groups
        }
        self._xrdf_template_by_group = {
            group: yaml.safe_load(xrdf)
            for group, xrdf in xrdf_text_by_group.items()
        }
        self._startup_backend_xrdf = xrdf_text_by_group['wholebody']
        for xrdf in self._xrdf_template_by_group.values():
            xrdf['tool_frames'] = [self._backend_tool_frame]
        segmentation_xrdf_path = self.declare_parameter(
            'segmentation_xrdf', '').value
        self._segmentation_xrdf_template = yaml.safe_load(
            Path(segmentation_xrdf_path).read_text(encoding='utf-8')
        )
        self._attachment_description_template = copy.deepcopy(
            self._xrdf_template_by_group['arm_r'])
        self._attached_object_spheres = None
        self._attachment_frame = 'end_effector_r_link'
        self._attachment_transaction_guard = threading.Lock()
        self._attachment_transaction_active = False
        self._attachment_transaction_snapshot = None
        self._grasp_contact_guard = threading.Lock()
        self._armed_grasp_contact = None
        self._collision_spheres_by_group = {
            group: xrdf['geometry'][
                'auto_generated_collision_sphere_group'
            ]['spheres']
            for group, xrdf in self._xrdf_template_by_group.items()
        }
        self._active_group = self.declare_parameter(
            'initial_group', 'wholebody'
        ).value
        # The backend starts from the on-disk XRDF. Keep this unset so the
        # first request always reloads a copy whose non-cspace defaults match
        # the same live joint-state snapshot used as the planning start state.
        self._loaded_xrdf = None
        self._loaded_urdf = None
        self._latest_joint_state = None
        self._latest_joint_state_monotonic = None
        self._force_current_start_state = self.declare_parameter(
            'force_current_start_state', True
        ).value
        self._max_current_state_age_sec = self.declare_parameter(
            'max_current_state_age_sec', 0.5
        ).value
        joint_states_topic = self.declare_parameter(
            'joint_states_topic', '/joint_states'
        ).value
        self._joint_state_subscription = self.create_subscription(
            JointState,
            joint_states_topic,
            self._on_joint_state,
            10,
            callback_group=callback_group,
        )

        self._set_description = self.create_client(
            SetRobotDescription,
            '/cumotion/set_robot_description',
            callback_group=callback_group,
        )
        self._attachment_get_description = self.create_service(
            GetRobotDescription,
            '/cyclo_cumotion/attachment/get_robot_description',
            self._on_attachment_get_description,
            callback_group=callback_group,
        )
        self._attachment_set_description = self.create_service(
            SetRobotDescription,
            '/cyclo_cumotion/attachment/set_robot_description',
            self._on_attachment_set_description,
            callback_group=callback_group,
        )
        self._attachment_frame_service = self.create_service(
            SetAttachmentFrame,
            '/cyclo_cumotion/attachment/set_frame',
            self._on_set_attachment_frame,
            callback_group=callback_group,
        )
        self._grasp_contact_service = self.create_service(
            SetGraspContactMode,
            '/cyclo_cumotion/grasp_contact_mode',
            self._on_set_grasp_contact_mode,
            callback_group=callback_group,
        )
        self._segmentation_get_description = self.create_service(
            GetRobotDescription,
            '/cyclo_cumotion/nvblox/segmentation/get_robot_description',
            self._on_segmentation_get_description,
            callback_group=callback_group,
        )
        self._segmentation_reload_required = bool(self.declare_parameter(
            'segmentation_reload_required', False).value)
        self._expected_segmenters = set(self.declare_parameter(
            'segmentation_node_names', Parameter.Type.STRING_ARRAY).value)
        self._segmentation_reload_timeout_sec = float(self.declare_parameter(
            'segmentation_reload_timeout_sec', 10.0).value)
        if self._segmentation_reload_required and not self._expected_segmenters:
            raise ValueError(
                'segmentation_node_names must not be empty when segmenter '
                'reload acknowledgement is required')
        if self._segmentation_reload_timeout_sec <= 0.0:
            raise ValueError('segmentation_reload_timeout_sec must be positive')
        self._segmentation_ack_condition = threading.Condition()
        self._segmentation_generation = 0
        self._pending_segmentation_generation = 0
        self._segmentation_acks = set()
        reload_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._segmentation_reload = self.create_publisher(
            UInt64,
            '/cyclo_cumotion/nvblox/segmentation/reload_robot_description',
            reload_qos,
        )
        self._segmentation_reload_ack = self.create_subscription(
            String,
            '/cyclo_cumotion/nvblox/segmentation/reload_robot_description_ack',
            self._on_segmentation_reload_ack,
            10,
            callback_group=callback_group,
        )
        self._attachment_begin_service = self.create_service(
            Trigger,
            '/cyclo_cumotion/attachment/begin_transaction',
            self._on_attachment_begin,
            callback_group=callback_group,
        )
        self._attachment_commit_service = self.create_service(
            Trigger,
            '/cyclo_cumotion/attachment/commit_transaction',
            self._on_attachment_commit,
            callback_group=callback_group,
        )
        self._attachment_rollback_service = self.create_service(
            Trigger,
            '/cyclo_cumotion/attachment/rollback_transaction',
            self._on_attachment_rollback,
            callback_group=callback_group,
        )
        self._compute_fk = self.create_client(
            GetPositionFK,
            '/compute_fk',
            callback_group=callback_group,
        )
        self._static_scene_file = self.declare_parameter(
            'static_scene_file', ''
        ).value
        self._static_scene_request_in_flight = False
        self._static_scene_server = None
        self._apply_planning_scene = None
        self._static_scene_timer = None
        if self._static_scene_file:
            self._static_scene_server = self.create_client(
                PublishStaticPlanningScene,
                '/publish_static_planning_scene',
                callback_group=callback_group,
            )
            self._apply_planning_scene = self.create_client(
                ApplyPlanningScene,
                '/apply_planning_scene',
                callback_group=callback_group,
            )
            self._static_scene_timer = self.create_timer(
                0.5,
                self._synchronize_static_scene,
                callback_group=callback_group,
                clock=Clock(clock_type=ClockType.STEADY_TIME),
            )
        self._goal_collision_spheres = self.create_publisher(
            MarkerArray,
            '/cumotion/goal_collision_spheres',
            1,
        )
        self._interactive_goal_state = None
        self._sphere_generation = 0
        self._pending_sphere_request = None
        self._sphere_debounce_timer = self.create_timer(
            0.04,
            self._flush_fk_spheres,
            callback_group=callback_group,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self._interactive_marker_names = set()
        self._interactive_group = None
        self._query_goal_spheres_service = self.create_service(
            GetPositionFK,
            '/cumotion_query_goal_spheres',
            self._on_query_goal_spheres,
            callback_group=callback_group,
        )
        self._interactive_marker_update_subscription = self.create_subscription(
            InteractiveMarkerUpdate,
            '/rviz_moveit_motion_planning_display/'
            'robot_interaction_interactive_marker_topic/update',
            self._on_interactive_marker_update,
            10,
            callback_group=callback_group,
        )

        self._backend = ActionClient(
            self,
            MoveGroup,
            '/cumotion/backend_move_group',
            callback_group=callback_group,
        )
        self._server = ActionServer(
            self,
            MoveGroup,
            '/cumotion/move_group',
            execute_callback=self._execute,
            goal_callback=self._goal,
            cancel_callback=self._cancel,
            callback_group=callback_group,
        )

    def _synchronize_static_scene(self):
        if self._static_scene_request_in_flight:
            return
        if not self._static_scene_server.service_is_ready():
            return
        if not self._apply_planning_scene.service_is_ready():
            return

        self._static_scene_request_in_flight = True
        request = PublishStaticPlanningScene.Request()
        request.scene_file_path = self._static_scene_file
        future = self._static_scene_server.call_async(request)
        future.add_done_callback(self._on_static_scene_received)

    def _on_static_scene_received(self, future):
        try:
            response = future.result()
        except Exception as exception:  # noqa: BLE001
            self._static_scene_request_in_flight = False
            self.get_logger().error(
                f'Failed to load static planning scene: {exception}')
            return

        planning_scene = response.planning_scene
        planning_scene.is_diff = True
        planning_scene.robot_state.is_diff = True
        request = ApplyPlanningScene.Request()
        request.scene = planning_scene
        apply_future = self._apply_planning_scene.call_async(request)
        apply_future.add_done_callback(
            lambda completed: self._on_static_scene_applied(
                completed, len(planning_scene.world.collision_objects)))

    def _on_static_scene_applied(self, future, object_count):
        self._static_scene_request_in_flight = False
        try:
            success = future.result().success
        except Exception as exception:  # noqa: BLE001
            self.get_logger().error(
                f'Failed to apply static planning scene to MoveIt: {exception}')
            return
        if not success:
            self.get_logger().error(
                'MoveIt rejected the static planning scene; retrying')
            return

        self._static_scene_timer.cancel()
        self.get_logger().info(
            f'Applied {object_count} static collision objects to MoveIt/RViz')

    def _on_joint_state(self, message):
        self._latest_joint_state = copy.deepcopy(message)
        self._latest_joint_state_monotonic = time.monotonic()

    @staticmethod
    def _collision_geometry_name(xrdf):
        for section_name in ('collision', 'world_collision'):
            section = xrdf.get(section_name)
            if isinstance(section, dict) and section.get('geometry'):
                return section['geometry']
        raise ValueError('XRDF has no collision or world_collision geometry')

    @classmethod
    def _object_spheres_from_xrdf(cls, xrdf):
        geometry_name = cls._collision_geometry_name(xrdf)
        spheres = xrdf.get('geometry', {}).get(geometry_name, {}).get(
            'spheres', {}).get('attached_object')
        if spheres is None:
            return None
        if not isinstance(spheres, list) or not spheres or len(spheres) > 1000:
            raise ValueError('attached_object spheres must be a non-empty list')
        validated = []
        for sphere in spheres:
            center = sphere.get('center') if isinstance(sphere, dict) else None
            radius = sphere.get('radius') if isinstance(sphere, dict) else None
            if (
                not isinstance(center, list)
                or len(center) != 3
                or not all(math.isfinite(float(value)) for value in center)
                or radius is None
                or not math.isfinite(float(radius))
                or float(radius) <= 0.0
            ):
                raise ValueError('attached_object contains an invalid sphere')
            validated.append({
                'center': [float(value) for value in center],
                'radius': float(radius),
            })
        return validated

    def _apply_attachment_overlay(self, xrdf):
        result = copy.deepcopy(xrdf)
        attachment_modifier = None
        for modifier in result.get('modifiers', []):
            add_frame = modifier.get('add_frame', {})
            if add_frame.get('frame_name') == 'attached_object':
                attachment_modifier = add_frame
                break
        if attachment_modifier is None:
            raise ValueError('XRDF has no attached_object frame modifier')
        attachment_modifier['parent_frame_name'] = self._attachment_frame
        geometry_name = self._collision_geometry_name(result)
        sphere_map = result['geometry'][geometry_name].setdefault('spheres', {})
        ignore_map = result.setdefault('self_collision', {}).setdefault(
            'ignore', {})
        if self._attached_object_spheres is None:
            sphere_map.pop('attached_object', None)
            ignore_map.pop('attached_object', None)
        else:
            sphere_map['attached_object'] = copy.deepcopy(
                self._attached_object_spheres)
            ignore_map['attached_object'] = copy.deepcopy(
                self._attachment_ignore_by_frame[self._attachment_frame])
        return result

    @classmethod
    def _apply_grasp_contact_overlay(cls, xrdf, attachment_frame):
        """Remove only intentional finger-contact spheres from one model."""
        result = copy.deepcopy(xrdf)
        geometry_name = cls._collision_geometry_name(result)
        sphere_map = result['geometry'][geometry_name].setdefault('spheres', {})
        contact_links = cls._grasp_contact_links_by_frame[attachment_frame]
        missing_links = [link for link in contact_links if link not in sphere_map]
        if missing_links:
            raise ValueError(
                f'Grasp-contact links are missing from XRDF geometry: '
                f'{missing_links}'
            )
        for link in contact_links:
            sphere_map.pop(link)
        return result

    def _on_set_grasp_contact_mode(self, request, response):
        if not self._planning_lock.acquire(blocking=False):
            response.success = False
            response.message = 'A planning request is active'
            return response
        try:
            with self._attachment_transaction_guard:
                if self._attachment_transaction_active:
                    response.success = False
                    response.message = 'An attachment transaction is active'
                    return response
            with self._grasp_contact_guard:
                if not request.enable:
                    self._armed_grasp_contact = None
                    response.success = True
                    response.message = 'Pending grasp-contact request cleared'
                    return response
                if request.planning_group not in self._enabled_groups:
                    response.success = False
                    response.message = (
                        f'Unsupported planning group {request.planning_group!r}; '
                        f'enabled groups: {self._enabled_groups}'
                    )
                    return response
                if request.attachment_frame not in self._grasp_contact_links_by_frame:
                    response.success = False
                    response.message = (
                        f'Unsupported grasp frame {request.attachment_frame!r}; '
                        f'allowed: {sorted(self._grasp_contact_links_by_frame)}'
                    )
                    return response
                frame_side = (
                    'arm_l' if request.attachment_frame == 'end_effector_l_link'
                    else 'arm_r'
                )
                if request.planning_group in ('arm_l', 'arm_r'):
                    if request.planning_group != frame_side:
                        response.success = False
                        response.message = (
                            f'{request.attachment_frame!r} does not belong to '
                            f'{request.planning_group!r}'
                        )
                        return response
                if self._armed_grasp_contact is not None:
                    response.success = False
                    response.message = 'A grasp-contact request is already armed'
                    return response
                self._armed_grasp_contact = (
                    request.planning_group,
                    request.attachment_frame,
                )
            response.success = True
            response.message = (
                'The next matching plan will use temporary contact spheres for '
                f'{request.attachment_frame}'
            )
            return response
        finally:
            self._planning_lock.release()

    def _consume_grasp_contact(self, group):
        with self._grasp_contact_guard:
            if self._armed_grasp_contact is None:
                return None
            armed_group, attachment_frame = self._armed_grasp_contact
            if armed_group != group:
                return None
            self._armed_grasp_contact = None
            return attachment_frame

    def _on_set_attachment_frame(self, request, response):
        if request.attachment_frame not in self._attachment_ignore_by_frame:
            response.success = False
            response.message = (
                f'Unsupported attachment frame {request.attachment_frame!r}; '
                f'allowed: {sorted(self._attachment_ignore_by_frame)}')
            return response
        if self._attached_object_spheres is not None:
            response.success = False
            response.message = 'Detach the current object before changing its frame'
            return response
        self._attachment_frame = request.attachment_frame
        response.success = True
        response.message = f'Attachment frame selected: {self._attachment_frame}'
        return response

    def _on_attachment_get_description(self, _request, response):
        # NVIDIA's attachment node caches this clean baseline and restores it
        # on detach. Model switching is handled by the overlay below.
        response.urdf = self._urdf
        response.xrdf = yaml.safe_dump(
            self._attachment_description_template, sort_keys=False)
        return response

    def _on_segmentation_get_description(self, _request, response):
        response.urdf = self._urdf
        response.xrdf = yaml.safe_dump(
            self._apply_attachment_overlay(self._segmentation_xrdf_template),
            sort_keys=False,
        )
        return response

    def _on_segmentation_reload_ack(self, message):
        generation_text, separator, node_name = message.data.partition('|')
        if not separator:
            self.get_logger().warning(
                f'Ignoring malformed segmenter reload acknowledgement: '
                f'{message.data!r}')
            return
        try:
            generation = int(generation_text)
        except ValueError:
            self.get_logger().warning(
                f'Ignoring malformed segmenter generation: {message.data!r}')
            return
        with self._segmentation_ack_condition:
            if (
                generation != self._pending_segmentation_generation
                or node_name not in self._expected_segmenters
            ):
                return
            self._segmentation_acks.add(node_name)
            self._segmentation_ack_condition.notify_all()

    def _reload_segmentation_and_wait(self):
        if not self._segmentation_reload_required:
            return
        deadline = time.monotonic() + self._segmentation_reload_timeout_sec
        while (
            self._segmentation_reload.get_subscription_count()
            < len(self._expected_segmenters)
        ):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    'Robot segmenter reload subscribers are unavailable: '
                    f'expected {len(self._expected_segmenters)}, found '
                    f'{self._segmentation_reload.get_subscription_count()}')
            time.sleep(0.05)

        with self._segmentation_ack_condition:
            self._segmentation_generation += 1
            generation = self._segmentation_generation
            self._pending_segmentation_generation = generation
            self._segmentation_acks.clear()
            self._segmentation_reload.publish(UInt64(data=generation))
            while self._segmentation_acks != self._expected_segmenters:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    missing = sorted(
                        self._expected_segmenters - self._segmentation_acks)
                    self._pending_segmentation_generation = 0
                    raise RuntimeError(
                        'Timed out waiting for attached-object segmentation '
                        f'reload generation {generation}; missing: {missing}')
                self._segmentation_ack_condition.wait(timeout=remaining)
            self._pending_segmentation_generation = 0
        self.get_logger().info(
            f'All robot segmenters acknowledged description generation '
            f'{generation}: {sorted(self._expected_segmenters)}')

    async def _set_backend_description(self, urdf, xrdf):
        if not self._set_description.service_is_ready():
            if not self._set_description.wait_for_service(timeout_sec=5.0):
                raise RuntimeError('cuMotion robot-description service unavailable')
        backend_request = SetRobotDescription.Request()
        backend_request.urdf = urdf
        backend_request.xrdf = xrdf
        backend_response = await self._set_description.call_async(backend_request)
        if backend_response is None or not backend_response.success:
            message = (
                backend_response.message
                if backend_response is not None else 'no response')
            raise RuntimeError(f'cuMotion rejected robot description: {message}')

    def _on_attachment_begin(self, _request, response):
        with self._attachment_transaction_guard:
            if self._attachment_transaction_active:
                response.success = False
                response.message = 'Another attachment transaction is active'
                return response
            with self._grasp_contact_guard:
                if self._armed_grasp_contact is not None:
                    response.success = False
                    response.message = 'A grasp-contact request is armed'
                    return response
            if not self._planning_lock.acquire(blocking=False):
                response.success = False
                response.message = 'A planning request is active'
                return response
            try:
                self._attachment_transaction_snapshot = {
                    'spheres': copy.deepcopy(self._attached_object_spheres),
                    'loaded_urdf': self._loaded_urdf,
                    'loaded_xrdf': self._loaded_xrdf,
                    'backend_urdf': self._loaded_urdf or self._urdf,
                    'backend_xrdf': (
                        self._loaded_xrdf or self._startup_backend_xrdf),
                }
                self._attachment_transaction_active = True
            finally:
                self._planning_lock.release()
        response.success = True
        response.message = 'Attachment transaction started; planning is blocked'
        return response

    def _on_attachment_commit(self, _request, response):
        with self._attachment_transaction_guard:
            if not self._attachment_transaction_active:
                response.success = False
                response.message = 'No attachment transaction is active'
                return response
            self._attachment_transaction_active = False
            self._attachment_transaction_snapshot = None
        response.success = True
        response.message = 'Attachment transaction committed; planning is enabled'
        return response

    async def _on_attachment_rollback(self, _request, response):
        with self._attachment_transaction_guard:
            if not self._attachment_transaction_active:
                response.success = False
                response.message = 'No attachment transaction is active'
                return response
            snapshot = copy.deepcopy(self._attachment_transaction_snapshot)
        if not self._planning_lock.acquire(blocking=False):
            response.success = False
            response.message = (
                'Cannot roll back while planning is active; planning remains blocked')
            return response
        try:
            self._attached_object_spheres = snapshot['spheres']
            await self._set_backend_description(
                snapshot['backend_urdf'], snapshot['backend_xrdf'])
            self._loaded_urdf = snapshot['loaded_urdf']
            self._loaded_xrdf = snapshot['loaded_xrdf']
            self._reload_segmentation_and_wait()
        except RuntimeError as error:
            response.success = False
            response.message = (
                f'Attachment rollback failed: {error}; planning remains blocked')
            return response
        finally:
            self._planning_lock.release()
        with self._attachment_transaction_guard:
            self._attachment_transaction_active = False
            self._attachment_transaction_snapshot = None
        response.success = True
        response.message = 'Attachment transaction rolled back; planning is enabled'
        return response

    async def _on_attachment_set_description(self, request, response):
        with self._attachment_transaction_guard:
            if not self._attachment_transaction_active:
                response.success = False
                response.message = (
                    'Attachment update rejected outside a guarded transaction; '
                    'use the predefined attachment services')
                return response
        if request.urdf != self._urdf:
            response.success = False
            response.message = 'Attachment update attempted to replace the canonical URDF'
            return response
        try:
            requested_xrdf = yaml.safe_load(request.xrdf)
            requested_spheres = self._object_spheres_from_xrdf(requested_xrdf)
        except (TypeError, ValueError, yaml.YAMLError) as error:
            response.success = False
            response.message = f'Invalid attachment XRDF: {error}'
            return response

        if not self._planning_lock.acquire(blocking=False):
            response.success = False
            response.message = 'A planning request is active; retry attachment update'
            return response
        previous_spheres = self._attached_object_spheres
        try:
            self._attached_object_spheres = requested_spheres
            backend_urdf = self._loaded_urdf or self._urdf
            if self._loaded_xrdf is None:
                if self._latest_joint_state is None:
                    updated_model = self._apply_attachment_overlay(
                        self._xrdf_template_by_group[self._active_group])
                    updated_xrdf = yaml.safe_dump(
                        updated_model, sort_keys=False)
                else:
                    updated_xrdf, _, _, _ = self._xrdf_with_live_frozen_joints(
                        self._active_group, self._latest_joint_state)
            else:
                updated_model = self._apply_attachment_overlay(
                    yaml.safe_load(self._loaded_xrdf))
                updated_xrdf = yaml.safe_dump(updated_model, sort_keys=False)
            await self._set_backend_description(backend_urdf, updated_xrdf)
            self._loaded_urdf = backend_urdf
            self._loaded_xrdf = updated_xrdf
            self._reload_segmentation_and_wait()
        except (RuntimeError, TypeError, ValueError, yaml.YAMLError) as error:
            self._attached_object_spheres = previous_spheres
            response.success = False
            response.message = str(error)
            return response
        finally:
            self._planning_lock.release()

        response.success = True
        if requested_spheres is None:
            response.message = 'Detached object overlay from all planning groups'
        else:
            response.message = (
                f'Applied {len(requested_spheres)} attached-object spheres '
                f'at {self._attachment_frame} to all planning groups')
        return response

    def _goal(self, goal_request):
        with self._attachment_transaction_guard:
            if self._attachment_transaction_active:
                self.get_logger().warning(
                    'Rejecting planning request while attachment/ESDF update '
                    'is incomplete')
                return GoalResponse.REJECT
        group = goal_request.request.group_name
        with self._grasp_contact_guard:
            if (
                self._armed_grasp_contact is not None
                and self._armed_grasp_contact[0] != group
            ):
                self.get_logger().warning(
                    f'Rejecting {group} request while one-shot grasp contact is '
                    f'armed for {self._armed_grasp_contact[0]}'
                )
                return GoalResponse.REJECT
        if group not in self._enabled_groups:
            self.get_logger().error(
                f'cuMotion planning group {group!r} is not enabled; '
                f'enabled groups: {self._enabled_groups}')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel(_goal_handle):
        return CancelResponse.ACCEPT

    @staticmethod
    def _failure_result():
        result = MoveGroup.Result()
        result.error_code.val = MoveItErrorCodes.FAILURE
        return result

    def _clear_goal_collision_spheres(self):
        self._sphere_generation += 1
        self._pending_sphere_request = None
        marker = Marker()
        marker.action = Marker.DELETEALL
        self._goal_collision_spheres.publish(MarkerArray(markers=[marker]))

    def _on_interactive_marker_update(self, update):
        if update.type == InteractiveMarkerUpdate.KEEP_ALIVE:
            return
        self._interactive_marker_names.difference_update(update.erases)
        self._interactive_marker_names.update(marker.name for marker in update.markers)

        goal_tips = {
            name[len('EE:goal_'):]
            for name in self._interactive_marker_names
            if name.startswith('EE:goal_')
        }
        if any(tip.startswith('end_effector_') for tip in goal_tips):
            group = 'wholebody'
        elif {'arm_l_link7', 'arm_r_link7'} <= goal_tips:
            group = 'both_arms'
        elif 'arm_l_link7' in goal_tips:
            group = 'arm_l'
        elif 'arm_r_link7' in goal_tips:
            group = 'arm_r'
        else:
            group = None

        if group != self._interactive_group:
            self._interactive_group = group
            self._interactive_goal_state = None
            self._clear_goal_collision_spheres()
            if group is not None:
                self.get_logger().info(
                    f'Interactive goal collision-sphere group: {group}'
                )

    def _on_query_goal_spheres(self, request, response):
        solution = request.robot_state.joint_state
        group = solution.header.frame_id
        sphere_group = self._interactive_group or group
        if group not in self._enabled_groups or sphere_group not in self._enabled_groups:
            response.error_code.val = MoveItErrorCodes.INVALID_GROUP_NAME
            return response
        seed = self._interactive_goal_state or self._latest_joint_state
        if seed is None:
            response.error_code.val = MoveItErrorCodes.INVALID_ROBOT_STATE
            return response
        merged = copy.deepcopy(seed)
        positions = dict(zip(merged.name, merged.position))
        positions.update(zip(solution.name, solution.position))
        merged.name = list(positions)
        merged.position = list(positions.values())
        merged.velocity = []
        merged.effort = []
        self._interactive_goal_state = merged
        self._schedule_fk_spheres(sphere_group, merged)
        response.error_code.val = MoveItErrorCodes.SUCCESS
        return response

    def _schedule_fk_spheres(self, group, joint_state):
        self._sphere_generation += 1
        self._pending_sphere_request = (
            self._sphere_generation,
            group,
            copy.deepcopy(joint_state),
        )

    def _flush_fk_spheres(self):
        pending = self._pending_sphere_request
        if pending is None:
            return
        if not self._compute_fk.service_is_ready():
            return
        self._pending_sphere_request = None
        generation, group, joint_state = pending
        request = GetPositionFK.Request()
        request.header.frame_id = 'base_link'
        request.fk_link_names = list(self._collision_spheres_by_group[group])
        request.robot_state = RobotState(
            joint_state=copy.deepcopy(joint_state)
        )
        future = self._compute_fk.call_async(request)
        future.add_done_callback(
            lambda result, selected_group=group, request_generation=generation: (
                self._on_fk_spheres_result(
                    selected_group, request_generation, result
                )
            )
        )

    def _on_fk_spheres_result(self, group, generation, future):
        if generation != self._sphere_generation:
            return
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().warning(f'Goal collision-sphere FK failed: {error}')
            return
        if generation == self._sphere_generation:
            self._publish_fk_spheres(group, response)

    @staticmethod
    def _rotate_point(point, orientation):
        x, y, z = point
        qx, qy, qz, qw = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        tx = 2.0 * (qy * z - qz * y)
        ty = 2.0 * (qz * x - qx * z)
        tz = 2.0 * (qx * y - qy * x)
        return (
            x + qw * tx + qy * tz - qz * ty,
            y + qw * ty + qz * tx - qx * tz,
            z + qw * tz + qx * ty - qy * tx,
        )

    def _publish_fk_spheres(self, group, response):
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().warning('Failed to compute goal collision-sphere poses')
            return

        spheres = self._collision_spheres_by_group[group]
        markers = []
        marker_id = 0
        for link_name, pose_stamped in zip(
            response.fk_link_names, response.pose_stamped
        ):
            link_pose = pose_stamped.pose
            for sphere in spheres[link_name]:
                offset = self._rotate_point(sphere['center'], link_pose.orientation)
                marker = Marker()
                marker.header = pose_stamped.header
                marker.ns = 'cumotion_goal_collision_spheres'
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = link_pose.position.x + offset[0]
                marker.pose.position.y = link_pose.position.y + offset[1]
                marker.pose.position.z = link_pose.position.z + offset[2]
                marker.pose.orientation.w = 1.0
                diameter = 2.0 * float(sphere['radius'])
                marker.scale.x = diameter
                marker.scale.y = diameter
                marker.scale.z = diameter
                marker.color.r = 0.95
                marker.color.g = 0.25
                marker.color.b = 0.05
                marker.color.a = 0.35
                markers.append(marker)
                marker_id += 1
        self._goal_collision_spheres.publish(MarkerArray(markers=markers))
        self.get_logger().info(
            f'Published {len(markers)} collision spheres at the {group} goal pose'
        )

    def _clamp_limit_roundoff(self, name, position):
        limits = self._position_limits.get(name)
        if limits is None:
            return position
        lower, upper = limits
        boundary_tolerance = 1.0e-6
        if lower - boundary_tolerance <= position < lower:
            return lower
        if upper < position <= upper + boundary_tolerance:
            return upper
        return position

    def _clamp_auxiliary_position(self, name, position):
        limits = self._position_limits.get(name)
        if limits is None:
            return position
        lower, upper = limits
        if lower <= position <= upper:
            return position
        clamped = min(max(position, lower), upper)
        violation = abs(position - clamped)
        if violation > self._auxiliary_joint_limit_tolerance:
            raise ValueError(
                f'Live auxiliary joint {name!r} is {position:.6f}, outside '
                f'effective cuMotion limits [{lower:.6f}, {upper:.6f}] by '
                f'{violation:.6f}; refusing to reload the backend model'
            )
        return clamped

    def _xrdf_with_live_frozen_joints(self, group, joint_state):
        """Set non-cspace XRDF defaults from a live full-robot joint state."""
        xrdf = self._apply_attachment_overlay(
            self._xrdf_template_by_group[group])
        defaults = xrdf.get('default_joint_positions', {})
        cspace_joints = set(xrdf['cspace']['joint_names'])
        live_positions = {
            name: float(joint_state.position[index])
            for index, name in enumerate(joint_state.name)
            if index < len(joint_state.position)
            and math.isfinite(float(joint_state.position[index]))
        }
        frozen_joints = set(defaults) - cspace_joints
        updated_joints = []
        clamped_joints = []
        for name in defaults:
            if name not in frozen_joints or name not in live_positions:
                continue
            raw_position = live_positions[name]
            safe_position = self._clamp_auxiliary_position(name, raw_position)
            defaults[name] = safe_position
            if safe_position != raw_position:
                clamped_joints.append((name, raw_position, safe_position))
            updated_joints.append(name)

        missing_joints = sorted(frozen_joints - set(live_positions))
        return (
            yaml.safe_dump(xrdf, sort_keys=False),
            updated_joints,
            missing_joints,
            clamped_joints,
        )

    def _normalize_goal(self, move_group_goal, group):
        """Use a live snapshot and filter the backend goal to its c-space."""
        goal = copy.deepcopy(move_group_goal)
        request = goal.request
        joint_state = request.start_state.joint_state
        if self._force_current_start_state:
            if (self._latest_joint_state is None
                    or self._latest_joint_state_monotonic is None):
                self.get_logger().error(
                    'Cannot plan without a current joint state')
                return None
            state_age = time.monotonic() - self._latest_joint_state_monotonic
            if state_age > float(self._max_current_state_age_sec):
                self.get_logger().error(
                    f'Current joint state is stale ({state_age:.3f}s old); '
                    'rejecting planning request')
                return None
            requested_positions = dict(zip(joint_state.name, joint_state.position))
            current_positions = dict(zip(
                self._latest_joint_state.name,
                self._latest_joint_state.position,
            ))
            differences = {
                name: abs(requested_positions[name] - current_positions[name])
                for name in self._joints_by_group[group]
                if name in requested_positions and name in current_positions
            }
            if differences:
                worst_joint = max(differences, key=differences.get)
                worst_difference = differences[worst_joint]
                if worst_difference > 0.01:
                    self.get_logger().warning(
                        'Replacing stale requested start state with current '
                        f'joint state; largest difference is '
                        f'{worst_difference:.4f} rad at {worst_joint}'
                    )
            joint_state = copy.deepcopy(self._latest_joint_state)
            request.start_state.joint_state = joint_state
            request.start_state.is_diff = False
        elif not joint_state.name and self._latest_joint_state is not None:
            joint_state = copy.deepcopy(self._latest_joint_state)
            request.start_state.joint_state = joint_state
            request.start_state.is_diff = False
        index_by_name = {name: index for index, name in enumerate(joint_state.name)}
        group_joints = self._joints_by_group[group]
        missing = [name for name in group_joints if name not in index_by_name]
        if missing:
            self.get_logger().error(
                f'Start state for {group} is missing joints: {missing}'
            )
            return None

        full_joint_state = copy.deepcopy(joint_state)
        indices = [index_by_name[name] for name in group_joints]
        original_count = len(joint_state.name)
        joint_state.name = list(group_joints)
        joint_state.position = [joint_state.position[index] for index in indices]
        for index, name in enumerate(group_joints):
            joint_state.position[index] = self._clamp_limit_roundoff(
                name, joint_state.position[index]
            )
        for field in ('velocity', 'effort'):
            values = getattr(joint_state, field)
            setattr(
                joint_state,
                field,
                [values[index] for index in indices]
                if len(values) == original_count else [],
            )

        def filter_joint_constraints(constraints):
            constraints.joint_constraints = [
                constraint
                for constraint in constraints.joint_constraints
                if constraint.joint_name in group_joints
            ]
            for constraint in constraints.joint_constraints:
                constraint.position = self._clamp_limit_roundoff(
                    constraint.joint_name, constraint.position
                )

        for constraints in request.goal_constraints:
            filter_joint_constraints(constraints)
        filter_joint_constraints(request.path_constraints)
        for constraints in request.trajectory_constraints.constraints:
            filter_joint_constraints(constraints)
        return goal, full_joint_state

    def _tool_target_from_goal(self, goal, group):
        for constraints in goal.request.goal_constraints:
            if constraints.position_constraints:
                target = constraints.position_constraints[0].link_name
                return self._equivalent_tool_frames.get(target, target)
            if constraints.orientation_constraints:
                target = constraints.orientation_constraints[0].link_name
                return self._equivalent_tool_frames.get(target, target)
        return self._default_tool_by_group[group]

    def _urdf_with_backend_tool_alias(self, target_tool_frame):
        if target_tool_frame == self._backend_tool_frame:
            return self._urdf

        root = ET.fromstring(self._urdf)
        link_names = {link.attrib.get('name') for link in root.findall('link')}
        if target_tool_frame not in link_names:
            raise ValueError(
                f'Tool target {target_tool_frame!r} is not present in the URDF'
            )

        for joint in root.findall('joint'):
            child = joint.find('child')
            if (
                child is None
                or child.attrib.get('link') != self._backend_tool_frame
            ):
                continue
            parent = joint.find('parent')
            origin = joint.find('origin')
            if parent is None:
                raise ValueError(
                    f'Backend tool joint for {self._backend_tool_frame!r} '
                    'has no parent'
                )
            parent.set('link', target_tool_frame)
            if origin is None:
                origin = ET.SubElement(joint, 'origin')
            origin.set('xyz', '0 0 0')
            origin.set('rpy', '0 0 0')
            return ET.tostring(root, encoding='unicode')

        raise ValueError(
            f'No URDF joint found for backend tool '
            f'{self._backend_tool_frame!r}'
        )

    def _rewrite_tool_constraints(self, goal, target_tool_frame):
        if target_tool_frame == self._backend_tool_frame:
            return

        def rewrite(constraints):
            for constraint in constraints.position_constraints:
                target = self._equivalent_tool_frames.get(
                    constraint.link_name, constraint.link_name
                )
                if target == target_tool_frame:
                    constraint.link_name = self._backend_tool_frame
            for constraint in constraints.orientation_constraints:
                target = self._equivalent_tool_frames.get(
                    constraint.link_name, constraint.link_name
                )
                if target == target_tool_frame:
                    constraint.link_name = self._backend_tool_frame

        for constraints in goal.request.goal_constraints:
            rewrite(constraints)
        rewrite(goal.request.path_constraints)
        for constraints in goal.request.trajectory_constraints.constraints:
            rewrite(constraints)

    async def _select_group(
        self, group, tool_target, urdf, xrdf, updated_joints
    ):
        if (
            group == self._active_group
            and urdf == self._loaded_urdf
            and xrdf == self._loaded_xrdf
        ):
            return True
        if not self._set_description.service_is_ready():
            if not self._set_description.wait_for_service(timeout_sec=30.0):
                self.get_logger().error('cuMotion robot-description service unavailable')
                return False
        request = SetRobotDescription.Request()
        request.urdf = urdf
        request.xrdf = xrdf
        self.get_logger().info(
            f'Loading cuMotion model for {group} (tool: {tool_target}) with '
            f'{len(updated_joints)} live frozen-joint defaults'
        )
        response = await self._set_description.call_async(request)
        if response is None or not response.success:
            message = response.message if response is not None else 'no response'
            self.get_logger().error(f'Failed to switch cuMotion model: {message}')
            return False
        self._active_group = group
        self._loaded_urdf = urdf
        self._loaded_xrdf = xrdf
        self.get_logger().info(
            f'cuMotion model is now: {group} (tool: {tool_target})'
        )
        return True

    async def _execute(self, goal_handle):
        group = goal_handle.request.request.group_name
        with self._attachment_transaction_guard:
            if self._attachment_transaction_active:
                self.get_logger().warning(
                    f'Rejecting {group} request while attachment/ESDF update '
                    'is incomplete')
                goal_handle.abort()
                return self._failure_result()
        if not self._planning_lock.acquire(blocking=False):
            self.get_logger().warning(
                f'Rejecting concurrent {group} request: another plan is still active'
            )
            goal_handle.abort()
            return self._failure_result()
        restore_model = None
        try:
            grasp_contact_frame = self._consume_grasp_contact(group)
            self._clear_goal_collision_spheres()
            normalized = self._normalize_goal(goal_handle.request, group)
            if normalized is None:
                goal_handle.abort()
                return self._failure_result()
            backend_goal, live_joint_state = normalized
            tool_target = self._tool_target_from_goal(backend_goal, group)
            try:
                backend_urdf = self._urdf_with_backend_tool_alias(tool_target)
            except ValueError as error:
                self.get_logger().error(str(error))
                goal_handle.abort()
                return self._failure_result()
            self._rewrite_tool_constraints(backend_goal, tool_target)
            try:
                (
                    normal_xrdf,
                    updated_joints,
                    missing_joints,
                    clamped_joints,
                ) = self._xrdf_with_live_frozen_joints(
                    group, live_joint_state
                )
            except ValueError as error:
                self.get_logger().error(str(error))
                goal_handle.abort()
                return self._failure_result()
            xrdf = normal_xrdf
            if grasp_contact_frame is not None:
                try:
                    contact_model = self._apply_grasp_contact_overlay(
                        yaml.safe_load(normal_xrdf), grasp_contact_frame
                    )
                    xrdf = yaml.safe_dump(contact_model, sort_keys=False)
                except (TypeError, ValueError, yaml.YAMLError) as error:
                    self.get_logger().error(str(error))
                    goal_handle.abort()
                    return self._failure_result()
                restore_model = (backend_urdf, normal_xrdf)
                self.get_logger().info(
                    f'Using one-shot grasp-contact model for {group}: '
                    f'{self._grasp_contact_links_by_frame[grasp_contact_frame]}'
                )
            for name, raw_position, safe_position in clamped_joints:
                self.get_logger().warning(
                    f'Clamped live auxiliary joint {name} from '
                    f'{raw_position:.6f} to {safe_position:.6f} before '
                    'cuMotion model reload'
                )
            if missing_joints:
                self.get_logger().warning(
                    f'Using static XRDF defaults for {group} joints missing '
                    'from '
                    f'the live joint state: {missing_joints}'
                )
            if not await self._select_group(
                group,
                tool_target,
                backend_urdf,
                xrdf,
                updated_joints,
            ):
                goal_handle.abort()
                return self._failure_result()

            if not self._backend.server_is_ready():
                if not self._backend.wait_for_server(timeout_sec=30.0):
                    self.get_logger().error(f'cuMotion backend unavailable for {group}')
                    goal_handle.abort()
                    return self._failure_result()

            def forward_feedback(feedback):
                goal_handle.publish_feedback(feedback.feedback)

            backend_handle = await self._backend.send_goal_async(
                backend_goal, feedback_callback=forward_feedback
            )
            if not backend_handle.accepted:
                self.get_logger().error(f'cuMotion backend rejected {group} goal')
                goal_handle.abort()
                return self._failure_result()

            wrapped_result = await backend_handle.get_result_async()
            if goal_handle.is_cancel_requested:
                await backend_handle.cancel_goal_async()
                goal_handle.canceled()
            elif wrapped_result.result.error_code.val == MoveItErrorCodes.SUCCESS:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return wrapped_result.result
        finally:
            if restore_model is not None:
                restore_urdf, restore_xrdf = restore_model
                try:
                    await self._set_backend_description(
                        restore_urdf, restore_xrdf
                    )
                except RuntimeError as error:
                    # Force the next request to reload the canonical model
                    # before planning; never cache a contact model as normal.
                    self._loaded_urdf = None
                    self._loaded_xrdf = None
                    self.get_logger().error(
                        f'Failed to restore normal collision model after grasp '
                        f'contact: {error}'
                    )
                else:
                    self._loaded_urdf = restore_urdf
                    self._loaded_xrdf = restore_xrdf
                    self.get_logger().info(
                        'Restored normal collision model after one-shot grasp '
                        'contact plan'
                    )
            self._planning_lock.release()


def main(args=None):
    rclpy.init(args=args)
    node = CumotionGroupRouter()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
