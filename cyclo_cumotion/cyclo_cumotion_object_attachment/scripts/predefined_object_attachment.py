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

"""Attach collision geometry selected from a predefined object catalog."""

import math
from pathlib import Path

from action_msgs.msg import GoalStatus
from isaac_ros_cumotion_interfaces.action import AttachObject
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robotis_interfaces.srv import AttachObjectByName, SetAttachmentFrame
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker
import yaml


class PredefinedObjectAttachment(Node):
    """Resolve named objects to markers and call NVIDIA's attachment action."""

    _marker_types = {
        'cube': Marker.CUBE,
        'sphere': Marker.SPHERE,
        'mesh': Marker.MESH_RESOURCE,
    }

    def __init__(self):
        super().__init__('predefined_object_attachment')
        callback_group = ReentrantCallbackGroup()
        catalog_path = Path(self.declare_parameter('catalog_file', '').value)
        self._allowed_frames = set(self.declare_parameter(
            'allowed_attachment_frames', [
                'end_effector_l_link',
                'end_effector_r_link',
            ]).value)
        self._objects = self._load_catalog(catalog_path)
        self._request_in_flight = False

        self._set_attachment_frame = self.create_client(
            SetAttachmentFrame,
            '/cyclo_cumotion/attachment/set_frame',
            callback_group=callback_group,
        )
        self._begin_transaction = self.create_client(
            Trigger,
            '/cyclo_cumotion/attachment/begin_transaction',
            callback_group=callback_group,
        )
        self._commit_transaction = self.create_client(
            Trigger,
            '/cyclo_cumotion/attachment/commit_transaction',
            callback_group=callback_group,
        )
        self._rollback_transaction = self.create_client(
            Trigger,
            '/cyclo_cumotion/attachment/rollback_transaction',
            callback_group=callback_group,
        )
        self._attach_action = ActionClient(
            self,
            AttachObject,
            self.declare_parameter('attach_action_name', '/attach_object').value,
            callback_group=callback_group,
        )
        self._attach_service = self.create_service(
            AttachObjectByName,
            '/cyclo_cumotion/object_attachment/attach_by_name',
            self._attach_by_name,
            callback_group=callback_group,
        )
        self._detach_service = self.create_service(
            Trigger,
            '/cyclo_cumotion/object_attachment/detach',
            self._detach,
            callback_group=callback_group,
        )
        self.get_logger().info(
            f'Loaded {len(self._objects)} predefined attachment objects: '
            f'{sorted(self._objects)}')

    def _load_catalog(self, path):
        if not path.is_file():
            raise ValueError(f'Object catalog does not exist: {path}')
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        objects = data.get('objects')
        if not isinstance(objects, dict) or not objects:
            raise ValueError('Object catalog must contain a non-empty objects map')
        for name, config in objects.items():
            self._validate_object(name, config)
        return objects

    def _validate_object(self, name, config):
        if not isinstance(config, dict):
            raise ValueError(f'Object {name!r} must be a map')
        shape = str(config.get('type', '')).lower()
        if shape not in self._marker_types:
            raise ValueError(
                f'Object {name!r} has unsupported type {shape!r}')
        pose = config.get('pose', {})
        position = pose.get('position', [])
        orientation = pose.get('orientation', [])
        scale = config.get('scale', [])
        if not self._finite_vector(position, 3):
            raise ValueError(f'Object {name!r} position must contain 3 values')
        if not self._finite_vector(orientation, 4):
            raise ValueError(f'Object {name!r} orientation must contain 4 values')
        norm = math.sqrt(sum(float(value) ** 2 for value in orientation))
        if norm < 1.0e-9:
            raise ValueError(f'Object {name!r} orientation has zero length')
        if not self._finite_vector(scale, 3) or any(
                float(value) <= 0.0 for value in scale):
            raise ValueError(f'Object {name!r} scale must contain 3 positive values')
        if shape == 'mesh' and not config.get('mesh_resource'):
            raise ValueError(f'Object {name!r} mesh_resource is empty')
        default_frame = config.get('default_attachment_frame', '')
        if default_frame and default_frame not in self._allowed_frames:
            raise ValueError(
                f'Object {name!r} uses unsupported attachment frame '
                f'{default_frame!r}')

    @staticmethod
    def _finite_vector(values, length):
        return (
            isinstance(values, list)
            and len(values) == length
            and all(math.isfinite(float(value)) for value in values)
        )

    def _marker(self, object_name, attachment_frame):
        config = self._objects[object_name]
        pose = config['pose']
        orientation = [float(value) for value in pose['orientation']]
        norm = math.sqrt(sum(value ** 2 for value in orientation))

        marker = Marker()
        # NVIDIA's orchestration expresses the object pose in the grasp frame.
        # The router maps that same convention to the selected end-effector.
        marker.header.frame_id = attachment_frame
        marker.ns = 'predefined_attached_object'
        marker.type = self._marker_types[str(config['type']).lower()]
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            float(value) for value in pose['position'])
        (
            marker.pose.orientation.x,
            marker.pose.orientation.y,
            marker.pose.orientation.z,
            marker.pose.orientation.w,
        ) = (value / norm for value in orientation)
        marker.scale.x, marker.scale.y, marker.scale.z = (
            float(value) for value in config['scale'])
        marker.mesh_resource = str(config.get('mesh_resource', ''))
        marker.frame_locked = True
        return marker

    async def _attach_by_name(self, request, response):
        if self._request_in_flight:
            return self._failure(response, 'Another attachment request is active')
        if request.object_name not in self._objects:
            return self._failure(
                response,
                f'Unknown object {request.object_name!r}; available objects: '
                f'{sorted(self._objects)}',
            )
        config = self._objects[request.object_name]
        attachment_frame = (
            request.attachment_frame
            or str(config.get('default_attachment_frame', ''))
        )
        if attachment_frame not in self._allowed_frames:
            return self._failure(
                response,
                f'Unsupported attachment frame {attachment_frame!r}; allowed: '
                f'{sorted(self._allowed_frames)}',
            )

        self._request_in_flight = True
        try:
            frame_response = await self._select_attachment_frame(
                attachment_frame)
            if not frame_response.success:
                return self._failure(response, frame_response.message)
            goal = AttachObject.Goal()
            goal.attach_object = True
            goal.object_config = self._marker(
                request.object_name, attachment_frame)
            return await self._run_guarded_action(
                goal, response, request.object_name)
        finally:
            self._request_in_flight = False

    async def _select_attachment_frame(self, attachment_frame):
        if not self._set_attachment_frame.service_is_ready():
            if not self._set_attachment_frame.wait_for_service(timeout_sec=2.0):
                response = SetAttachmentFrame.Response()
                response.success = False
                response.message = 'Attachment-frame service is unavailable'
                return response
        request = SetAttachmentFrame.Request()
        request.attachment_frame = attachment_frame
        return await self._set_attachment_frame.call_async(request)

    async def _detach(self, _request, response):
        if self._request_in_flight:
            return self._failure(response, 'Another attachment request is active')
        self._request_in_flight = True
        try:
            goal = AttachObject.Goal()
            goal.attach_object = False
            return await self._run_guarded_action(goal, response, 'object')
        finally:
            self._request_in_flight = False

    async def _send_goal(self, goal, response, label):
        if not self._attach_action.server_is_ready():
            if not self._attach_action.wait_for_server(timeout_sec=2.0):
                return self._failure(
                    response, 'Isaac ROS AttachObject action is unavailable')
        goal_handle = await self._attach_action.send_goal_async(goal)
        if not goal_handle.accepted:
            return self._failure(response, 'AttachObject action rejected the request')
        wrapped_result = await goal_handle.get_result_async()
        response.success = wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
        response.message = wrapped_result.result.outcome
        if response.success:
            verb = 'Attached' if goal.attach_object else 'Detached'
            self.get_logger().info(f'{verb} {label}: {response.message}')
        return response

    async def _call_transaction_service(self, client, unavailable_message):
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=2.0):
                result = Trigger.Response()
                result.success = False
                result.message = unavailable_message
                return result
        return await client.call_async(Trigger.Request())

    async def _run_guarded_action(self, goal, response, label):
        begin = await self._call_transaction_service(
            self._begin_transaction,
            'Attachment transaction service is unavailable',
        )
        if not begin.success:
            return self._failure(response, begin.message)

        action_response = None
        try:
            action_response = await self._send_goal(goal, response, label)
            if action_response.success:
                commit = await self._call_transaction_service(
                    self._commit_transaction,
                    'Attachment commit service is unavailable',
                )
                if commit.success:
                    return action_response
                action_response.success = False
                action_response.message = (
                    f'{action_response.message}; commit failed: {commit.message}')

            rollback = await self._call_transaction_service(
                self._rollback_transaction,
                'Attachment rollback service is unavailable',
            )
            if not rollback.success:
                action_response.message = (
                    f'{action_response.message}; CRITICAL: {rollback.message}')
            return action_response
        except Exception as error:  # noqa: BLE001
            rollback = await self._call_transaction_service(
                self._rollback_transaction,
                'Attachment rollback service is unavailable',
            )
            detail = f'Attachment action raised an exception: {error}'
            if not rollback.success:
                detail += f'; CRITICAL: {rollback.message}'
            return self._failure(response, detail)

    @staticmethod
    def _failure(response, message):
        response.success = False
        response.message = message
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PredefinedObjectAttachment()
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
