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
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState


class JointStateSanitizer(Node):
    """Normalize platform joint states for the common planning model."""

    def __init__(self):
        super().__init__('joint_state_sanitizer')
        urdf = Path(self.declare_parameter('urdf_path', '').value).read_text(
            encoding='utf-8'
        )
        self._tolerance = self.declare_parameter(
            'limit_roundoff_tolerance', 1.0e-4
        ).value
        input_topic = self.declare_parameter(
            'input_topic', '/joint_states'
        ).value
        output_topic = self.declare_parameter(
            'output_topic', '/isaac_ros/joint_states'
        ).value
        if input_topic == output_topic:
            raise ValueError('joint-state sanitizer input and output must differ')

        root = ET.fromstring(urdf)
        known_joints = {joint.attrib['name'] for joint in root.findall('joint')}
        default_joint_names = list(
            self.declare_parameter(
                'default_joint_names', Parameter.Type.STRING_ARRAY
            ).value
        )
        default_joint_positions = list(
            self.declare_parameter(
                'default_joint_positions', Parameter.Type.DOUBLE_ARRAY
            ).value
        )
        if len(default_joint_names) != len(default_joint_positions):
            raise ValueError(
                'default_joint_names and default_joint_positions must have '
                'the same length'
            )
        unknown_defaults = set(default_joint_names) - known_joints
        if unknown_defaults:
            raise ValueError(
                'default joint is not present in the URDF: '
                + ', '.join(sorted(unknown_defaults))
            )
        self._default_positions = dict(
            zip(default_joint_names, default_joint_positions)
        )

        self._limits = {}
        for joint in root.findall('joint'):
            limit = joint.find('limit')
            if limit is None or 'lower' not in limit.attrib or 'upper' not in limit.attrib:
                continue
            self._limits[joint.attrib['name']] = (
                float(limit.attrib['lower']),
                float(limit.attrib['upper']),
            )

        self._last_warning = {}
        self._publisher = self.create_publisher(JointState, output_topic, 50)
        self._subscription = self.create_subscription(
            JointState, input_topic, self._on_joint_state, 50
        )
        self.get_logger().info(
            f'Sanitizing {input_topic} -> {output_topic} '
            f'(limit round-off tolerance={self._tolerance:g})'
        )

    def _warn_clamp(self, name, original, clamped):
        now = time.monotonic()
        if now - self._last_warning.get(name, -math.inf) < 5.0:
            return
        self._last_warning[name] = now
        self.get_logger().warning(
            f'Clamped small joint-limit round-off for {name}: '
            f'{original:.12g} -> {clamped:.12g}'
        )

    def _on_joint_state(self, message):
        sanitized = copy.deepcopy(message)
        names = list(sanitized.name)
        positions = list(sanitized.position)
        velocities = list(sanitized.velocity)
        efforts = list(sanitized.effort)

        present = set(names)
        for name, position in self._default_positions.items():
            if name in present:
                continue
            names.append(name)
            positions.append(position)
            if velocities:
                velocities.append(0.0)
            if efforts:
                efforts.append(0.0)

        for index, name in enumerate(names):
            if index >= len(positions) or name not in self._limits:
                continue
            value = positions[index]
            lower, upper = self._limits[name]
            if lower - self._tolerance <= value < lower:
                positions[index] = lower
                self._warn_clamp(name, value, lower)
            elif upper < value <= upper + self._tolerance:
                positions[index] = upper
                self._warn_clamp(name, value, upper)
        sanitized.name = names
        sanitized.position = positions
        sanitized.velocity = velocities
        sanitized.effort = efforts
        self._publisher.publish(sanitized)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateSanitizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
