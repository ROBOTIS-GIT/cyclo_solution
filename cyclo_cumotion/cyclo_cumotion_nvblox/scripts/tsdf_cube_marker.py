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

"""Convert incremental nvblox TSDF blocks to persistent RViz cubes."""

from geometry_msgs.msg import Point

from nvblox_msgs.msg import VoxelBlockLayer

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import ColorRGBA

from visualization_msgs.msg import Marker, MarkerArray


class TsdfCubeMarker(Node):
    """Publish changed TSDF blocks while retaining a snapshot for new RViz clients."""

    def __init__(self):
        """Create the TSDF subscriber and transient-local marker publisher."""
        super().__init__('tsdf_cube_marker')

        self.declare_parameter('input_topic', '/nvblox_node/tsdf_layer')
        self.declare_parameter(
            'output_topic', '/cyclo_cumotion/nvblox/tsdf_cube_markers'
        )
        self.declare_parameter('voxel_scale', 1.0)

        self._input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self._voxel_scale = self.get_parameter('voxel_scale').value
        self._blocks = {}
        self._block_ids = {}
        self._next_marker_id = 0
        self._last_header = None
        self._voxel_size = None
        self._last_subscription_count = 0
        self._reported_first_map = False

        self._input_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            MarkerArray, output_topic, output_qos
        )
        self._subscription = None
        self._subscriber_timer = self.create_timer(
            0.5, self._publish_snapshot_for_new_subscriber
        )

    def _layer_callback(self, message):
        changed_markers = []
        has_subscribers = self._publisher.get_subscription_count() > 0
        if message.clear:
            self._blocks.clear()
            self._block_ids.clear()
            self._next_marker_id = 0
            if has_subscribers:
                delete_all = Marker()
                delete_all.header = message.header
                delete_all.action = Marker.DELETEALL
                changed_markers.append(delete_all)

        if (
            len(message.block_indices)
            != len(message.blocks)
        ):
            if has_subscribers and changed_markers:
                self._publisher.publish(
                    MarkerArray(markers=changed_markers))
            self.get_logger().warning(
                'Ignoring malformed TSDF update: '
                'block index/data counts differ'
            )
            return

        for index, block in zip(message.block_indices, message.blocks):
            key = (index.x, index.y, index.z)
            if block.centers:
                # Keep the received block alive instead of copying every
                # voxel. ROS message objects remain valid while referenced.
                self._blocks[key] = block
                marker_id = self._block_ids.get(key)
                if marker_id is None:
                    marker_id = self._next_marker_id
                    self._next_marker_id += 1
                    self._block_ids[key] = marker_id
                if has_subscribers:
                    changed_markers.append(self._make_block_marker(
                        message.header,
                        message.voxel_size_m,
                        marker_id,
                        self._blocks[key],
                    ))
            else:
                self._blocks.pop(key, None)
                marker_id = self._block_ids.pop(key, None)
                if marker_id is not None and has_subscribers:
                    marker = Marker()
                    marker.header = message.header
                    marker.ns = 'tsdf_voxel_blocks'
                    marker.id = marker_id
                    marker.action = Marker.DELETE
                    changed_markers.append(marker)

        self._last_header = message.header
        self._voxel_size = message.voxel_size_m
        if has_subscribers and changed_markers:
            self._publisher.publish(MarkerArray(markers=changed_markers))

        if self._blocks and not self._reported_first_map:
            voxel_count = sum(
                len(block.centers) for block in self._blocks.values()
            )
            self.get_logger().info(
                f'Publishing {voxel_count} persistent TSDF cubes in '
                f'{len(self._blocks)} incremental blocks'
            )
            self._reported_first_map = True

    def _make_block_marker(self, header, voxel_size, marker_id, block):
        centers = block.centers
        colors = block.colors
        marker = Marker()
        marker.header = header
        marker.ns = 'tsdf_voxel_blocks'
        marker.id = marker_id
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        cube_size = voxel_size * self._voxel_scale
        marker.scale.x = cube_size
        marker.scale.y = cube_size
        marker.scale.z = cube_size
        marker.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=1.0)

        fallback_color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=1.0)
        marker.points.extend(
            Point(x=center.x, y=center.y, z=center.z)
            for center in centers
        )
        if len(colors) == len(centers):
            marker.colors.extend(colors)
        else:
            marker.colors.extend(fallback_color for _ in centers)
        return marker

    def _clear_rviz_markers(self, header=None):
        """Delete stale TSDF markers retained by the current RViz display."""
        marker = Marker()
        if header is not None:
            marker.header = header
        marker.ns = 'tsdf_voxel_blocks'
        marker.action = Marker.DELETEALL
        self._publisher.publish(MarkerArray(markers=[marker]))

    def _publish_snapshot_for_new_subscriber(self):
        subscription_count = self._publisher.get_subscription_count()
        if subscription_count == 0:
            if self._subscription is not None:
                self.destroy_subscription(self._subscription)
                self._subscription = None
                self._blocks.clear()
                self._block_ids.clear()
                self._next_marker_id = 0
                self._last_header = None
                self._voxel_size = None
                self._reported_first_map = False
            self._last_subscription_count = 0
            return

        if self._subscription is None:
            # RViz retains markers after the previous publisher disappears.
            # Clear that old ID set before rebuilding this publisher's
            # snapshot, otherwise blocks left by an earlier node instance can
            # overlap the current map even though they are not in nvblox.
            self._clear_rviz_markers()
            # nvblox sends the complete layer when a new TSDF subscriber
            # appears, so there is no need to deserialize the high-bandwidth
            # stream while RViz is not displaying these markers.
            self._subscription = self.create_subscription(
                VoxelBlockLayer,
                self._input_topic,
                self._layer_callback,
                self._input_qos,
            )
            self._last_subscription_count = subscription_count
            return

        if subscription_count <= self._last_subscription_count:
            self._last_subscription_count = subscription_count
            return

        # An additional RViz display may already contain markers retained from
        # a former publisher. Reset every subscriber, then publish one exact
        # snapshot from the block cache maintained above.
        self._clear_rviz_markers(self._last_header)
        if (
            not self._blocks
            or self._last_header is None
            or self._voxel_size is None
        ):
            self._last_subscription_count = subscription_count
            return

        markers = [
            self._make_block_marker(
                self._last_header,
                self._voxel_size,
                self._block_ids[key],
                block,
            )
            for key, block in self._blocks.items()
        ]
        # Bound individual publications so a newly connected RViz display does
        # not monopolize the executor with one multi-megabyte MarkerArray.
        chunk_size = 64
        for start in range(0, len(markers), chunk_size):
            self._publisher.publish(MarkerArray(
                markers=markers[start:start + chunk_size]
            ))
        self._last_subscription_count = subscription_count

    def clear_visualization(self):
        """Remove TSDF cubes before this publisher shuts down."""
        if self._publisher.get_subscription_count() > 0:
            self._clear_rviz_markers(self._last_header)


def main(args=None):
    """Run the TSDF cube marker node."""
    rclpy.init(args=args)
    node = TsdfCubeMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.clear_visualization()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
