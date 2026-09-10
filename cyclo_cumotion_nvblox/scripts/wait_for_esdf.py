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

import time

import rclpy
from nvblox_msgs.srv import EsdfAndGradients
from rclpy.node import Node


class EsdfReadinessProbe(Node):
    def __init__(self):
        super().__init__('wait_for_esdf')
        self._client = self.create_client(
            EsdfAndGradients, '/nvblox_node/get_esdf_and_gradient'
        )

    def wait(self):
        self.get_logger().info('Waiting for nvblox to produce a usable ESDF...')
        while rclpy.ok():
            if not self._client.wait_for_service(timeout_sec=1.0):
                continue

            request = EsdfAndGradients.Request()
            request.update_esdf = True
            request.frame_id = 'base_link'
            future = self._client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if not future.done():
                future.cancel()
                continue

            try:
                response = future.result()
            except Exception as error:  # noqa: BLE001
                self.get_logger().warning(f'ESDF readiness request failed: {error}')
                continue

            has_timestamp = (
                response is not None
                and (
                    response.header.stamp.sec != 0
                    or response.header.stamp.nanosec != 0
                )
            )
            has_voxels = (
                response is not None
                and bool(response.esdf_and_gradients.data)
            )
            if response is not None and response.success and has_timestamp and has_voxels:
                self.get_logger().info('nvblox ESDF is ready; starting MoveIt/cuMotion')
                return True
            time.sleep(0.25)
        return False


def main(args=None):
    rclpy.init(args=args)
    node = EsdfReadinessProbe()
    success = False
    try:
        success = node.wait()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
