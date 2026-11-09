#!/usr/bin/env python3
"""Dump evenly spaced frames from bags: color PNG, raw depth PNG, CameraInfo + TF at the image stamp as JSON."""
import argparse
import json
import os

import cv2
from ffw_vision.depth_stats import color_from_msg
import numpy as np
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
import rosbag2_py
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_msgs.msg import TFMessage
import tf2_ros

PRESETS = {
    'right_wrist': ('/camera_right/camera_right/color/image_rect_raw',
                    '/camera_right/camera_right/color/camera_info',
                    '/camera_right/camera_right/aligned_depth_to_color/image_raw'),
    'left_wrist': ('/camera_left/camera_left/color/image_rect_raw',
                   '/camera_left/camera_left/color/camera_info',
                   '/camera_left/camera_left/aligned_depth_to_color/image_raw'),
    'head': ('/zed/zed_node/left/image_rect_color',
             '/zed/zed_node/left/camera_info',
             '/zed/zed_node/depth/depth_registered'),
}


def stamp_ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


def read_bag(bag_dir, topics):
    """All messages of the given topics, deserialized, as {topic: [msg, ...]}."""
    types = {topics[0]: Image, topics[1]: CameraInfo, topics[2]: Image,
             '/tf': TFMessage, '/tf_static': TFMessage, '/joint_states': JointState}
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(types)))
    out = {t: [] for t in types}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        out[topic].append(deserialize_message(data, types[topic]))
    return out


def raw_depth(msg):
    """16UC1 as uint16 rows, or 32FC1 meters as float32."""
    dtype = '<u2' if msg.encoding == '16UC1' else '<f4'
    row = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, -1)
    return row[:, :msg.width].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bags', nargs='+')
    parser.add_argument('--camera', choices=list(PRESETS), default='right_wrist')
    parser.add_argument('--base', default='base_link')
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--out', default='frames_out')
    args = parser.parse_args()
    topics = PRESETS[args.camera]
    os.makedirs(args.out, exist_ok=True)

    for bag in args.bags:
        if not os.path.isdir(bag):  # e.g. a scene memo matched by the glob
            continue
        name = os.path.basename(os.path.normpath(bag))
        msgs = read_bag(bag, topics)
        tf = tf2_ros.Buffer(cache_time=Duration(seconds=600))
        for m in msgs['/tf_static']:
            for t in m.transforms:
                tf.set_transform_static(t, 'bag')
        for m in msgs['/tf']:
            for t in m.transforms:
                tf.set_transform(t, 'bag')
        depths = {stamp_ns(m.header.stamp): m for m in msgs[topics[2]]}
        info = msgs[topics[1]][0]
        # paired with depth and inside the recorded TF range
        colors = [m for m in msgs[topics[0]] if stamp_ns(m.header.stamp) in depths
                  and tf.can_transform(args.base, m.header.frame_id,
                                       Time.from_msg(m.header.stamp))]
        joints = msgs['/joint_states']
        picks = np.linspace(0, len(colors) - 1, min(args.frames, len(colors))).round().astype(int)
        for i, k in enumerate(picks):
            color = colors[k]
            ns = stamp_ns(color.header.stamp)
            try:
                t = tf.lookup_transform(args.base, color.header.frame_id,
                                        Time.from_msg(color.header.stamp)).transform
            except tf2_ros.TransformException as e:  # stamp outside the TF cache
                print(f'{name}[{i}] TF failed: {e}')
                continue
            depth = depths[ns]
            stem = os.path.join(args.out, f'{name}_{i}')
            cv2.imwrite(stem + '_color.png', color_from_msg(color))
            d = raw_depth(depth)
            if d.dtype == np.uint16:
                cv2.imwrite(stem + '_depth.png', d)
            else:
                np.save(stem + '_depth.npy', d)
            js = min(joints, key=lambda m: abs(stamp_ns(m.header.stamp) - ns)) if joints else None
            meta = {
                'bag': name, 'stamp_ns': ns, 'frame_id': color.header.frame_id,
                'width': info.width, 'height': info.height,
                'k': list(info.k), 'd': list(info.d), 'distortion_model': info.distortion_model,
                'depth_encoding': depth.encoding, 'depth_frame_id': depth.header.frame_id,
                'tf_base': args.base,
                'tf_translation': [t.translation.x, t.translation.y, t.translation.z],
                'tf_rotation_xyzw': [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w],
                'joints': dict(zip(js.name, js.position)) if js else {},
            }
            with open(stem + '.json', 'w') as f:
                json.dump(meta, f, indent=1)
        print(f'{name}: {len(picks)} frames of {len(colors)} paired')
    print(f'wrote to {args.out}')


if __name__ == '__main__':
    main()
