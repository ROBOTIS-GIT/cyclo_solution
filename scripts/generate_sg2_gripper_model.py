#!/usr/bin/env python3
"""Build the demo's SG2 model with 45 mm fingers and explicit tool frames.

The bringup model/frames stay unchanged. Extra collision boxes are attached to
the existing distal finger links, so they follow the real gripper/mimic joints.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

EXTENSION_M = 0.045


def extended_model(text):
    root = ET.fromstring(text)
    for side in ('l', 'r'):
        if root.find(f"link[@name='gripper_{side}_tcp']") is not None:
            raise ValueError('Model already has extended TCP frames')
        parent = f'end_effector_{side}_link'
        if root.find(f"link[@name='{parent}']") is None:
            raise ValueError(f'Missing SG2 tool link: {parent}')
        ET.SubElement(root, 'link', name=f'gripper_{side}_tcp')
        joint = ET.SubElement(root, 'joint', name=f'gripper_{side}_tcp_joint', type='fixed')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=f'gripper_{side}_tcp')
        ET.SubElement(joint, 'origin', xyz=f'0 0 {-EXTENSION_M}', rpy='0 0 0')
        for finger, sign in (('r2', 1), ('l2', -1)):
            link = root.find(f"link[@name='gripper_{side}_rh_p12_rn_{finger}']")
            if link is None:
                raise ValueError(f'Missing SG2 distal finger: {side}/{finger}')
            # RH-P12-RN distal STL ends at local Z=40 mm. Approximate the
            # printed taper with layers; include 10 mm overlap at its root.
            edges = (0.0, 0.01, 0.02, 0.03, EXTENSION_M, EXTENSION_M + 0.01)
            for index, (low, high) in enumerate(zip(edges, edges[1:])):
                thickness = min(0.005 + 0.020 * high / 0.030, 0.030)
                width = 0.015 + 0.010 * min(high / 0.100, 1.0)
                xyz = f'0 {sign * thickness / 2:.9f} {0.040 + EXTENSION_M - (low + high) / 2:.9f}'
                size = f'{width:.9f} {thickness:.9f} {high - low:.9f}'
                for kind in ('collision', 'visual'):
                    part = ET.SubElement(link, kind, name=f'demo_extension_{index}')
                    ET.SubElement(part, 'origin', xyz=xyz, rpy='0 0 0')
                    ET.SubElement(ET.SubElement(part, 'geometry'), 'box', size=size)
                    if kind == 'visual':
                        material = ET.SubElement(part, 'material', name='demo_finger_extension')
                        ET.SubElement(material, 'color', rgba='0.3 0.3 0.3 1')
    ET.indent(root)
    return ET.tostring(root, encoding='unicode') + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(extended_model(Path(args.input).read_text()))
