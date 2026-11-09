#!/usr/bin/env python3
"""One-time migration: keep taught handover poses after extending the TCP.

Preserves all sequence timing, gripper commands and tuning. Only solves arm
joint targets against the same Cartesian poses with the new tool length.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from generate_sg2_targets import Arm, RosDumper


def retarget(base_model, new_model, config_path, report_path):
    old = ET.parse(base_model).getroot()
    new = ET.parse(new_model).getroot()
    config = yaml.safe_load(Path(config_path).read_text())
    params = config['handover_demo']['ros__parameters']
    if params.get('arms', {}).get('left', {}).get('tcp') == 'gripper_l_tcp':
        raise ValueError('Handover targets are already migrated; do not apply twice')
    arms = {}
    for side, suffix in (('left', 'l'), ('right', 'r')):
        arms[side] = (Arm(old, side), Arm(new, side, f'gripper_{suffix}_tcp_joint'))
    worst = 0.0
    for sequence in params['motion_sequences'].values():
        for step in sequence.values():
            if not isinstance(step, dict):
                continue
            for side, (old_arm, new_arm) in arms.items():
                key = side + '_target'
                if key not in step:
                    continue
                q = np.array(step[key])
                target = old_arm.fk(q)
                step[key] = new_arm.solve(target[:3, 3], target[:3, :3], q)
                error = np.linalg.norm(new_arm.fk(step[key])[:3, 3] - target[:3, 3])
                worst = max(worst, error)
    for side, suffix in (('left', 'l'), ('right', 'r')):
        params.setdefault('arms', {}).setdefault(side, {})['tcp'] = f'gripper_{suffix}_tcp'
    Path(config_path).write_text(
        '# SG2 handover: original Cartesian targets preserved for 45 mm extended TCPs.\n'
        + yaml.dump(config, Dumper=RosDumper, sort_keys=False))
    report = json.loads(Path(report_path).read_text())
    for side, poses in report['poses'].items():
        old_arm, new_arm = arms[side]
        for pose in poses.values():
            q = np.array(pose['joints'])
            target = old_arm.fk(q)
            pose['joints'] = new_arm.solve(target[:3, 3], target[:3, :3], q)
            transform = new_arm.fk(pose['joints'])
            pose.update(position=transform[:3, 3].tolist(),
                        quaternion_xyzw=Rotation.from_matrix(transform[:3, :3]).as_quat().tolist(),
                        tcp_forward=(-transform[:3, 2]).tolist())
    report['urdf_sha256'] = hashlib.sha256(Path(new_model).read_bytes()).hexdigest()
    report['tcp_frames'] = {'left': 'gripper_l_tcp', 'right': 'gripper_r_tcp'}
    report['finger_extension_m'] = 0.045
    Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
    print(f'Handover targets migrated; largest TCP position error={worst:.9f} m')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--new-model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    retarget(args.base_model, args.new_model, args.config, args.report)
