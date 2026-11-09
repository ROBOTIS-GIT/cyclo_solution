"""Arm chain from the URDF: forward kinematics, joint limits and the link sets the pick needs."""
import math
import xml.etree.ElementTree as ET

import numpy as np

FLIP = np.diag([-1.0, -1.0, 1.0])  # gripper is symmetric: same grasp turned 180 deg about z


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                              math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_matrix(axis, angle):
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * k @ k


def transform(rotation, translation):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rotation, translation
    return out


def quat_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def floats(element, attribute, default):
    text = default if element is None else element.get(attribute, default)
    return np.array([float(v) for v in text.split()])


class Chain:
    """Joints from root to tip link, read from URDF text."""

    def __init__(self, urdf, root, tip):
        joints = {}
        for joint in ET.fromstring(urdf).findall('joint'):  # skip ros2_control joints
            limit = joint.find('limit')
            joints[joint.find('child').get('link')] = {
                'name': joint.get('name'),
                'type': joint.get('type'),
                'parent': joint.find('parent').get('link'),
                'xyz': floats(joint.find('origin'), 'xyz', '0 0 0'),
                'rpy': floats(joint.find('origin'), 'rpy', '0 0 0'),
                'axis': floats(joint.find('axis'), 'xyz', '1 0 0'),
                'limit': (-math.inf, math.inf) if limit is None else
                (float(limit.get('lower', -math.inf)), float(limit.get('upper', math.inf))),
                'velocity': math.inf if limit is None else float(limit.get('velocity', math.inf)),
            }
        self.joints = []
        link = tip
        while link != root:
            if link not in joints:
                raise ValueError(f'no path from {root} to {tip} (stopped at {link})')
            self.joints.insert(0, joints[link])
            link = joints[link]['parent']
        for j in self.joints:  # origin transforms never change
            j['static'] = transform(rpy_matrix(*j['rpy']), j['xyz'])

    def moving(self):
        return [j['name'] for j in self.joints if j['type'] != 'fixed']

    def limits(self, name):
        return next(j['limit'] for j in self.joints if j['name'] == name)

    def velocity(self, name):
        """URDF velocity limit of a joint (inf when not given)."""
        return next(j['velocity'] for j in self.joints if j['name'] == name)

    def fk(self, values):
        """Tip pose (4x4) in the root frame; joints missing from values are 0."""
        out = np.eye(4)
        for j in self.joints:
            out = out @ j['static']
            q = values.get(j['name'], 0.0)
            if j['type'] in ('revolute', 'continuous'):
                out = out @ transform(axis_matrix(j['axis'], q), np.zeros(3))
            elif j['type'] == 'prismatic':
                out = out @ transform(np.eye(3), j['axis'] * q)
        return out


def urdf_links(urdf, below=None):
    """Link names of the URDF text, or only those at and below link below."""
    root = ET.fromstring(urdf)
    if below is None:
        return {link.get('name') for link in root.findall('link')}
    children = {}
    for joint in root.findall('joint'):
        children.setdefault(joint.find('parent').get('link'), []).append(
            joint.find('child').get('link'))
    out, todo = set(), [below]
    while todo:
        out.add(todo[-1])
        todo += children.get(todo.pop(), [])
    return out


def tip(chain, values, tip_below):
    """Fingertip position: tip_below m down the end effector z axis."""
    pose = chain.fk(values)
    return pose[:3, 3] - tip_below * pose[:3, 2]


def pose_parts(msg):
    """(position, rotation matrix) of a PoseStamped."""
    p, o = msg.pose.position, msg.pose.orientation
    return np.array([p.x, p.y, p.z]), quat_matrix(o.x, o.y, o.z, o.w)
