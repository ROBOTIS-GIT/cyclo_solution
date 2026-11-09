"""ffw_pick settings from ROS 2 parameter files (read as plain YAML, no ROS needed)."""
from dataclasses import dataclass, fields

import yaml


def ros_parameters(path):
    """ros__parameters of the first entry of a ROS 2 parameter file."""
    with open(path) as f:
        return next(iter(yaml.safe_load(f).values()))['ros__parameters']


def from_file(kind, path):
    """The dataclass kind filled from path: every field must be there and nothing else."""
    values = ros_parameters(path)
    names = [f.name for f in fields(kind)]
    missing = [n for n in names if n not in values]
    unknown = [n for n in values if n not in names]
    if missing or unknown:
        raise ValueError(f'{path}: missing {missing}, unknown {unknown}')
    return kind(**{n: tuple(v) if isinstance(v, list) else v for n, v in values.items()})


def require(values, names, path):
    """Raises ValueError naming the keys of names that values (from path) lacks."""
    missing = [n for n in names if n not in values]
    if missing:
        raise ValueError(f'{path}: missing {missing}')


@dataclass(frozen=True)
class ArmConfig:
    """One arm: its names in the URDF, MoveIt and bringup, home pose and limits (arm_r.yaml)."""

    group: str
    base_frame: str
    end_effector_link: str
    wrist_link: str
    arm_joints: tuple
    gripper_joint: str
    finger_links: tuple
    touch_links: tuple
    gripper_action: str
    controller_topic: str
    home: tuple
    home_tolerance: float
    joint_margin: float
    max_acceleration: float
    max_fingertip_speed: float


@dataclass(frozen=True)
class PickConfig:
    """Pick behaviour (pick.yaml); the first nine are also command line options."""

    ns: str
    above: float
    lift: float
    speed: float
    accel: float
    turn_window: float
    budget: float
    hold: float
    home_lift: float
    plan_tries: int
    retry_wait: float
    empty_tries: int
    state_tolerance: float


COMMAND_LINE = ('ns', 'above', 'lift', 'speed', 'accel', 'turn_window', 'budget', 'hold',
                'home_lift')
