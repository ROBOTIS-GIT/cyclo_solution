"""Right gripper joint <-> fingertip opening from the measured table, and the closing settings."""
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass(frozen=True)
class Gripper:
    table: list  # [(joint rad, opening m)] sorted by joint; opening shrinks as the joint closes
    hold_below: float  # rad: stopped below this after closing = holding
    close_command: float  # rad
    move_time: float  # s to open or close


def load_gripper(path):
    """Gripper from a table file (table rows, hold_below, close_command, move_time)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    rows = sorted((float(j), float(o)) for j, o in data['table'])
    if len(rows) < 2:
        raise ValueError(f'{path}: need at least 2 rows')
    missing = [k for k in ('hold_below', 'close_command', 'move_time') if k not in data]
    if missing:
        raise ValueError(f'{path}: missing {missing}')
    return Gripper(rows, float(data['hold_below']), float(data['close_command']),
                   float(data['move_time']))


def joint_for(table, opening):
    """Gripper joint for an opening (m), linear between measured rows, clipped to the table."""
    joints, openings = zip(*table)
    return float(np.interp(opening, openings[::-1], joints[::-1]))
