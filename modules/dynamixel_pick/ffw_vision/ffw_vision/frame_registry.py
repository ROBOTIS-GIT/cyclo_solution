from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameInfo:
    sec: int
    nanosec: int
    frame_id: str
    width: int
    height: int
    rx_ns: int

    @property
    def stamp_ns(self):
        return self.sec * 1_000_000_000 + self.nanosec


class FrameRegistry:
    """Headers of recently received frames, looked up by stamp."""

    def __init__(self, retention_ns):
        self.retention_ns = retention_ns
        self.frames = OrderedDict()

    def add(self, info):
        self.frames[info.stamp_ns] = info
        self.frames.move_to_end(info.stamp_ns)
        oldest_allowed = info.rx_ns - self.retention_ns
        while next(iter(self.frames.values())).rx_ns < oldest_allowed:
            self.frames.popitem(last=False)

    def get(self, stamp_ns):
        return self.frames.get(stamp_ns)

    def latest(self):
        return next(reversed(self.frames.values()), None)
