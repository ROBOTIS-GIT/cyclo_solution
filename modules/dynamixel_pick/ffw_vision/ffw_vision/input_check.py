from bisect import bisect_left
from collections import deque

NS_PER_SEC = 1_000_000_000
NS_PER_MS = 1_000_000
DEPTH_ENCODINGS = ('32FC1', '16UC1')


def stamp_to_ns(stamp):
    return stamp.sec * NS_PER_SEC + stamp.nanosec


def nearest_gap_ns(sorted_stamps, stamp):
    index = bisect_left(sorted_stamps, stamp)
    neighbors = sorted_stamps[max(index - 1, 0):index + 1]
    return min((abs(stamp - other) for other in neighbors), default=None)


class CompanionStream:
    """Recent stamps of a stream that must share stamps with the image."""

    def __init__(self, history):
        self.stamps = deque(maxlen=history)
        self.count = 0

    def add(self, stamp_ns):
        self.stamps.append(stamp_ns)
        self.count += 1

    def match(self, image_stamps):
        sorted_stamps = sorted(self.stamps)
        known = set(sorted_stamps)
        gaps = [nearest_gap_ns(sorted_stamps, s) for s in image_stamps if s not in known]
        paired = len(image_stamps) - len(gaps)
        return paired, len(gaps), max((g for g in gaps if g is not None), default=0)


class InputMonitor:

    def __init__(self, config, timeout_ns, settle_ns=100 * NS_PER_MS, history=300):
        self.config = config
        self.timeout_ns = timeout_ns
        self.settle_ns = settle_ns  # wait before judging an image's pairs
        self.info = CompanionStream(history)
        self.depth = CompanionStream(history) if config.depth_topic else None
        self.info_meta = None  # (frame_id, width, height)
        self.last_image_rx_ns = None
        self.last_image_size = None
        self.waiting = []  # (stamp_ns, rx_ns)
        self.window_start_ns = None
        self._reset_window()

    def _reset_window(self):
        self.window_stamps = []
        self.latencies_ns = []
        self.encodings = set()
        self.sizes = set()
        self.errors = dict.fromkeys(
            ('frame_id', 'size', 'info_frame_id', 'depth_frame_id', 'depth_size',
             'depth_encoding'), 0)
        self.info.count = 0
        if self.depth:
            self.depth.count = 0

    def on_camera_info(self, stamp_ns, frame_id, width, height):
        self.info.add(stamp_ns)
        self.info_meta = (frame_id, width, height)
        if frame_id != self.config.frame_id:
            self.errors['info_frame_id'] += 1

    def on_depth(self, stamp_ns, frame_id, width, height, encoding):
        self.depth.add(stamp_ns)
        if frame_id != self.config.frame_id:
            self.errors['depth_frame_id'] += 1
        if self.last_image_size is not None and (width, height) != self.last_image_size:
            self.errors['depth_size'] += 1
        if encoding not in DEPTH_ENCODINGS:
            self.errors['depth_encoding'] += 1

    def on_image(self, stamp_ns, frame_id, width, height, encoding, rx_ns):
        if self.window_start_ns is None:
            self.window_start_ns = rx_ns
        self.last_image_rx_ns = rx_ns
        self.last_image_size = (width, height)
        self.window_stamps.append(stamp_ns)
        self.waiting.append((stamp_ns, rx_ns))
        self.latencies_ns.append(rx_ns - stamp_ns)
        self.encodings.add(encoding)
        self.sizes.add((width, height))
        if frame_id != self.config.frame_id:
            self.errors['frame_id'] += 1
        if self.info_meta is not None and (width, height) != self.info_meta[1:]:
            self.errors['size'] += 1

    def report(self, now_ns):
        if self.window_start_ns is None:
            self.window_start_ns = now_ns
        window_sec = max(now_ns - self.window_start_ns, 1) / NS_PER_SEC

        ready = [s for s, rx in self.waiting if now_ns - rx >= self.settle_ns]
        self.waiting = [(s, rx) for s, rx in self.waiting if now_ns - rx < self.settle_ns]
        info = self.info.match(ready)
        depth = self.depth.match(ready) if self.depth else None
        stamps = sorted(self.window_stamps)
        intervals = [b - a for a, b in zip(stamps, stamps[1:])]

        if self.last_image_rx_ns is None or now_ns - self.last_image_rx_ns > self.timeout_ns:
            state = 'NO_INPUT'
        elif self.info.count == 0:
            state = 'NO_CAMERA_INFO'
        elif self.depth and self.depth.count == 0:
            state = 'NO_DEPTH'
        elif any(self.errors.values()) or info[1] or (depth and depth[1]):
            state = 'MISMATCH'
        else:
            state = 'OK'

        result = {
            'state': state,
            'images': len(self.window_stamps),
            'rate_hz': len(self.window_stamps) / window_sec,
            'max_interval_ms': max(intervals, default=0) / NS_PER_MS,
            'latency_avg_ms': (sum(self.latencies_ns) / len(self.latencies_ns) / NS_PER_MS
                               if self.latencies_ns else 0.0),
            'latency_max_ms': max(self.latencies_ns, default=0) / NS_PER_MS,
            'sizes': sorted(self.sizes),
            'encodings': sorted(self.encodings),
            'info_meta': self.info_meta,
            'info': (self.info.count, *info),
            'depth': (self.depth.count, *depth) if self.depth else None,
            'waiting': len(self.waiting),
            'errors': {k: v for k, v in self.errors.items() if v},
        }
        self.window_start_ns = now_ns
        self._reset_window()
        return result


def format_pairs(name, pairs):
    count, paired, unpaired, gap_ns = pairs
    return (f'{name} n={count} paired={paired} unpaired={unpaired} '
            f'gap_max={gap_ns / NS_PER_MS:.1f}ms')


def format_report(result):
    sizes = ','.join(f'{w}x{h}' for w, h in result['sizes']) or '-'
    meta = result['info_meta']
    parts = [
        f"state={result['state']} images={result['images']} rate={result['rate_hz']:.1f}Hz "
        f"interval_max={result['max_interval_ms']:.1f}ms "
        f"latency_avg={result['latency_avg_ms']:.1f}ms "
        f"latency_max={result['latency_max_ms']:.1f}ms "
        f"size={sizes} enc={','.join(result['encodings']) or '-'}",
        format_pairs('info', result['info'])
        + (f' frame={meta[0]} {meta[1]}x{meta[2]}' if meta else ''),
    ]
    if result['depth']:
        parts.append(format_pairs('depth', result['depth']))
    errors = ','.join(f'{k}:{v}' for k, v in result['errors'].items()) or 'none'
    parts.append(f"waiting={result['waiting']} errors={errors}")
    return ' | '.join(parts)
