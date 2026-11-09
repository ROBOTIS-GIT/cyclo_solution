import numpy as np

CHANNELS = {'bgra8': 4, 'rgba8': 4, 'bgr8': 3, 'rgb8': 3, 'mono8': 1}


def _rows(msg, dtype, channels):
    itemsize = np.dtype(dtype).itemsize
    data = np.frombuffer(msg.data, dtype=dtype)
    data = data.reshape(msg.height, msg.step // itemsize)
    return data[:, :msg.width * channels].reshape(msg.height, msg.width, channels)


def color_from_msg(msg):
    """Return a BGR uint8 array."""
    if msg.encoding not in CHANNELS:
        raise ValueError(f'unsupported image encoding: {msg.encoding}')
    image = _rows(msg, np.uint8, CHANNELS[msg.encoding])
    if msg.encoding == 'mono8':
        return np.repeat(image, 3, axis=2)
    if msg.encoding.startswith('rgb'):
        return image[:, :, 2::-1].copy()
    return image[:, :, :3].copy()


def depth_from_msg(msg, depth_scale=0.001):
    """Return depth in meters as float32; 16UC1 is multiplied by depth_scale, 0 becomes NaN."""
    endian = '>' if msg.is_bigendian else '<'
    if msg.encoding == '32FC1':
        return _rows(msg, endian + 'f4', 1)[:, :, 0].astype(np.float32)
    if msg.encoding == '16UC1':
        raw = _rows(msg, endian + 'u2', 1)[:, :, 0]
        depth = raw.astype(np.float32) * np.float32(depth_scale)
        depth[raw == 0] = np.nan
        return depth
    raise ValueError(f'unsupported depth encoding: {msg.encoding}')


def clip_roi(roi, width, height):
    """roi = (u_min, v_min, u_max, v_max), max exclusive; all zeros = full image."""
    if not any(roi):
        return 0, 0, width, height
    u0, v0, u1, v1 = roi
    u0, u1 = max(0, min(u0, width)), max(0, min(u1, width))
    v0, v1 = max(0, min(v0, height)), max(0, min(v1, height))
    if u0 >= u1 or v0 >= v1:
        raise ValueError(f'empty roi after clipping: {roi}')
    return u0, v0, u1, v1


def valid_mask(depth, min_depth, max_depth):
    with np.errstate(invalid='ignore'):
        return np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)


def roi_summary(depth_frames, roi, min_depth, max_depth):
    u0, v0, u1, v1 = roi
    crops = np.stack([frame[v0:v1, u0:u1] for frame in depth_frames])
    masks = valid_mask(crops, min_depth, max_depth)
    ratios = masks.reshape(len(crops), -1).mean(axis=1)
    values = crops[masks]
    medians = [np.median(c[m]) for c, m in zip(crops, masks) if m.any()]
    nan = float('nan')
    return {
        'frames': len(crops),
        'pixels': int(crops[0].size),
        'valid_ratio': float(masks.mean()),
        'valid_ratio_min': float(ratios.min()),
        'valid_ratio_max': float(ratios.max()),
        'median_m': float(np.median(values)) if values.size else nan,
        'p10_m': float(np.percentile(values, 10)) if values.size else nan,
        'p90_m': float(np.percentile(values, 90)) if values.size else nan,
        'median_std_m': float(np.std(medians)) if medians else nan,  # frame-to-frame
    }
