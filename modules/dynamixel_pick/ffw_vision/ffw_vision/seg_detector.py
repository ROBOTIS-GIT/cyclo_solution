from dataclasses import dataclass
import math
import os

import cv2
from ffw_vision.blob_detector import normalize_angle
import numpy as np


@dataclass(frozen=True)
class SegParams:
    conf: float = 0.5  # detections below this score are dropped
    dup_overlap: float = 0.5  # masks sharing this share of the smaller one are one object
    basket_h_min: int = 95  # basket: largest blue component, HSV
    basket_h_max: int = 125
    basket_s_min: int = 120
    basket_v_min: int = 80
    inner_erode_px: int = 7  # basket rim left out of the object area
    ring_gap_px: int = 9  # table ring around the basket for the support plane
    ring_outer_px: int = 41
    plane_samples: int = 3000
    plane_iters: int = 200
    plane_tol: float = 0.004  # m
    min_plane_points: int = 300
    mask_erode_px: int = 5  # model masks bleed about 1 px past the object
    min_mask_px: int = 20
    top_share: float = 20.0  # top level = median of this percent of highest points
    top_tolerance: float = 0.006  # m below the top level still counted as top face
    edge_percentile: float = 3.0  # top face extent ignores this percent at each end (depth noise)
    min_top_px: int = 10
    standing_min_height: float = 0.042  # m; measured lying 28-39 mm, standing 44-53 mm
    max_single_length: float = 0.056  # m; a longer top face is not one object
    max_single_width: float = 0.040
    square_ratio: float = 1.25  # length / width below this = near square
    standing_square: bool = True  # standing top face is near square (28.5 x 34 mm): period pi/2
    angle_method: str = 'edges'  # 'edges' = color edge directions, 'min_area_rect' = top points
    edge_min_px: int = 30  # fewer strong edge pixels -> min_area_rect
    edge_min_share: float = 0.15  # weight within 5 deg of the peak (even 0.12); less -> rect
    edge_skip_blue: bool = True  # edges only from non-blue mask pixels +1 px (bled onto walls)
    face_long: float = 0.0  # m, known box sides for the drawn top face (0: draw the measured size)
    face_mid: float = 0.0
    face_short: float = 0.0
    min_depth: float = 0.07
    max_depth: float = 1.0


@dataclass(frozen=True)
class SegObject:
    u: float  # top face center in source pixels
    v: float
    length: float  # top face long side, projected, px
    width: float
    angle: float
    angle_period: float
    pose_label: str
    ambiguous: bool
    score: float
    plane_height: float  # top face above the support plane, m
    length_m: float
    width_m: float
    outline: np.ndarray  # mask contour (OBB/hybrid: the box), for drawing
    angle_source: str = ''  # 'edges', 'min_area_rect', 'obb', 'obb+edges', 'hybrid:...'
    mask_outline: np.ndarray = None  # hybrid: the seg mask contour, for drawing
    face_corners: np.ndarray = None  # drawn top face: 4 image corners (known size, 3D projected)


def basket_masks(color_bgr, params):
    """(basket area filled from its outer contour, blue pixels); basket is None if no blue."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = ((h >= params.basket_h_min) & (h <= params.basket_h_max)
            & (s >= params.basket_s_min) & (v >= params.basket_v_min)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
    if count < 2:
        return None, blue.astype(bool)
    largest = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    basket = np.zeros_like(blue)
    cv2.drawContours(basket, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return basket, blue.astype(bool)


def pixel_rays(width, height, k, d):
    """Undistorted normalized (x, y) of every pixel, shape (height, width, 2)."""
    vv, uu = np.mgrid[0:height, 0:width].astype(np.float32)
    pixels = np.column_stack((uu.ravel(), vv.ravel())).reshape(-1, 1, 2)
    camera = np.array(k, np.float64).reshape(3, 3)
    rays = cv2.undistortPoints(pixels, camera, np.array(d, np.float64))
    return rays.reshape(height, width, 2)


def fit_plane(points, params, rng):
    """RANSAC plane (n, d) with n . p + d = height; unit n toward the camera origin."""
    if len(points) > params.plane_samples:
        points = points[rng.choice(len(points), params.plane_samples, replace=False)]
    best = None
    for _ in range(params.plane_iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inliers = np.abs((points - a) @ n) < params.plane_tol
        if best is None or inliers.sum() > best.sum():
            best = inliers
    q = points[best]
    center = q.mean(axis=0)
    n = np.linalg.svd(q - center)[2][2]
    d = -float(n @ center)
    return (n, d) if d > 0 else (-n, -d)


def plane_axes(n):
    """Two unit vectors spanning the plane."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(n, seed)
    a /= np.linalg.norm(a)
    return a, np.cross(n, a)


def known_face(top_level, params):
    """(long, short) of the known top face for the top height: lowest side down -> long x mid,
    mid side down -> long x short, standing -> mid x short; None without face_long."""
    if params.face_long <= 0:
        return None
    if top_level >= params.standing_min_height:
        return params.face_mid, params.face_short
    if top_level < (params.face_short + params.face_mid) / 2:
        return params.face_long, params.face_mid
    return params.face_long, params.face_short


def face_corners(center, axis, n, top_level, length_m, width_m, k, d, params):
    """Image corners of the top face for drawing: the known face for the top height, else the
    given (measured) size."""
    length_m, width_m = known_face(top_level, params) or (length_m, width_m)
    across = np.cross(n, axis)
    pts = [center + sa * length_m / 2 * axis + sb * width_m / 2 * across
           for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    return np.round(project(pts, k, d)).astype(np.int32).reshape(-1, 1, 2)


def project(points, k, d):
    camera = np.array(k, np.float64).reshape(3, 3)
    pixels, _ = cv2.projectPoints(np.asarray(points, np.float64).reshape(-1, 1, 3),
                                  np.zeros(3), np.zeros(3), camera, np.array(d, np.float64))
    return pixels.reshape(-1, 2)


def merge_duplicates(segments, params):
    """Overlapping masks become their intersection with the higher score (never larger)."""
    kept = []
    for mask, score in sorted(segments, key=lambda seg: -seg[1]):
        for i, (other, other_score) in enumerate(kept):
            inter = np.logical_and(mask, other).sum()
            if inter >= params.dup_overlap * min(mask.sum(), other.sum()):
                kept[i] = (np.logical_and(mask, other), other_score)
                break
        else:
            kept.append((mask, score))
    return kept


def edge_angle(gray, region, rays, n, plane_d, level, params, debug=None):
    """Box direction (rad, mod pi/2, plane_axes frame) from color edges in region, or None.

    Every horizontal box edge (top rim, side bottoms, straight markings) is parallel to a box
    axis; edge directions are mapped onto the plane at level and the strongest one wins.
    debug: dict filled with the votes (tools/angle_debug.py).
    """
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        return None
    y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, region.shape[0])
    x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, region.shape[1])
    g = gray[y0:y1, x0:x1]
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    inside = region[y0:y1, x0:x1]
    sel = inside & (mag > np.percentile(mag[inside], 50))
    if sel.sum() < params.edge_min_px:
        return None
    r = np.dstack((rays[y0:y1, x0:x1], np.ones((y1 - y0, x1 - x0))))
    grid = r * ((level - plane_d) / (r @ n))[:, :, None]  # pixel -> point on the raised plane
    du, dv = np.gradient(grid, axis=1)[sel], np.gradient(grid, axis=0)[sel]
    m = mag[sel]
    edge = (-gy[sel] / m)[:, None] * du + (gx[sel] / m)[:, None] * dv  # along the edge
    a, b = plane_axes(n)
    deg = np.degrees(np.arctan2(edge @ b, edge @ a)) % 90.0
    hist, _ = np.histogram(deg, bins=90, range=(0, 90), weights=m)
    if debug is not None:
        sy, sx = np.nonzero(sel)
        debug.update(edge_v=sy + y0, edge_u=sx + x0, edge_deg=deg, edge_w=m,
                     edge_img=np.column_stack((-gy[sel], gx[sel])) / m[:, None])
    kernel = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
    smooth = np.convolve(np.concatenate((hist[-9:], hist, hist[:9])), kernel, 'same')[9:-9]
    peak = int(np.argmax(smooth))
    near = hist[[(peak + i) % 90 for i in range(-5, 6)]].sum()
    if debug is not None:
        debug.update(peak_deg=peak + 0.5, share=float(near / hist.sum()))
    if near < params.edge_min_share * hist.sum():
        return None
    return math.radians(peak + 0.5)


def measure(points, heights, mask, n, k, d, params, edges=None, debug=None):
    """Top face of one mask: (top level, center, long axis, length, width, angle source) or None.

    edges = (blurred gray, edge region, rays, plane d) for angle_method 'edges'.
    """
    h = heights[mask]
    top_level = float(np.median(h[h >= np.percentile(h, 100 - params.top_share)]))
    top = mask & (heights > top_level - params.top_tolerance)
    if top.sum() < params.min_top_px:
        return None
    q = points[top]
    origin = q.mean(axis=0)
    a, b = plane_axes(n)
    flat = np.column_stack(((q - origin) @ a, (q - origin) @ b)).astype(np.float32)
    rad = None
    if edges is not None and params.angle_method == 'edges':
        gray, region, rays, plane_d = edges
        rad = edge_angle(gray, region, rays, n, plane_d, top_level, params, debug)
    source = 'edges' if rad is not None else 'min_area_rect'
    if debug is not None:
        debug.update(top=top, top_level=top_level,
                     rect_deg=cv2.minAreaRect(flat)[2] % 90.0)
    if rad is None:
        rad = math.radians(cv2.minAreaRect(flat)[2])  # orientation only; extents from percentiles
    axis_a = np.array([math.cos(rad), math.sin(rad)])
    along = np.column_stack((flat @ axis_a, flat @ np.array([-axis_a[1], axis_a[0]])))
    edge = params.edge_percentile
    low, high = np.percentile(along, [edge, 100 - edge], axis=0)
    # trimming shortens a uniform spread by 2 * edge %; undo it
    (ca, cb), (sa, sb) = (low + high) / 2, (high - low) / (1 - edge / 50)
    side_a = math.cos(rad) * a + math.sin(rad) * b
    side_b = np.cross(n, side_a)
    center = origin + ca * side_a + cb * side_b
    if sa >= sb:
        return top_level, center, side_a, sa, sb, source
    return top_level, center, side_b, sb, sa, source


@dataclass
class SupportScene:
    """Per-image geometry shared by the detectors (camera optical frame)."""

    blue: np.ndarray  # basket-blue pixels
    inner: np.ndarray  # basket area without its rim
    valid: np.ndarray  # usable depth
    points: np.ndarray  # (h, w, 3) camera-frame points
    heights: np.ndarray  # above the support plane, m
    n: np.ndarray  # support plane normal, toward the camera
    plane_d: float
    gray: np.ndarray  # blurred gray, float32, for color edges
    rays: np.ndarray


def support_scene(color_bgr, depth, k, d, params, rays=None):
    """Basket, support plane (table ring around the basket) and heights; None without them."""
    height_px, width_px = color_bgr.shape[:2]
    basket, blue = basket_masks(color_bgr, params)
    if basket is None:
        return None
    if rays is None:
        rays = pixel_rays(width_px, height_px, k, d)
    with np.errstate(invalid='ignore'):
        valid = np.isfinite(depth) & (depth >= params.min_depth) & (depth <= params.max_depth)
    z = np.nan_to_num(depth)
    points = np.dstack((rays[:, :, 0] * z, rays[:, :, 1] * z, z))

    ring = ((cv2.dilate(basket, np.ones((params.ring_outer_px,) * 2, np.uint8)) > 0)
            & (cv2.dilate(basket, np.ones((params.ring_gap_px,) * 2, np.uint8)) == 0) & valid)
    if ring.sum() < params.min_plane_points:
        return None
    n, plane_d = fit_plane(points[ring], params, np.random.default_rng(0))
    inner = cv2.erode(basket, np.ones((params.inner_erode_px,) * 2, np.uint8)).astype(bool)
    gray = cv2.GaussianBlur(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    return SupportScene(blue, inner, valid, points, points @ n + plane_d, n, plane_d,
                 gray.astype(np.float32), rays)


def detect_seg(color_bgr, depth, k, d, segments, params, rays=None, debug=None, scene=None):
    """Objects from model masks + aligned depth (m, NaN invalid). Returns (objects, plane or None).

    segments: [(bool mask, score)] in source pixels. Geometry is in the camera optical frame.
    debug: list that gets one dict per returned object (tools/angle_debug.py, hybrid_detector).
    scene: support_scene of this image when the caller has it already.
    """
    scene = scene or support_scene(color_bgr, depth, k, d, params, rays)
    if scene is None:
        return [], None
    blue, inner, valid, points = scene.blue, scene.inner, scene.valid, scene.points
    heights, n, plane_d, gray, rays = scene.heights, scene.n, scene.plane_d, scene.gray, scene.rays
    erode = np.ones((params.mask_erode_px,) * 2, np.uint8)

    objects = []
    segments = [(mask, score) for mask, score in segments if score >= params.conf]
    for mask, score in merge_duplicates(segments, params):
        # eroded and without blue: oversized masks leak onto the basket wall
        core = cv2.erode(mask.astype(np.uint8), erode).astype(bool) & inner & valid & ~blue
        if core.sum() < params.min_mask_px:
            continue
        # edges: the mask (+1 px, top rim and side faces) inside the basket; without the blue
        # part the wall and shadow lines under a bled mask do not vote
        voters = mask & ~blue if params.edge_skip_blue else mask
        region = cv2.dilate(voters.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        region &= inner
        info = None if debug is None else {'mask': mask, 'region': region}
        top = measure(points, heights, core, n, k, d, params, (gray, region, rays, plane_d), info)
        if top is None:
            continue
        top_level, center, axis, length_m, width_m, angle_source = top
        if debug is not None:
            info['axis'] = axis
            debug.append(info)
        half_long, half_short = axis * length_m / 2, np.cross(n, axis) * width_m / 2
        (u, v), tip, tail, side, other = project(
            [center, center + half_long, center - half_long, center + half_short,
             center - half_short], k, d)
        # span between both ends: the near end looks longer in a tilted view
        length_px = float(np.hypot(*(tip - tail)))
        width_px = float(np.hypot(*(side - other)))
        angle = normalize_angle(math.atan2(tip[1] - v, tip[0] - u))
        ambiguous = length_m > params.max_single_length or width_m > params.max_single_width
        pose = 'standing' if top_level >= params.standing_min_height else 'lying'
        square = (length_m / max(width_m, 1e-6) < params.square_ratio
                  or (pose == 'standing' and params.standing_square))
        period = math.pi / 2 if square else math.pi
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        objects.append(SegObject(
            float(u), float(v), length_px, width_px, angle, 0.0 if ambiguous else period,
            'unknown' if ambiguous else pose, ambiguous, float(score), top_level,
            float(length_m), float(width_m), max(contours, key=cv2.contourArea), angle_source,
            face_corners=face_corners(center, axis, n, top_level, length_m, width_m, k, d,
                                      params)))
    return objects, (n, plane_d)


def resolve_model(model_path, model_file, models_dir):
    """YOLO weights to load: model_path (launch override) wins, else model_file, a name in
    models_dir or an absolute path."""
    if model_path:
        return model_path
    if not model_file:
        raise ValueError('no YOLO weights: set model_file in the task file or model_path:=')
    return model_file if os.path.isabs(model_file) else os.path.join(models_dir, model_file)


def yolo_segmenter(model_path, imgsz):
    """Callable color -> [(bool mask, score)] backed by an ultralytics seg model."""
    from ultralytics import YOLO  # optional dependency, only for detector:=yolo_seg

    model = YOLO(model_path)

    def segment(color_bgr, conf):
        result = model.predict(color_bgr, conf=conf, imgsz=imgsz, retina_masks=True,
                               verbose=False)[0]
        if result.masks is None:
            return []
        masks = result.masks.data.cpu().numpy() > 0.5
        scores = result.boxes.conf.cpu().numpy()
        return list(zip(masks, scores.tolist()))

    return segment
