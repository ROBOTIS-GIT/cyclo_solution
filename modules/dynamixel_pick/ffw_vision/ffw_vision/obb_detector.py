from dataclasses import dataclass
import math

import cv2
from ffw_vision.blob_detector import normalize_angle
from ffw_vision.seg_detector import edge_angle, face_corners, known_face, plane_axes, project
from ffw_vision.seg_detector import SegObject, SegParams
from ffw_vision.seg_detector import support_scene
import numpy as np


@dataclass(frozen=True)
class ObbParams(SegParams):
    top_inner: float = 0.6  # top height from depth inside the box shrunk to this size
    refine_deg: float = 0.0  # > 0: color edge peak within this many deg of the box angle
    vote_band_px: int = 0  # > 0: only edges within this many px of the box border vote
    box_merge_px: float = 8.0  # boxes with centers this close are one object (best score kept)
    report_face_size: bool = False  # output the known face (face_long/mid/short) as the size;
    # measured sizes run small (seg) or large (OBB), and a near-square look opens wide both ways


def on_plane_px(pixels, k, d, n, plane_d, level):
    """Camera-frame points seen at (sub)pixels on the support plane raised by level."""
    xy = cv2.undistortPoints(np.asarray(pixels, np.float64).reshape(-1, 1, 2),
                             np.array(k, np.float64).reshape(3, 3), np.array(d, np.float64))
    rays = np.column_stack((xy.reshape(-1, 2), np.ones(len(xy))))
    return rays * ((level - plane_d) / (rays @ n))[:, None]


def windowed_peak(deg, weights, center, window):
    """Edge-vote peak (deg mod 90) within window of center, smoothed like edge_angle; or None."""
    hist, _ = np.histogram(deg, bins=90, range=(0, 90), weights=weights)
    if hist.sum() <= 0:
        return None
    kernel = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
    smooth = np.convolve(np.concatenate((hist[-9:], hist, hist[:9])), kernel, 'same')[9:-9]
    bins = [b for b in range(90) if abs((b + 0.5 - center + 45) % 90 - 45) <= window]
    best = max(bins, key=lambda b: smooth[b])
    return best + 0.5 if smooth[best] > 0 else None


def box_frame(corners3d, n):
    """(center, long axis, long side, short side) of a 4-corner face on the plane."""
    e = [corners3d[(i + 1) % 4] - corners3d[i] for i in range(4)]
    pairs = [(e[0] - e[2], (np.linalg.norm(e[0]) + np.linalg.norm(e[2])) / 2),
             (e[1] - e[3], (np.linalg.norm(e[1]) + np.linalg.norm(e[3])) / 2)]
    (axis, long_side), (_, short_side) = sorted(pairs, key=lambda p: -p[1])
    axis = axis - (axis @ n) * n
    return corners3d.mean(axis=0), axis / np.linalg.norm(axis), long_side, short_side


def face_object(center, axis, length_m, width_m, top_level, score, outline, source, n, k, d,
                params, mask_outline=None, limits=None):
    """SegObject of a top face on the plane (camera frame): image box, pose, period, ambiguity.

    limits: (long, short) above which the face is ambiguous; default max_single_length/width.
    The ambiguity uses the given (measured) size; with report_face_size the known face is output.
    """
    max_long, max_short = limits or (params.max_single_length, params.max_single_width)
    ambiguous = length_m > max_long or width_m > max_short
    if params.report_face_size:
        length_m, width_m = known_face(top_level, params) or (length_m, width_m)
    # pixel sizes from the reported size: the grasp planner reads these, not length_m
    half_long, half_short = axis * length_m / 2, np.cross(n, axis) * width_m / 2
    (u, v), tip, tail, side, other = project(
        [center, center + half_long, center - half_long, center + half_short,
         center - half_short], k, d)
    angle = normalize_angle(math.atan2(tip[1] - v, tip[0] - u))
    pose = 'standing' if top_level >= params.standing_min_height else 'lying'
    square = (length_m / max(width_m, 1e-6) < params.square_ratio
              or (pose == 'standing' and params.standing_square))
    period = math.pi / 2 if square else math.pi
    return SegObject(
        float(u), float(v), float(np.hypot(*(tip - tail))), float(np.hypot(*(side - other))),
        angle, 0.0 if ambiguous else period, 'unknown' if ambiguous else pose, ambiguous,
        float(score), top_level, float(length_m), float(width_m), outline, source, mask_outline,
        face_corners(center, axis, n, top_level, length_m, width_m, k, d, params))


def detect_obb(color_bgr, depth, k, d, boxes, params, rays=None, debug=None, scene=None):
    """Objects from oriented top-face boxes + aligned depth. Returns (objects, plane or None).

    boxes: [(4x2 corners in source pixels, score)]. Same output as seg_detector.detect_seg.
    debug: list that gets one dict per returned object (tools/angle_debug.py, hybrid_detector).
    scene: support_scene of this image when the caller has it already.
    """
    scene = scene or support_scene(color_bgr, depth, k, d, params, rays)
    if scene is None:
        return [], None
    n, plane_d = scene.n, scene.plane_d
    a, b = plane_axes(n)
    objects, kept = [], []
    # near-square faces sometimes get two boxes turned ~45 deg apart on one object: their overlap
    # is too small for the model's NMS, their centers nearly the same
    for corners, score in sorted(boxes, key=lambda box: -box[1]):
        if score < params.conf:
            continue
        corners = np.asarray(corners, np.float64).reshape(4, 2)
        middle = corners.mean(axis=0)
        if any(np.hypot(*(middle - other)) <= params.box_merge_px for other in kept):
            continue
        kept.append(middle)
        quad = np.zeros(scene.valid.shape, np.uint8)
        cv2.fillPoly(quad, [np.round(corners).astype(np.int32)], 1)
        inner_box = np.zeros_like(quad)
        cv2.fillPoly(inner_box, [np.round(middle + (corners - middle) * params.top_inner)
                                 .astype(np.int32)], 1)
        top = inner_box.astype(bool) & scene.valid & ~scene.blue
        if top.sum() < params.min_top_px:
            top = quad.astype(bool) & scene.valid & ~scene.blue
            if top.sum() < params.min_top_px:
                continue
        top_level = float(np.median(scene.heights[top]))
        center, axis, length_m, width_m = box_frame(
            on_plane_px(corners, k, d, n, plane_d, top_level), n)
        box_deg = math.degrees(math.atan2(axis @ b, axis @ a)) % 90.0
        source = 'obb'
        # vote region: the box with a margin for its rim, without basket blue
        near = cv2.dilate(quad, np.ones((5, 5), np.uint8)).astype(bool) & ~scene.blue
        if params.vote_band_px > 0:  # the border only: no horn or holes inside the top face
            band = np.ones((2 * params.vote_band_px + 1,) * 2, np.uint8)
            near &= ~cv2.erode(quad, band).astype(bool)
        region = cv2.dilate(near.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        region &= scene.inner
        info = {'mask': quad.astype(bool), 'region': region, 'top': top, 'top_level': top_level,
                'rect_deg': box_deg, 'center': center, 'axis': axis, 'length_m': length_m,
                'width_m': width_m}
        if params.refine_deg > 0 or debug is not None:
            votes = {}
            edge_angle(scene.gray, region, scene.rays, n, plane_d, top_level, params, votes)
            info.update(votes)
            if params.refine_deg > 0 and 'edge_deg' in votes:
                peak = windowed_peak(votes['edge_deg'], votes['edge_w'], box_deg,
                                     params.refine_deg)
                if peak is not None:
                    # rotate the box axis by the correction, long side stays long
                    turn = math.radians((peak - box_deg + 45) % 90 - 45)
                    across = np.cross(n, axis)
                    axis = math.cos(turn) * axis + math.sin(turn) * across
                    source = 'obb+edges'
        objects.append(face_object(center, axis, length_m, width_m, top_level, score,
                                   np.round(corners).astype(np.int32).reshape(-1, 1, 2), source,
                                   n, k, d, params))
        if debug is not None:
            debug.append(info)
    return objects, (n, plane_d)


def yolo_obb_detector(model_path, imgsz):
    """Callable color -> [(4x2 corners, score)] backed by an ultralytics OBB model."""
    from ultralytics import YOLO  # optional dependency, only for detector:=yolo_obb

    model = YOLO(model_path)

    def detect(color_bgr, conf):
        result = model.predict(color_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
        if result.obb is None or len(result.obb) == 0:
            return []
        corners = result.obb.xyxyxyxy.cpu().numpy()
        scores = result.obb.conf.cpu().numpy()
        return list(zip(corners, scores.tolist()))

    return detect
