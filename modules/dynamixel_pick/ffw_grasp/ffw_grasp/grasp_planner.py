from collections import Counter
from dataclasses import dataclass, replace
import math

import cv2
from ffw_vision.blob_detector import normalize_angle
from ffw_vision.seg_detector import basket_masks, plane_axes, project
import numpy as np

UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class GraspParams:
    max_open: float = 0.090  # gripper max opening, m
    clearance: float = 0.012  # opening beyond the object on each side
    min_clearance: float = 0.005  # less than this per side = too wide to grasp
    backup_clearances: tuple = (0.0,)  # m per side: narrower openings tried after the normal
    # ones; values under min_clearance are skipped (not empty: a ROS parameter needs a type)
    finger_tip: float = 0.005  # finger thickness along the closing axis at the tip
    finger_thick: float = 0.025  # ... at taper_height above the tip
    taper_height: float = 0.030
    finger_max: float = 0.030  # thickest part
    finger_width: float = 0.030  # across the closing axis at the tip
    finger_width_root: float = 0.0  # ... where the finger meets the gripper body; 0 = same
    finger_length: float = 0.0  # tip to the gripper body along the finger axis; 0 = no body check
    body_half_y: float = 0.0  # gripper body: half size along the closing axis
    body_x_min: float = 0.0  # ... extent along the end effector x axis
    body_x_max: float = 0.0
    link_start: float = 0.0  # next part up the axis (e.g. the wrist link with the camera)
    link_half_y: float = 0.0
    link_x_min: float = 0.0
    link_x_max: float = 0.0
    grasp_depth: float = 0.015  # fingertip below the top face
    floor_height: float = 0.0  # where objects rest, above the support plane (e.g. basket floor)
    floor_clearance: float = 0.012  # lowest finger point above the floor after closing
    close_drop: float = 0.0  # fingertips move this far down their axis while closing
    square_ratio: float = 1.25  # measured long / short below this: which side is long is unsure
    long_side_poses: str = ''  # comma list: when unsure, open these poses for known_long
    hover_height: float = 0.10  # hover pose above the grasp pose
    obstacle_points: int = 10  # this many scene points in one finger's space = blocked
    veil_step: int = 2  # px; collision ignores pixels in the middle of a depth jump: mixed
    veil_jump: float = 0.008  # m   depth at object edges, it lands in empty space
    known_short: float = 0.0  # largest the top face can be across its short side; 0 = off
    known_long: float = 0.0  # ... along its long side (measured tops are eroded, so smaller)
    max_tilt_deg: float = 0.0  # tilt about the closing axis, top toward lean_x/y; 0 = vertical
    tilt_step_deg: float = 10.0  # a blocked tilt is retried this much smaller, down to vertical
    lean_x: float = -1.0  # base direction the gripper top leans to (e.g. toward the robot)
    lean_y: float = 0.0
    max_roll_deg: float = 0.0  # tilt about the across axis, only when every upright grasp is
    roll_step_deg: float = 5.0  # blocked: the top leans away from one finger; 0 = off
    roll_top_margin: float = 0.004  # m: the raised fingertip stays this far under the top face
    roll_backup: bool = False  # also list free rolls after the upright options (arm side walls)
    roll_backup_max: int = 8  # rolled backups per candidate, half per lean side
    tip_below_end_effector: float = 0.0  # fingertip below the end effector origin at grasp opening
    camera_z_bias: float = 0.0  # camera-measured base z minus true z (kept apart from arm_z_bias)
    camera_x_bias: float = 0.0  # camera-measured base x minus true x (each kept apart)
    camera_y_bias: float = 0.0
    arm_z_bias: float = 0.0  # reached base z minus commanded z of the arm
    min_depth: float = 0.07
    max_depth: float = 1.0
    basket_length: float = 0.0  # basket outer size, m; 0 = no basket measurement
    basket_width: float = 0.0
    basket_height: float = 0.0
    basket_inner_top: tuple = (0.0, 0.0)  # inner length, width at the rim (walls slope)
    basket_inner_floor: tuple = (0.0, 0.0)  # ... at the floor
    wall_band: float = 0.0  # m: points this close inside a basket wall (or in it) do not block
    # the fingers (fingers may touch the walls; the body still checks them); 0 = off
    basket_rim_band: float = 0.008  # rim = blue points this close to the top level
    basket_min_points: int = 300
    basket_edge_px: int = 4  # rim points this close to the image border = side cut off
    basket_height_tol: float = 0.02  # m: a measured rim further from basket_height = not it
    basket_rim_percentile: float = 97.0  # rim top level = this percentile of blue heights
    basket_extent_percentile: float = 0.5  # rim extents leave out this percent at each end
    basket_cut_margin: float = 0.01  # m: an extent this much under the known size may be cut
    basket_cut_window: float = 0.02  # m: ... when rim points this near its end touch the border
    near_radius: float = 0.3  # m: rolled grasps check the scene points this close to the object
    basket_h_min: int = 95  # blue HSV, as ffw_vision SegParams (both get blue_basket.yaml)
    basket_h_max: int = 125
    basket_s_min: int = 120
    basket_v_min: int = 80


@dataclass(frozen=True)
class Target:
    """The ObjectObservation fields a grasp needs (pixels in the source image)."""

    u: float
    v: float
    length: float
    width: float
    angle: float
    angle_period: float
    pose_label: str
    top_height: float
    score: float


@dataclass(frozen=True)
class Candidate:
    name: str  # A: closes across the short side, B: across the long side
    close_axis: np.ndarray  # camera frame, in the support plane
    grip: float  # object size along close_axis, m
    opening: float
    blocked: int  # most scene points inside one finger's space
    reason: str  # empty = usable
    y_axis: np.ndarray  # end effector y in the base frame (horizontal, closer to the current one)
    wrist_turn: float  # rad about base z from the current end effector
    tilt: float  # rad from vertical about y_axis
    z_axis: np.ndarray  # end effector z (up the fingers) in the base frame
    lean: np.ndarray  # camera frame, in the plane: where the fingers run with height
    lean_base: np.ndarray  # the same direction in the base frame (horizontal)
    tip_height: float  # fingertip center above the support plane
    free_tilts: tuple = ()  # every unblocked tilt, largest first
    roll: float = 0.0  # rad about the across axis; + = the +close_axis fingertip is higher
    free_rolls: tuple = ()  # unblocked (tilt, roll) when no upright tilt is free
    narrow: tuple = ()  # (opening, free tilts) at backup_clearances, widest first
    backup: bool = False  # narrowed from its normal opening
    block_points: np.ndarray = None  # camera frame points in the more blocked finger (drawing)


@dataclass(frozen=True)
class Grasp:
    target: Target
    normal: np.ndarray  # support plane normal (up), camera frame
    center: np.ndarray  # top face center, camera frame
    axis: np.ndarray  # top face long side, camera frame
    length_m: float
    width_m: float
    tip_height: float  # fingertip above the support plane
    table_z: float  # support plane below the center, base z
    candidates: list
    chosen: Candidate  # None if no candidate is usable
    reason: str
    tip: np.ndarray = None  # fingertip target in true base z (camera bias removed)
    position: np.ndarray = None  # end effector command, base frame
    rotation: np.ndarray = None  # columns = end effector x, y, z in the base frame
    hover: np.ndarray = None
    wrist_turn: float = 0.0  # rad about base z from the current end effector
    options: list = None  # every usable GraspOption, chosen first


@dataclass(frozen=True)
class GraspOption:
    """One usable candidate at one unblocked tilt (and roll), as base frame poses."""

    name: str
    tilt: float
    opening: float
    wrist_turn: float
    tip: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    hover: np.ndarray
    roll: float = 0.0  # rad about the across axis; + = the fingertip on +y (end effector) higher
    backup: bool = False  # narrow opening or roll: tried after every normal option


@dataclass(frozen=True)
class Basket:
    center: np.ndarray  # outer bottom center on the support plane, camera frame
    axis: np.ndarray  # long side, camera frame, in the plane
    length_m: float  # measured rim extent (the pose uses the known size)
    width_m: float
    cut: str  # extents cut off by the image border: 'long', 'short', 'long,short' or ''


def quat_to_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(m):
    """[x, y, z, w] of a rotation matrix."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    return [float(x) for x in q]


def pick_target(targets):
    """Target with a measured height and the highest score, or None."""
    measured = [t for t in targets if math.isfinite(t.top_height)]
    return max(measured, key=lambda t: t.score) if measured else None


def rank_objects(frames, radius, min_count):
    """Objects over frames, best first: [(targets, frame indices)].

    frames: per frame, the targets Vision found. Targets with a measured height are matched
    to the objects of the frames before by center (nearest first, one per object and frame,
    within radius px), others start a new object. Order: seen in min_count frames or more
    first, then the higher median score (the best of a single frame may change between two
    objects), then more frames.
    """
    objects = []  # [last center, [(frame index, target)]]
    for f, targets in enumerate(frames):
        measured = [t for t in targets if math.isfinite(t.top_height)]
        pairs = sorted((math.hypot(t.u - o[0][0], t.v - o[0][1]), i, j)
                       for i, t in enumerate(measured) for j, o in enumerate(objects))
        used_t, used_o = set(), set()
        for distance, i, j in pairs:
            if distance > radius:
                break
            if i in used_t or j in used_o:
                continue
            used_t.add(i)
            used_o.add(j)
            objects[j][0] = (measured[i].u, measured[i].v)
            objects[j][1].append((f, measured[i]))
        objects += [[(t.u, t.v), [(f, t)]] for i, t in enumerate(measured) if i not in used_t]
    if not objects:
        raise ValueError('no target in any frame')

    def rank(o):
        return len(o[1]) >= min_count, float(np.median([t.score for _, t in o[1]])), len(o[1])
    objects.sort(key=rank, reverse=True)  # stable: ties keep the first seen first
    return [([t for _, t in o[1]], [f for f, _ in o[1]]) for o in objects]


def pick_object(frames, radius, min_count):
    """The best object of rank_objects: (targets, frame indices, objects seen in min_count
    frames or more)."""
    ranked = rank_objects(frames, radius, min_count)
    seen = sum(len(targets) >= min_count for targets, _ in ranked)
    return ranked[0][0], ranked[0][1], seen


def near_any(xy, points, radius):
    """True if xy is within radius of one of points (base x, y)."""
    return any(math.hypot(xy[0] - p[0], xy[1] - p[1]) <= radius for p in points)


def circular_mean(angles, period):
    t = np.asarray(angles) * 2 * math.pi / period
    return math.atan2(np.sin(t).mean(), np.cos(t).mean()) * period / (2 * math.pi)


def merge_targets(targets, max_jitter, min_count, standing_min_height=0.0):
    """Median target over frames. Returns (target, kept indices).

    Frames whose center is more than max_jitter px from the median are dropped. The pose is
    the majority label (frames with another one dropped), or with standing_min_height > 0 the
    median top height decides it once and every frame stays (a tilted object near the
    threshold flips between labels). Frames that disagree on near square (a measured ratio near
    the threshold) are kept: the shorter period says which side is long is unsure.
    """
    if not targets:
        raise ValueError('no target in any frame')
    uv = np.array([(t.u, t.v) for t in targets])
    offset = np.hypot(*(uv - np.median(uv, axis=0)).T)
    near = [i for i in range(len(targets)) if offset[i] <= max_jitter]
    if standing_min_height > 0 and near:
        height = float(np.median([targets[i].top_height for i in near]))
        pose = 'standing' if height >= standing_min_height else 'lying'
        kept = near
    else:
        poses = Counter(targets[i].pose_label for i in near)
        pose = poses.most_common(1)[0][0] if poses else None
        kept = [i for i in near if targets[i].pose_label == pose]
    if len(kept) < min_count:
        raise ValueError(f'only {len(kept)} of {len(targets)} frames agree on one target'
                         f' ({len(targets) - len(near)} moved over {max_jitter:g} px,'
                         f' {len(near) - len(kept)} with another pose)')
    period = min(targets[i].angle_period for i in kept)
    if period <= 0:
        raise ValueError('target has no orientation')
    group = [targets[i] for i in kept]

    def median(name):
        return float(np.median([getattr(t, name) for t in group]))

    angle = normalize_angle(circular_mean([t.angle for t in group], period))
    return Target(median('u'), median('v'), median('length'), median('width'), angle, period,
                  pose, median('top_height'), median('score')), kept


def merge_planes(planes):
    """Median of (nx, ny, nz, d) rows, normal renormalized."""
    plane = np.median(np.asarray(planes, np.float64), axis=0)
    norm = np.linalg.norm(plane[:3])
    return plane[:3] / norm, float(plane[3])


def on_plane(u, v, k, d, normal, plane_d, height):
    """Camera-frame point seen at pixel (u, v) on the support plane raised by height."""
    camera = np.array(k, np.float64).reshape(3, 3)
    x, y = cv2.undistortPoints(np.array([[[u, v]]], np.float64), camera,
                               np.array(d, np.float64))[0, 0]
    ray = np.array([x, y, 1.0])
    return ray * (height - plane_d) / (ray @ normal)


def half_extent(center, direction, pixels, k, d):
    """Distance along direction whose two projected ends are pixels apart, halved."""
    c = project([center], k, d)[0]
    low, high = 0.0, 0.2
    for _ in range(40):
        mid = (low + high) / 2
        a, b = project([center + direction * mid, center - direction * mid], k, d)
        if (np.hypot(*(a - c)) + np.hypot(*(b - c))) / 2 < pixels:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def top_face(target, k, d, normal, plane_d):
    """(center, long axis, length, width) of the top face in the camera frame."""
    center = on_plane(target.u, target.v, k, d, normal, plane_d, target.top_height)
    # the image of a line on the plane is a line: a nearby pixel on it gives the direction
    step = 10 * np.array([math.cos(target.angle), math.sin(target.angle)])
    toward = on_plane(target.u + step[0], target.v + step[1], k, d, normal, plane_d,
                      target.top_height)
    axis = (toward - center) / np.linalg.norm(toward - center)
    length = 2 * half_extent(center, axis, target.length / 2, k, d)
    width = 2 * half_extent(center, np.cross(normal, axis), target.width / 2, k, d)
    return center, axis, length, width


def merge_faces(targets, k, d, normal, plane_d, period):
    """Top faces measured frame by frame, merged in the camera frame.

    Pixel sizes only match their own frame's angle (near square tops swap sides between frames).
    """
    faces = [top_face(t, k, d, normal, plane_d) for t in targets]
    a, b = plane_axes(normal)
    angle = circular_mean([math.atan2(f[1] @ b, f[1] @ a) for f in faces], period)
    center = np.median([f[0] for f in faces], axis=0)
    return (center, math.cos(angle) * a + math.sin(angle) * b,
            float(np.median([f[2] for f in faces])), float(np.median([f[3] for f in faces])))


def shifted(image, du, dv):
    """out[v, u] = image[v + dv, u + du]; NaN outside."""
    out = np.full(image.shape, np.nan, np.float32)
    h, w = image.shape
    out[max(0, -dv):h - max(0, dv), max(0, -du):w - max(0, du)] = \
        image[max(0, dv):h - max(0, -dv), max(0, du):w - max(0, -du)]
    return out


def veil_pixels(depth, step, jump):
    """Pixels whose depth lies between both neighbors step px away, each more than jump off."""
    veil = np.zeros(depth.shape, bool)
    with np.errstate(invalid='ignore'):
        for du, dv in ((step, 0), (0, step), (step, step), (step, -step)):
            before = depth - shifted(depth, -du, -dv)
            after = shifted(depth, du, dv) - depth
            veil |= ((before > jump) & (after > jump)) | ((before < -jump) & (after < -jump))
    return veil


def finger_blocks(points, heights, center, close_axis, normal, opening, tip_height, params,
                  lean=None, tilt=0.0):
    """Most scene points inside either finger's space; the finger thickens above its tip.

    A tilted finger runs toward lean (camera frame, in the plane) by tan(tilt) per height.
    """
    return max(int(m.sum()) for m in finger_masks(points, heights, center, close_axis, normal,
                                                  opening, tip_height, params, lean, tilt))


def blocking_points(points, heights, center, close_axis, normal, opening, tip_height, params,
                    lean=None, tilt=0.0):
    """(points, heights above the plane) in the more blocked finger, for diagnostics."""
    masks = finger_masks(points, heights, center, close_axis, normal, opening, tip_height,
                         params, lean, tilt)
    worst = max(masks, key=lambda m: m.sum())
    return points[worst], heights[worst]


def finger_masks(points, heights, center, close_axis, normal, opening, tip_height, params,
                 lean=None, tilt=0.0):
    """Scene points inside each finger's space (see finger_blocks)."""
    dz = heights - tip_height
    rel = points - center
    if tilt:
        rel = rel - np.outer(dz * math.tan(tilt), lean)
    along = rel @ close_axis
    across = np.abs(rel @ np.cross(normal, close_axis))
    run = dz / math.cos(tilt)  # length along the finger
    thick = np.minimum(params.finger_tip + (params.finger_thick - params.finger_tip)
                       * run / params.taper_height, params.finger_max)
    width = params.finger_width
    if params.finger_width_root and params.finger_length:
        width = width + (params.finger_width_root - width) * np.clip(
            run / params.finger_length, 0.0, 1.0)
    near = (dz > 0) & (across <= width / 2)
    return [near & (out >= 0) & (out <= thick)
            for out in (along - opening / 2, -along - opening / 2)]


def body_blocks(points, tip, x_axis, y_axis, z_axis, params):
    """Scene points in the gripper body or the next link, each swept up the finger axis."""
    rel = points - tip
    a, b, c = rel @ x_axis, np.abs(rel @ y_axis), rel @ z_axis
    count = 0
    for start, half_y, x_min, x_max in (
            (params.finger_length, params.body_half_y, params.body_x_min, params.body_x_max),
            (params.link_start, params.link_half_y, params.link_x_min, params.link_x_max)):
        if half_y > 0:
            count += int(((c >= start) & (b <= half_y) & (a >= x_min) & (a <= x_max)).sum())
    return count


def rolled_blocks(points, tip, x_axis, y_axis, z_axis, opening, params):
    """Most scene points inside either finger's space for any gripper frame (tip = between the
    fingertips): the finger thickens outward along y as it runs up z."""
    counts = []
    for side in (1, -1):
        rel = points - (tip + y_axis * side * opening / 2)
        run, out, across = rel @ z_axis, side * (rel @ y_axis), np.abs(rel @ x_axis)
        thick = np.minimum(params.finger_tip + (params.finger_thick - params.finger_tip)
                           * run / params.taper_height, params.finger_max)
        width = params.finger_width
        if params.finger_width_root and params.finger_length:
            width = width + (params.finger_width_root - width) * np.clip(
                run / params.finger_length, 0.0, 1.0)
        counts.append(int(((run > 0) & (across <= width / 2) & (out >= 0)
                           & (out <= thick)).sum()))
    return max(counts)


def rolled_frame(up, lean, close, tilt, roll):
    """(x, y, z) of the gripper: z tilted toward lean about close, then rolled about x by roll
    (+ raises the +close fingertip, the top leans toward -close)."""
    z1 = math.cos(tilt) * up + math.sin(tilt) * lean
    y = math.cos(roll) * close + math.sin(roll) * z1
    z = math.cos(roll) * z1 - math.sin(roll) * close
    return np.cross(y, z), y, z


def roll_tip_height(top_height, tilt, roll, opening, params):
    """Center height between the fingertips when rolled: the lower tip above the floor as in
    tip_height_for, the higher tip roll_top_margin under the top; None if both do not fit."""
    rise = opening / 2 * math.sin(abs(roll)) * math.cos(tilt)
    low = tip_height_for(0.0, tilt, params) + rise
    high = top_height - params.roll_top_margin - rise
    if low > high:
        return None
    return min(max(top_height - params.grasp_depth, low), high)


def tip_height_for(top_height, tilt, params):
    """Fingertip center height: grasp_depth under the top, but the lowest finger corner
    (tilted end face, then the drop while closing) stays floor_clearance above the floor."""
    lowest = (params.floor_height + params.floor_clearance
              + params.finger_width / 2 * math.sin(tilt) + params.close_drop * math.cos(tilt))
    return max(top_height - params.grasp_depth, lowest)


def wall_points(points, heights, basket, normal, params):
    """Scene points on the measured basket's walls: from wall_band inside a wall's inner face
    (sloping between the floor and rim sizes) out to wall_band past the outer size."""
    rel = points - basket.center
    x, y = rel @ basket.axis, rel @ np.cross(normal, basket.axis)
    rise = np.clip((heights - params.floor_height)
                   / max(params.basket_height - params.floor_height, 1e-6), 0.0, 1.0)
    (top_l, top_w), (floor_l, floor_w) = params.basket_inner_top, params.basket_inner_floor
    half_l = (floor_l + (top_l - floor_l) * rise) / 2 - params.wall_band
    half_w = (floor_w + (top_w - floor_w) * rise) / 2 - params.wall_band
    out_l = params.basket_length / 2 + params.wall_band
    out_w = params.basket_width / 2 + params.wall_band
    inside_out = (np.abs(x) <= out_l) & (np.abs(y) <= out_w)
    return inside_out & ((np.abs(x) >= half_l) | (np.abs(y) >= half_w))


def plan_grasp(target, plane, depth, k, d, rays, rotation, translation, ee_y, params,
               samples=None, basket=None):
    """Grasp from above: end effector y along the closing axis (horizontal), z up the fingers,
    vertical or tilted about y up to max_tilt_deg.

    plane: (n, d) in the camera frame, n up. depth: m, NaN invalid. rays: pixel_rays of the image.
    rotation, translation: camera frame -> base frame. ee_y: current end effector y in the base.
    samples: the per-frame targets merged into target (default: target alone).
    basket: fit_basket result; with wall_band its walls are left to the arm side's model.
    """
    normal, plane_d = plane
    center, axis, length, width = merge_faces(samples or [target], k, d, normal, plane_d,
                                              target.angle_period)
    tip_height = tip_height_for(target.top_height, 0.0, params)
    with np.errstate(invalid='ignore'):
        valid = np.isfinite(depth) & (depth >= params.min_depth) & (depth <= params.max_depth)
    valid &= ~veil_pixels(depth, params.veil_step, params.veil_jump)
    z = depth[valid]
    points = np.column_stack((rays[valid] * z[:, None], z))
    heights = points @ normal + plane_d
    finger_points, finger_heights = points, heights  # what the fingers may not touch
    if basket is not None and params.wall_band > 0 and params.basket_inner_top[0] > 0:
        keep = ~wall_points(points, heights, basket, normal, params)
        finger_points, finger_heights = points[keep], heights[keep]

    # near square (period pi/2): which side is long is not known, open for the long one
    square = target.angle_period < math.pi - 1e-6
    long_poses = [p.strip() for p in params.long_side_poses.split(',') if p.strip()]
    if target.pose_label in long_poses and (square or length < params.square_ratio * width):
        # an eroded long top can look square: either side may be the long one
        square = True
        grips = (max(length, params.known_long),) * 2
    else:
        short = max(length if square else width, params.known_short)
        grips = (short, short) if square else (short, max(length, params.known_long))
    lean_to = np.array([params.lean_x, params.lean_y, 0.0])
    lean_to /= max(np.linalg.norm(lean_to), 1e-9)
    candidates = []
    for name, close_axis, grip in (('A', np.cross(normal, axis), grips[0]),
                                   ('B', axis, grips[1])):
        opening = min(params.max_open, grip + 2 * params.clearance)
        y = rotation @ close_axis
        y = np.array([y[0], y[1], 0.0]) / math.hypot(y[0], y[1])
        if y @ ee_y < 0:  # gripper is symmetric: take the yaw closer to the current one
            y = -y
        turn = math.atan2(ee_y[0] * y[1] - ee_y[1] * y[0], ee_y[0] * y[0] + ee_y[1] * y[1])
        # tilt only about the closing axis (fingers stay parallel to the gripped faces),
        # as far as the across direction lines up with lean_to
        lean = np.array([-y[1], y[0], 0.0])
        if lean @ lean_to < 0:
            lean = -lean
        lean_cam = rotation.T @ lean
        lean_cam -= (lean_cam @ normal) * normal
        lean_cam /= np.linalg.norm(lean_cam)
        top = math.radians(params.max_tilt_deg) * abs(lean @ lean_to)
        step = math.radians(params.tilt_step_deg)
        y_cam = rotation.T @ y  # the chosen yaw sign: the body is not symmetric about it
        y_cam -= (y_cam @ normal) * normal
        y_cam /= np.linalg.norm(y_cam)
        tilts = sorted({max(top - i * step, 0.0) for i in range(int(top / step) + 1)} | {0.0},
                       reverse=True)

        def scan(width):
            # every tilt is checked: the motion side may need a smaller one
            free, first, last = [], None, None
            for tilt in tilts:
                cand_tip = tip_height_for(target.top_height, tilt, params)
                blocked = finger_blocks(finger_points, finger_heights, center, close_axis,
                                        normal, width, cand_tip, params, lean_cam, tilt)
                z_cam = math.cos(tilt) * normal + math.sin(tilt) * lean_cam
                tip_point = center - normal * (target.top_height - cand_tip)
                body = body_blocks(points, tip_point, np.cross(y_cam, z_cam), y_cam, z_cam,
                                   params)
                last = (tilt, cand_tip, blocked, body)
                if blocked < params.obstacle_points and body < params.obstacle_points:
                    free.append(tilt)
                    first = first or last
            return free, first, last
        free, first, last = scan(opening)
        # the candidate is shown at its largest free tilt
        tilt, cand_tip, blocked, body = first or last
        narrow = []
        for gap in sorted(params.backup_clearances, reverse=True):
            narrow_open = min(params.max_open, grip + 2 * gap)  # not width: that is the top face
            if gap < params.min_clearance or narrow_open >= opening:
                continue
            free_n = scan(narrow_open)[0]
            if free_n:
                narrow.append((narrow_open, tuple(free_n)))
        where, shown = '', None
        if first is None and blocked >= params.obstacle_points:
            # where the blocking points are: above the object = wall or a taller neighbor
            shown, h = blocking_points(finger_points, finger_heights, center, close_axis, normal,
                                       opening, cand_tip, params, lean_cam, tilt)
            where = (f' ({np.percentile(h, 10) * 1000:.0f}-{np.percentile(h, 90) * 1000:.0f}mm'
                     f' high)')
        reason = ('too wide' if grip + 2 * params.min_clearance > params.max_open
                  else f'blocked {blocked}{where}' if blocked >= params.obstacle_points
                  else f'body blocked {body}' if body >= params.obstacle_points else '')
        z_axis = math.cos(tilt) * UP + math.sin(tilt) * lean
        candidates.append(Candidate(name, close_axis, grip, opening, max(blocked, body), reason,
                                    y, turn, tilt, z_axis, lean_cam, lean, cand_tip,
                                    tuple(free), narrow=tuple(narrow) if reason != 'too wide'
                                    else (), block_points=shown))
    usable = [c for c in candidates if not c.reason]
    if not usable and any(c.narrow for c in candidates):
        # nothing fits at the normal opening: the widest free narrow one is shown and chosen
        candidates = [narrowed(c, target.top_height, params) if c.narrow else c
                      for c in candidates]
        usable = [c for c in candidates if not c.reason]
    if not usable and params.max_roll_deg > 0:
        candidates = [roll_candidate(c, points, center, normal, target.top_height, params,
                                     finger_points)
                      for c in candidates]
        usable = [c for c in candidates if not c.reason]
    backups = []  # (candidate, tilt, roll) tried after every upright option
    if usable and not usable[0].roll and params.roll_backup and params.max_roll_deg > 0:
        for c in sorted(candidates, key=lambda c: abs(c.wrist_turn)):
            if c.reason == 'too wide':
                continue
            rolls, tilts, blocked = roll_grid(c, points, center, normal, target.top_height,
                                              params, finger_points)
            away = away_roll_sign(center, c.close_axis, normal, basket, params)

            def free(t, r, blocked=blocked):
                count = blocked(t, r)
                return count is not None and count < params.obstacle_points
            backups += [(c, t, r) for t, r in backup_rolls(rolls, tilts, away,
                                                            params.roll_backup_max, free)]
    # least wrist turn (user 2026-10-29): it also leans best, so the arm moves least
    chosen = min(usable, key=lambda c: abs(c.wrist_turn)) if usable else None

    def to_base(p):
        return rotation @ p + translation

    table_z = float(to_base(center - normal * target.top_height)[2])
    reason = '' if chosen else ', '.join(f'{c.name} {c.reason}' for c in candidates)
    grasp = Grasp(target, normal, center, axis, length, width, tip_height, table_z,
                  candidates, chosen, reason)
    if chosen is None:
        return grasp
    bias = np.array([params.camera_x_bias, params.camera_y_bias, params.camera_z_bias])

    def option(c, tilt, roll=0.0, width=None, backup=False):
        # roll is about close_axis (camera); the base y may point the other way
        base_roll = roll if c.y_axis @ (rotation @ c.close_axis) > 0 else -roll
        x_axis, y_axis, z_axis = rolled_frame(UP, c.lean_base, c.y_axis, tilt, base_roll)
        height = (roll_tip_height(target.top_height, tilt, roll, c.opening, params) if roll
                  else tip_height_for(target.top_height, tilt, params))
        tip = to_base(center - normal * (target.top_height - height)) - bias
        # end effector up the finger axis; hover backs off along the same axis
        position = tip + z_axis * params.tip_below_end_effector - UP * params.arm_z_bias
        return GraspOption(c.name, tilt, width or c.opening, c.wrist_turn, tip, position,
                           np.column_stack((x_axis, y_axis, z_axis)),
                           position + z_axis * params.hover_height, base_roll,
                           backup or c.backup or bool(roll))
    by_turn = sorted(candidates, key=lambda c: abs(c.wrist_turn))
    options = [option(c, t) for c in by_turn if not c.reason for t in c.free_tilts]
    options += [option(c, t, width=w, backup=True) for c in by_turn for w, f in c.narrow
                for t in f]
    options += [option(c, t, r) for c in by_turn if not c.reason for t, r in c.free_rolls]
    options += [option(c, t, r) for c, t, r in backups]
    best = options[0]
    return Grasp(target, normal, center, axis, length, width, chosen.tip_height, table_z,
                 candidates, chosen, '', best.tip, best.position, best.rotation, best.hover,
                 chosen.wrist_turn, options)


def narrowed(c, top_height, params):
    """c at its widest free narrow opening (backup), the rest kept in narrow."""
    width, free = c.narrow[0]
    tilt = free[0]
    return replace(c, opening=width, reason='', tilt=tilt, free_tilts=free, narrow=c.narrow[1:],
                   backup=True, z_axis=math.cos(tilt) * UP + math.sin(tilt) * c.lean_base,
                   tip_height=tip_height_for(top_height, tilt, params), block_points=None)


def roll_grid(c, points, center, normal, top_height, params, finger_points=None):
    """(rolls, tilts, blocked) for rolling c: rolls by roll_step_deg up to max_roll_deg, its
    tilts largest first, blocked(tilt, signed roll) = most scene points in a finger or the
    body, None when the fingertips do not fit the side face."""
    near = points[np.linalg.norm(points - center, axis=1) < params.near_radius]  # fingers, body
    if finger_points is None:
        finger_points = points
    near_f = finger_points[np.linalg.norm(finger_points - center, axis=1) < params.near_radius]
    step, top = math.radians(params.roll_step_deg), math.radians(params.max_roll_deg)
    rolls = [step * i for i in range(1, int(round(top / step)) + 1)]
    lean_to = np.array([params.lean_x, params.lean_y, 0.0])
    lean_to /= max(np.linalg.norm(lean_to), 1e-9)
    tilt_top = math.radians(params.max_tilt_deg) * abs(c.lean_base @ lean_to)
    tilt_step = math.radians(params.tilt_step_deg)
    tilts = sorted({max(tilt_top - i * tilt_step, 0.0)
                    for i in range(int(tilt_top / tilt_step) + 1)} | {0.0}, reverse=True)

    def blocked(tilt, signed):
        height = roll_tip_height(top_height, tilt, signed, c.opening, params)
        if height is None:
            return None
        x, y, z = rolled_frame(normal, c.lean, c.close_axis, tilt, signed)
        tip = center - normal * (top_height - height)
        return max(rolled_blocks(near_f, tip, x, y, z, c.opening, params),
                   body_blocks(near, tip, x, y, z, params))
    return rolls, tilts, blocked


def roll_candidate(c, points, center, normal, top_height, params, finger_points=None):
    """Candidate retried rolled (both ways, roll_step_deg up to max_roll_deg) at each of its
    tilts: a finger whose thick upper part meets a wall leans away from it."""
    if c.reason == 'too wide':
        return c
    rolls, tilts, blocked = roll_grid(c, points, center, normal, top_height, params,
                                      finger_points)
    free, least = [], None
    for roll in rolls:
        for tilt in tilts:
            for signed in (roll, -roll):
                count = blocked(tilt, signed)
                if count is None:
                    continue
                if least is None or count < least:
                    least = count
                if count < params.obstacle_points:
                    free.append((tilt, signed))
    if not free:
        return c
    tilt, roll = free[0]
    return replace(c, reason='', blocked=least, tilt=tilt, roll=roll,
                   tip_height=roll_tip_height(top_height, tilt, roll, c.opening, params),
                   free_rolls=tuple(free))


def away_roll_sign(center, close_axis, normal, basket, params):
    """Roll sign whose top leans away from the nearer basket wall along close_axis
    (+ leans toward -close); 0 without a basket."""
    if basket is None or not params.basket_length:
        return 0
    across = np.cross(normal, basket.axis)
    rel = center - basket.center
    p = np.array([rel @ basket.axis, rel @ across])
    u = np.array([close_axis @ basket.axis, close_axis @ across])
    half = np.array([params.basket_length, params.basket_width]) / 2

    def reach(s):  # distance from the object to the wall along s * close_axis
        return min((half[i] - s * u[i] * p[i] / abs(u[i])) / abs(u[i])
                   for i in range(2) if abs(u[i]) > 1e-9)
    return 1 if reach(1) < reach(-1) else -1


def backup_rolls(rolls, tilts, away, count, free):
    """Rolled backups (tilt, signed roll): per side the count // 2 largest rolls that are
    free (free(tilt, roll)) at some tilt, each at its largest free tilt; small rolls first,
    the away side (sign) first. Stops checking once a side has enough."""
    out = []
    for side in ((away, -away) if away else (1, -1)):
        found = []
        for roll in sorted(rolls, reverse=True):
            if len(found) >= count // 2:
                break
            tilt = next((t for t in tilts if free(t, side * roll)), None)
            if tilt is not None:
                found.append((tilt, side * roll))
        out += found[::-1]
    return out


def fit_basket(color_bgr, depth, k, d, rays, plane, params):
    """Basket from its rim: blue points near the top level, projected on the plane, fitted
    with a rectangle of the known size. Returns (Basket or None, reason).

    A side cut off by the image border is placed from the opposite, visible side.
    """
    if not params.basket_length:
        return None, 'basket off'
    _, blue = basket_masks(color_bgr, params)
    normal, plane_d = plane
    with np.errstate(invalid='ignore'):
        valid = np.isfinite(depth) & (depth >= params.min_depth) & (depth <= params.max_depth)
    # mixed depth at the rim edges spreads the outline outward
    valid &= blue & ~veil_pixels(depth, params.veil_step, params.veil_jump)
    if valid.sum() < params.basket_min_points:
        return None, 'basket not seen'
    z = depth[valid]
    points = np.column_stack((rays[valid] * z[:, None], z))
    heights = points @ normal + plane_d
    top = float(np.percentile(heights, params.basket_rim_percentile))
    if abs(top - params.basket_height) > params.basket_height_tol:
        return None, (f'basket rim at {top * 1000:.0f}mm,'
                      f' expected {params.basket_height * 1000:.0f}mm')
    rim = np.abs(heights - top) < params.basket_rim_band
    if rim.sum() < params.basket_min_points:
        return None, f'basket rim {int(rim.sum())} points'
    vv, uu = np.nonzero(valid)
    vv, uu = vv[rim], uu[rim]
    h, w = depth.shape
    m = params.basket_edge_px
    edge = (uu < m) | (uu >= w - m) | (vv < m) | (vv >= h - m)
    a, b = plane_axes(normal)
    xy = np.column_stack((points[rim] @ a, points[rim] @ b))
    _, _, angle = cv2.minAreaRect((xy * 1000).astype(np.float32))
    t = math.radians(angle)
    axes = [np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])]
    spans, trim = [], params.basket_extent_percentile
    for e in axes:
        s = xy @ e
        spans.append((s, float(np.percentile(s, trim)), float(np.percentile(s, 100 - trim))))
    ext = [hi - lo for _, lo, hi in spans]
    size = (params.basket_length, params.basket_width)

    def over(i):
        return max(0.0, ext[i] - size[0]) + max(0.0, ext[1 - i] - size[1])
    # long side: the pairing where the extents exceed the known sizes least
    if over(1) < over(0) or (over(1) == over(0) and ext[1] > ext[0]):
        axes, spans, ext = axes[::-1], spans[::-1], ext[::-1]
    center, cut = [], []
    for (s, lo, hi), full, name in zip(spans, size, ('long', 'short')):
        # a cut side shortens the extent; a corner touching the border alone does not
        short = hi - lo < full - params.basket_cut_margin
        window = params.basket_cut_window
        low_cut = short and bool(edge[s < lo + window].any())
        high_cut = short and bool(edge[s > hi - window].any())
        if low_cut and not high_cut:
            center.append(hi - full / 2)
        elif high_cut and not low_cut:
            center.append(lo + full / 2)
        else:
            center.append((lo + hi) / 2)
        if low_cut or high_cut:
            cut.append(name)
    c2 = center[0] * axes[0] + center[1] * axes[1]
    return Basket(c2[0] * a + c2[1] * b - plane_d * normal,
                  axes[0][0] * a + axes[0][1] * b, ext[0], ext[1], ','.join(cut)), ''
