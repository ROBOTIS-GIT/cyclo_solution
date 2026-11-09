from dataclasses import dataclass
import math

from ffw_vision.obb_detector import detect_obb, face_object, ObbParams, windowed_peak
from ffw_vision.seg_detector import detect_seg, plane_axes, support_scene
import numpy as np

# angle sources: seg direction kept / seg vote peak near the box / box angle (no seg partner)
SEG, OBB_PEAK, OBB = 'hybrid:seg', 'hybrid:obb-peak', 'hybrid:obb'


@dataclass(frozen=True)
class HybridParams(ObbParams):
    hybrid_window_deg: float = 25.0  # seg direction kept within this of the box angle
    hybrid_match_px: float = 10.0  # seg partner = mask center this close to the box center
    # m; a box larger than this holds two objects (ambiguous). Single boxes measured up to 67 x
    # 44 mm (they run large), two side by side are 57+ mm wide, end to end 93+ mm long
    box_max_length: float = 0.075
    box_max_width: float = 0.048


def plane_deg(axis, a, b):
    """Direction of a plane vector in plane_axes coordinates, deg mod 90 (box axes)."""
    return math.degrees(math.atan2(axis @ b, axis @ a)) % 90.0


def diff90(x, y):
    return abs((x - y + 45) % 90 - 45)


def detect_hybrid(color_bgr, depth, k, d, segments, boxes, params, rays=None, debug=None):
    """OBB boxes give the objects, center and top height; the seg mask gives the direction (color
    edge votes) and the size (top face points; the boxes come out 4-12 mm too large). The seg
    direction is kept within hybrid_window_deg of the box angle; further off (a wrong peak won,
    seen on tilted objects) the strongest seg vote peak near the box is used. Which side is long
    comes with the direction: the seg axis (as measured with the seg size) when the seg direction
    is kept, else the box's long side. A box without a seg partner keeps its own angle and size.
    Ambiguity (two objects as one) is judged on the box size only (box_max_length/width): the
    boxes count the objects, and a seg mask can swell with a wall shadow next to the object.

    segments: [(bool mask, score)], boxes: [(4x2 corners, score)] in source pixels.
    Returns (objects, plane or None) like detect_seg; debug gets one dict per object.
    """
    scene = support_scene(color_bgr, depth, k, d, params, rays)
    if scene is None:
        return [], None
    n, plane_d = scene.n, scene.plane_d
    a, b = plane_axes(n)
    seg_info, box_info = [], []
    seg_objects, _ = detect_seg(color_bgr, depth, k, d, segments, params, scene.rays, seg_info,
                                scene)
    box_objects, _ = detect_obb(color_bgr, depth, k, d, boxes, params, scene.rays, box_info,
                                scene)
    objects = []
    for box, info in zip(box_objects, box_info):
        box_deg = plane_deg(info['axis'], a, b)
        partner = None
        for seg, sinfo in zip(seg_objects, seg_info):
            dist = math.hypot(seg.u - box.u, seg.v - box.v)
            if dist <= params.hybrid_match_px and (partner is None or dist < partner[0]):
                partner = (dist, seg, sinfo)
        final, source, mask_outline, votes = box_deg, OBB, None, {}
        length_m, width_m = info['length_m'], info['width_m']
        box_large = (length_m > params.box_max_length or width_m > params.box_max_width)
        if partner is not None:
            _, seg, sinfo = partner
            mask_outline, votes = seg.outline, sinfo
            length_m, width_m = seg.length_m, seg.width_m
            seg_deg = plane_deg(sinfo['axis'], a, b)
            if diff90(seg_deg, box_deg) <= params.hybrid_window_deg:
                final, source = seg_deg, SEG
            elif 'edge_deg' in sinfo:
                peak = windowed_peak(sinfo['edge_deg'], sinfo['edge_w'], box_deg,
                                     params.hybrid_window_deg)
                if peak is not None:
                    final, source = peak, OBB_PEAK
        if source == SEG:  # seg long side, the one its size was measured along
            axis = partner[2]['axis']
        else:  # turn the box's long axis onto the final direction (the long side stays long)
            turn = math.radians((final - box_deg + 45) % 90 - 45)
            axis = math.cos(turn) * info['axis'] + math.sin(turn) * np.cross(n, info['axis'])
        # the face is ambiguous exactly when the box is too large (limits set accordingly)
        limits = (0.0, 0.0) if box_large else (math.inf, math.inf)
        objects.append(face_object(info['center'], axis, length_m, width_m, info['top_level'],
                                   box.score, box.outline, source, n, k, d, params, mask_outline,
                                   limits))
        if debug is not None:
            region = info['region'] if partner is None else partner[2]['region']
            debug.append(dict(votes, mask=info['mask'], region=region | info['region'],
                              top_level=info['top_level'], rect_deg=box_deg, box_deg=box_deg,
                              final_deg=final, source=source))
    return objects, (n, plane_d)

