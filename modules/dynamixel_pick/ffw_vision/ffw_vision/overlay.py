import math

import cv2
import numpy as np

GRID_PX = 50
GRID_COLOR = (160, 160, 160)
TEXT_COLOR = (255, 255, 255)
ROI_COLOR = (0, 255, 0)
COLOR_BOX_COLOR = (128, 128, 128)
VALID_COLOR = (0, 255, 0)
AMBIGUOUS_COLOR = (0, 165, 255)
POSE_TEXT = {'lying': 'L', 'standing': 'S', 'unknown': '?'}
# hybrid direction: s = seg kept, o = seg peak near the OBB box, b = box angle (no seg partner)
ANGLE_SOURCE_TEXT = {'hybrid:seg': 's', 'hybrid:obb-peak': 'o', 'hybrid:obb': 'b'}
FONT = cv2.FONT_HERSHEY_PLAIN


def draw_grid(image, step=GRID_PX):
    out = image.copy()
    height, width = out.shape[:2]
    for u in range(0, width, step):
        cv2.line(out, (u, 0), (u, height - 1), GRID_COLOR, 1)
        cv2.putText(out, str(u), (u + 2, 10), FONT, 0.7, TEXT_COLOR, 1)
    for v in range(step, height, step):
        cv2.line(out, (0, v), (width - 1, v), GRID_COLOR, 1)
        cv2.putText(out, str(v), (2, v + 10), FONT, 0.7, TEXT_COLOR, 1)
    return out


def draw_roi(image, roi):
    u0, v0, u1, v1 = roi
    out = image.copy()
    cv2.rectangle(out, (u0, v0), (u1 - 1, v1 - 1), ROI_COLOR, 1)
    return out


def depth_colormap(depth, mask):
    """Color valid depth by its 2-98 percentile range; invalid pixels stay black."""
    out = np.zeros((*depth.shape, 3), np.uint8)
    if not mask.any():
        return out, float('nan'), float('nan')
    low, high = np.percentile(depth[mask], (2, 98))
    scaled = np.nan_to_num(np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0))
    colored = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[mask] = colored[mask]
    text = f'{low:.2f}m (blue) - {high:.2f}m (red), black = invalid'
    cv2.putText(out, text, (2, out.shape[0] - 4), FONT, 0.8, TEXT_COLOR, 1)
    return out, float(low), float(high)


def draw_box(image, box, color):
    u, v, length, width, angle = box
    rect = ((u, v), (length, width), math.degrees(angle))
    cv2.polylines(image, [np.intp(cv2.boxPoints(rect))], True, color, 1)


def draw_blobs(image, blobs, roi=None, floor=float('nan')):
    """Gray = whole color blob, green/orange = output box, letter = pose, number = mm."""
    out = image.copy()
    if roi is not None and any(roi):
        out = draw_roi(out, roi)
    for blob in blobs:
        color = AMBIGUOUS_COLOR if blob.ambiguous else VALID_COLOR
        if blob.color_box is not None:
            draw_box(out, blob.color_box, COLOR_BOX_COLOR)
        draw_box(out, (blob.u, blob.v, blob.length, blob.width, blob.angle), color)
        half = blob.length / 2
        tip = (blob.u + half * math.cos(blob.angle), blob.v + half * math.sin(blob.angle))
        cv2.line(out, (round(blob.u), round(blob.v)), (round(tip[0]), round(tip[1])), color, 1)
        height = f'{blob.height * 1000:.0f}' if math.isfinite(blob.height) else '-'
        cv2.putText(out, f'{POSE_TEXT.get(blob.pose_label, "?")}{height}',
                    (round(blob.u) + 4, round(blob.v) - 4), FONT, 0.9, color, 1)
    floor_text = f'{floor:.3f}m' if math.isfinite(floor) else '-'
    cv2.putText(out, f'n={len(blobs)} floor={floor_text}', (2, out.shape[0] - 4),
                FONT, 0.9, TEXT_COLOR, 1)
    return out


def draw_seg_objects(image, objects, plane_found):
    """Top face (known size projected from the plane, see seg_detector.face_corners), its center
    and '#number score' (the number of the info panel row): green = usable, orange = ambiguous."""
    out = image.copy()
    for i, obj in enumerate(objects, 1):
        color = AMBIGUOUS_COLOR if obj.ambiguous else VALID_COLOR
        if obj.face_corners is not None:
            cv2.polylines(out, [obj.face_corners], True, color, 1, cv2.LINE_AA)
            corners = obj.face_corners.reshape(-1, 2)
            at = (int(corners[:, 0].max()) + 3, int(corners[:, 1].min()) + 8)
        else:
            draw_box(out, (obj.u, obj.v, obj.length, obj.width, obj.angle), color)
            at = (round(obj.u) + 6, round(obj.v) - 6)
        cv2.circle(out, (round(obj.u), round(obj.v)), 2, color, -1)
        cv2.putText(out, f'#{i} {obj.score:.2f}', at, FONT, 0.8, color, 1, cv2.LINE_AA)
    return out
