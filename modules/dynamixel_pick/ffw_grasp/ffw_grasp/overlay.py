import cv2
from ffw_vision.seg_detector import project
import numpy as np

TOP = (0, 255, 0)
CHOSEN = (255, 255, 0)
BLOCKED = (0, 0, 255)
FREE = (160, 160, 160)
BASKET = (0, 200, 255)
UPPER = 0.05  # m above the tip


def draw_box(image, center, axis, normal, half_along, half_across, color, thickness, k, d):
    across = np.cross(normal, axis)
    corners = [center + su * half_along * axis + sv * half_across * across
               for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    pixels = np.round(project(corners, k, d)).astype(np.int32)
    cv2.polylines(image, [pixels], True, color, thickness)


def draw_grasp(image, grasp, k, d, params, text):
    """Green = top face; finger spaces at tip height (and 5 cm up when tilted):
    cyan = chosen, red = blocked (red dots = the scene points blocking it), gray = free."""
    out = image.copy()
    n = grasp.normal
    draw_box(out, grasp.center, grasp.axis, n, grasp.length_m / 2, grasp.width_m / 2, TOP, 1,
             k, d)
    for c in grasp.candidates:
        tip = grasp.center - n * (grasp.target.top_height - c.tip_height)
        color = CHOSEN if c is grasp.chosen else BLOCKED if c.reason else FREE
        if c.reason and c.block_points is not None and len(c.block_points):
            for u, v in np.round(project(c.block_points[::max(1, len(c.block_points) // 300)],
                                         k, d)).astype(int):
                cv2.circle(out, (int(u), int(v)), 1, BLOCKED, -1)
        # the finger at tip height and UPPER above it (shifted when tilted)
        upper = n * UPPER + c.lean * UPPER * np.tan(c.tilt)
        rise = c.opening / 2 * np.sin(c.roll) * np.cos(c.tilt)  # rolled: +close_axis tip higher
        for side in (1, -1):
            finger = (tip + side * (c.opening / 2 + params.finger_max / 2) * c.close_axis
                      + side * rise * n)
            draw_box(out, finger, c.close_axis, n, params.finger_max / 2,
                     params.finger_width / 2, color, 2 if c is grasp.chosen else 1, k, d)
            if c.tilt:
                draw_box(out, finger + upper, c.close_axis, n, params.finger_max / 2,
                         params.finger_width / 2, color, 1, k, d)
    u, v = project([grasp.center], k, d)[0]
    cv2.circle(out, (int(round(u)), int(round(v))), 2, CHOSEN, -1)
    cv2.putText(out, text, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def draw_basket(image, basket, normal, k, d, params):
    """Orange = basket rim at its known size and height (the scene walls)."""
    out = image.copy()
    draw_box(out, basket.center + normal * params.basket_height, basket.axis, normal,
             params.basket_length / 2, params.basket_width / 2, BASKET, 1, k, d)
    return out
