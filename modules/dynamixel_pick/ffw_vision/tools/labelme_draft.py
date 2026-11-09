#!/usr/bin/env python3
"""Rule-based labelme drafts: non-blue blobs inside the basket; blobs longer than --max-len mm on the table become split_me."""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np


def fit_plane(points, rng, tol=0.004, iters=300):
    best = None
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inliers = np.abs((points - a) @ n) < tol
        if best is None or inliers.sum() > best.sum():
            best = inliers
    q = points[best]
    return np.linalg.svd(q - q.mean(0))[2][2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', help='extract_frames.py output folder')
    parser.add_argument('out', help='labelme folder to write')
    parser.add_argument('--frame', default='0', help='frame index suffix to use')
    parser.add_argument('--max-len', type=float, default=62.0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    for f in sorted(glob.glob(os.path.join(args.captures, f'*_{args.frame}_color.png'))):
        stem = f[:-len('_color.png')]
        name = os.path.basename(stem)
        img = cv2.imread(f)
        H, W = img.shape[:2]
        Z = cv2.imread(stem + '_depth.png', cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        k = json.load(open(stem + '.json'))['k']
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        blue = ((h >= 95) & (h <= 125) & (s >= 120) & (v >= 80)).astype(np.uint8)
        _, lab, st, _ = cv2.connectedComponentsWithStats(blue)
        cnts, _ = cv2.findContours((lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        basket = np.zeros_like(blue)
        cv2.drawContours(basket, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
        inner = cv2.erode(basket, np.ones((5, 5), np.uint8))
        obj = cv2.morphologyEx(((inner > 0) & ~blue.astype(bool)).astype(np.uint8),
                               cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # camera-frame points are enough for sizes on the table plane
        vv, uu = np.mgrid[0:H, 0:W]
        valid = (Z > 0.07) & (Z < 1.5)
        pc = np.dstack(((uu - k[2]) * Z / k[0], (vv - k[5]) * Z / k[4], Z))
        ring = (cv2.dilate(basket, np.ones((41, 41), np.uint8)) > 0) & \
            (cv2.dilate(basket, np.ones((9, 9), np.uint8)) == 0) & valid
        n = fit_plane(pc[ring], rng)
        a = np.cross(n, [1, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)

        shapes = []
        for c in cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) < 30:
                continue
            m = np.zeros_like(obj)
            cv2.drawContours(m, [c], -1, 1, -1)
            q = pc[(m > 0) & valid]
            length = max(cv2.minAreaRect((np.column_stack((q @ a, q @ b)) * 1000).astype(np.float32))[1]) \
                if len(q) >= 5 else 0
            shapes.append({'label': 'dynamixel' if length <= args.max_len else 'split_me',
                           'points': cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).tolist(),
                           'group_id': None, 'description': f'{length:.0f} mm',
                           'shape_type': 'polygon', 'flags': {}, 'mask': None})
        shutil.copy(f, os.path.join(args.out, name + '.png'))
        with open(os.path.join(args.out, name + '.json'), 'w') as fp:
            json.dump({'version': '5.5.0', 'flags': {}, 'shapes': shapes, 'imagePath': name + '.png',
                       'imageData': None, 'imageHeight': H, 'imageWidth': W}, fp, indent=1)
        print(name, ' '.join(f'{s["label"]}({s["description"]})' for s in shapes))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
