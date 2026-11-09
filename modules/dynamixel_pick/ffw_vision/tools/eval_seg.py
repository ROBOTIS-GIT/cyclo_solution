#!/usr/bin/env python3
"""Compare YOLO-seg masks with labels per object: IoU, oversize, center and angle error; writes CSV + overlay sheet."""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO


def label_masks(txt, h, w):
    masks = []
    for line in open(txt).read().split('\n'):
        if not line.strip():
            continue
        xy = np.array(line.split()[1:], dtype=np.float32).reshape(-1, 2) * [w, h]
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(xy).astype(np.int32)], 1)
        masks.append(m.astype(bool))
    return masks


def shape(mask):
    """Centroid (u, v) and long-side angle in deg [0, 180) of the mask's rotated box."""
    ys, xs = np.nonzero(mask)
    (u, v), (a, b), ang = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
    if a < b:
        ang += 90
    return xs.mean(), ys.mean(), ang % 180, max(a, b) / max(min(a, b), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/workspace/vision_ws/runs/dyn_seg_n/weights/best.pt')
    parser.add_argument('--data', default='/workspace/vision_ws/datasets/dynamixel_seg')
    parser.add_argument('--split', default='val')
    parser.add_argument('--captures', default='/workspace/vision_ws/captures/wrist2')  # depth + CameraInfo for mm
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=448)  # same as training
    parser.add_argument('--out', default='/workspace/vision_ws/captures/eval_seg')
    parser.add_argument('--save-pred', action='store_true', help='write predicted polygons (+ .scores)')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.save_pred:
        os.makedirs(os.path.join(args.out, 'pred_labels'), exist_ok=True)
    model = YOLO(args.model)

    rows, tiles = [], []
    for img_path in sorted(glob.glob(os.path.join(args.data, 'images', args.split, '*.png'))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        gt = label_masks(os.path.join(args.data, 'labels', args.split, stem + '.txt'), h, w)
        r = model.predict(img, conf=args.conf, imgsz=args.imgsz, retina_masks=True, verbose=False)[0]
        pred = [m.cpu().numpy() > 0.5 for m in r.masks.data] if r.masks is not None else []
        scores = r.boxes.conf.cpu().numpy() if len(pred) else []
        if args.save_pred:
            polys = [r.masks.xy[i] for i in range(len(pred))] if len(pred) else []
            keep = [i for i, poly in enumerate(polys) if len(poly) >= 3]
            with open(os.path.join(args.out, 'pred_labels', stem + '.txt'), 'w') as f:
                f.writelines('0 ' + ' '.join(f'{x / w:.6f} {y / h:.6f}' for x, y in polys[i]) + '\n' for i in keep)
            with open(os.path.join(args.out, 'pred_labels', stem + '.scores'), 'w') as f:
                f.writelines(f'{scores[i]:.4f}\n' for i in keep)

        depth, fx = None, None
        meta = os.path.join(args.captures, stem + '.json')
        if os.path.exists(meta):
            depth = cv2.imread(os.path.join(args.captures, stem + '_depth.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            fx = json.load(open(meta))['k'][0]

        # greedy one-to-one match by IoU
        iou = np.array([[(p & g).sum() / max((p | g).sum(), 1) for g in gt] for p in pred]).reshape(len(pred), len(gt))
        matched_p, matched_g = set(), set()
        for p, g in sorted(np.ndindex(iou.shape), key=lambda pg: -iou[pg]):
            if iou[p, g] < 0.3 or p in matched_p or g in matched_g:
                continue
            matched_p.add(p)
            matched_g.add(g)
            P, G = pred[p], gt[g]
            pu, pv, pa, _ = shape(P)
            gu, gv, ga, elong = shape(G)
            d_ang = abs((pa - ga + 90) % 180 - 90)
            outside = (P & ~G).sum() / G.sum()  # predicted pixels beyond the label, share of label area
            d_px = np.hypot(pu - gu, pv - gv)
            z = np.median(depth[G & (depth > 0)]) if depth is not None and (G & (depth > 0)).any() else np.nan
            rows.append([stem, g, round(float(scores[p]), 3), round(float(iou[p, g]), 3), round(P.sum() / G.sum(), 3),
                         round(float(outside), 3), round(float(d_px), 2), round(float(d_px * z / fx * 1000), 1) if fx else '',
                         round(float(d_ang), 1), round(float(elong), 2)])
        for g in set(range(len(gt))) - matched_g:
            rows.append([stem, g, 'MISSED', '', '', '', '', '', '', ''])
        for p in set(range(len(pred))) - matched_p:
            rows.append([stem, 'FP', round(float(scores[p]), 3), '', '', '', '', '', '', ''])

        vis = img.copy()
        for G in gt:
            cs, _ = cv2.findContours(G.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, (0, 255, 0), 1)
        for P in pred:
            cs, _ = cv2.findContours(P.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, (0, 0, 255), 1)
        cv2.putText(vis, stem[-9:], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(vis)

    head = 'image,gt_idx,score,iou,area_ratio,outside_ratio,center_err_px,center_err_mm,angle_err_deg,gt_elongation'
    with open(os.path.join(args.out, f'{args.split}.csv'), 'w') as f:
        f.write(head + '\n' + '\n'.join(','.join(map(str, r)) for r in rows) + '\n')
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 2)
    cv2.imwrite(os.path.join(args.out, f'{args.split}_sheet.png'),
                np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    print(head)
    print('\n'.join(','.join(map(str, r)) for r in rows))
    print('wrote', args.out, '(green = label, red = prediction)')


if __name__ == '__main__':
    main()
