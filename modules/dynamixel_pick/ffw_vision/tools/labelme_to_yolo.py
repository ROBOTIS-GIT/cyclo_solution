#!/usr/bin/env python3
"""Convert labelme polygon folders (train/, val/) into an ultralytics YOLO-seg dataset."""
import argparse
import glob
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('src', help='folder holding train/ and val/ labelme folders')
    parser.add_argument('dst', help='output dataset folder')
    parser.add_argument('--names', default='dynamixel', help='comma separated class names, in id order')
    parser.add_argument('--yaml-path', default=None, help='dataset path written into data.yaml (container path)')
    parser.add_argument('--splits', default='train,val,test', help='sub folders to convert, skipped when missing')
    args = parser.parse_args()
    names = args.names.split(',')

    splits = [s for s in args.splits.split(',') if os.path.isdir(os.path.join(args.src, s))]
    for split in splits:
        os.makedirs(os.path.join(args.dst, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(args.dst, 'labels', split), exist_ok=True)
        count = 0
        for jf in sorted(glob.glob(os.path.join(args.src, split, '*.json'))):
            j = json.load(open(jf))
            w, h = j['imageWidth'], j['imageHeight']
            stem = os.path.splitext(os.path.basename(jf))[0]
            lines = []
            for s in j['shapes']:
                if s['label'] not in names:  # e.g. a leftover split_me
                    raise SystemExit(f'{jf}: unknown label {s["label"]!r}')
                if s['shape_type'] != 'polygon' or len(s['points']) < 3:
                    raise SystemExit(f'{jf}: not a polygon')
                xy = ' '.join(f'{min(max(x / w, 0), 1):.6f} {min(max(y / h, 0), 1):.6f}' for x, y in s['points'])
                lines.append(f'{names.index(s["label"])} {xy}')
            image = os.path.join(os.path.dirname(jf), j['imagePath'])
            shutil.copy(image, os.path.join(args.dst, 'images', split, stem + os.path.splitext(image)[1]))
            with open(os.path.join(args.dst, 'labels', split, stem + '.txt'), 'w') as f:
                f.write('\n'.join(lines) + '\n')
            count += 1
        print(f'{split}: {count} images')

    with open(os.path.join(args.dst, 'data.yaml'), 'w') as f:
        f.write(f'path: {args.yaml_path or os.path.abspath(args.dst)}\n')
        f.writelines(f'{s}: images/{s}\n' for s in splits)
        f.write('names:\n')
        f.writelines(f'  {i}: {n}\n' for i, n in enumerate(names))
    print('wrote', os.path.join(args.dst, 'data.yaml'))


if __name__ == '__main__':
    main()
