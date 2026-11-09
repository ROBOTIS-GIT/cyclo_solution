#!/usr/bin/env python3
"""Train a YOLO11-seg model on the labelme-converted DYNAMIXEL dataset."""
import argparse

from ultralytics import YOLO

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', default='yolo11n-seg.pt')  # COCO-pretrained start point, downloaded if missing
parser.add_argument('--data', default='dynamixel_seg/data.yaml')
parser.add_argument('--project', default='runs')
parser.add_argument('--name', default='dyn_seg_n')
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--device', default='0')  # GPU index, or 'cpu'
parser.add_argument('--imgsz', type=int, default=448)  # network input, multiple of 32; larger upsamples small objects
args = parser.parse_args()

model = YOLO(args.model)
model.train(
    data=args.data,
    epochs=args.epochs,  # passes over the 50 training images
    patience=50,         # stop early when val has not improved for this many epochs
    imgsz=args.imgsz,
    batch=8,             # images per step; lower it if CUDA runs out of memory
    device=args.device,
    workers=2,           # dataloader processes
    degrees=15,          # random rotation, deg
    fliplr=0.5,          # left-right flip probability
    flipud=0.5,          # up-down flip probability
    mosaic=1.0,          # 4-image mosaic probability, off for the last 10 epochs
    project=args.project,
    name=args.name,      # results in runs/<name>, weights in runs/<name>/weights/best.pt
    plots=True,
)
