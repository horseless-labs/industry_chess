#!/usr/bin/env python3
"""
Train a chess-piece detector in PyTorch (torchvision Faster R-CNN) from a
small JSON created from ONE video. Designed to be schema-tolerant and
work even with tiny datasets via transfer learning + heavy regularization.

Usage (typical):
  python chess_detector_training.py \
    --video data/chess.mp4 \
    --json data/annotations.json \
    --out runs/exp1 \
    --classes white_pawn white_rook white_knight white_bishop white_queen white_king \
              black_pawn black_rook black_knight black_bishop black_queen black_king

If you already extracted frames, pass --frames_dir instead of --video.
If your JSON is in Label Studio (video) or a COCO-like custom format, the
adapter will try to auto-detect. If your schema differs, edit
`adapter_build_index()` near the TODO blocks.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.ops import box_convert
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import cv2
from PIL import Image
from tqdm import tqdm

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_frames_from_video(video_path: Path, out_dir: Path, every_n: int = 1) -> List[Path]:
    """Extract frames using OpenCV. Returns list of extracted frame paths."""
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    idx = 0
    saved = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            idx += 1
            fpath = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(fpath), frame)
            saved.append(fpath)
        i += 1
    cap.release()
    return saved


# ---------------------------
# Annotation adapter (schema-tolerant)
# ---------------------------

def _norm_to_abs(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    """Convert [0,100] or [0,1] normalized xywh to absolute pixel xywh by guessing scale."""
    # Heuristic: if values look like percentages (<= 100), treat as percent; else treat as fraction
    scale = 100.0 if max(x, y, w, h) > 1.5 else 1.0
    return x/scale*W, y/scale*H, w/scale*W, h/scale*H


def adapter_build_index(
    json_path: Path,
    frames_dir: Path,
    class_names: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build an index mapping image filename -> list of {bbox:[x1,y1,x2,y2], label:str}
    Tries a few common schemas:
      A) COCO-like (images[], annotations[] with image_id, bbox [x,y,w,h], category_id)
      B) Label Studio video export (result[] with value:{x,y,width,height}, frame, original_width/height)
      C) Simple custom: {"frames":[{"file":"frame_000001.jpg","objects":[{"label":"white_pawn","bbox":[x,y,w,h],"format":"xywh","normalized":true}]}]}
    Edit the TODO blocks for custom schemas.
    """
    data = json.loads(Path(json_path).read_text())

    # Normalize class name set
    class_set = set(class_names)

    index: Dict[str, List[Dict[str, Any]]] = {}

    def add_record(img_file: str, bbox_xyxy: List[float], label: str):
        if label not in class_set:
            # Skip labels not in our training set to keep the head small
            return
        index.setdefault(img_file, []).append({"bbox": bbox_xyxy, "label": label})

    # --- Try COCO-like ---
    if isinstance(data, dict) and all(k in data for k in ("images", "annotations")):
        images = {im["id"]: im for im in data["images"]}
        id_to_cat = {}
        if "categories" in data:
            id_to_cat = {c["id"]: c["name"] for c in data["categories"]}
        for ann in data["annotations"]:
            img_info = images.get(ann["image_id"])  # type: ignore
            if not img_info:
                continue
            file_name = img_info["file_name"]
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            label = id_to_cat.get(ann.get("category_id", -1), str(ann.get("category_id", "unknown")))
            add_record(file_name, [x1, y1, x2, y2], label)
        return index

    # --- Try Label Studio (video) ---
    if isinstance(data, list):
        # Label Studio export is often a list of tasks with 
        # task["annotations"][i]["result"][j]
        for task in data:
            results = []
            anns = task.get("annotations", [])
            if anns:
                for a in anns:
                    results.extend(a.get("result", []))
            else:
                results = task.get("result", [])
            for r in results:
                val = r.get("value", {})
                if not {"x", "y", "width", "height"} <= set(val.keys()):
                    continue
                frame_idx = r.get("frame")
                # Guess the file name from frame index
                if frame_idx is None:
                    continue
                img_file = f"frame_{int(frame_idx)+1:06d}.jpg"
                # Need actual image size to denormalize
                img_path = frames_dir / img_file
                if not img_path.exists():
                    continue
                with Image.open(img_path) as im:
                    W, H = im.size
                x, y, w, h = val["x"], val["y"], val["width"], val["height"]
                ax, ay, aw, ah = _norm_to_abs(x, y, w, h, W, H)
                bbox_xyxy = [ax, ay, ax + aw, ay + ah]
                label = val.get("rectanglelabels", val.get("labels", ["object"]))
                if isinstance(label, list):
                    label = label[0]
                add_record(img_file, bbox_xyxy, str(label))
        if index:
            return index

    # --- Try simple custom schema ---
    if isinstance(data, dict) and "frames" in data:
        for f in data["frames"]:
            img_file = f.get("file") or f.get("filename")
            if not img_file:
                # fallback to frame number
                frame_n = f.get("frame")
                if frame_n is None:
                    continue
                img_file = f"frame_{int(frame_n)+1:06d}.jpg"
            objects = f.get("objects", [])
            # Need W,H to denormalize if necessary
            img_path = frames_dir / img_file
            W = H = None
            if img_path.exists():
                with Image.open(img_path) as im:
                    W, H = im.size
            for obj in objects:
                label = obj.get("label", obj.get("class", "object"))
                bbox = obj.get("bbox") or obj.get("xywh")
                fmt = (obj.get("format") or obj.get("fmt") or "xywh").lower()
                normalized = obj.get("normalized", False)
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    if normalized and W is not None and H is not None:
                        x, y, w, h = _norm_to_abs(x, y, w, h, W, H)
                    if fmt == "xywh":
                        bbox_xyxy = [x, y, x + w, y + h]
                    elif fmt == "cxcywh":
                        bbox_xyxy = box_convert(torch.tensor([[x, y, w, h]], dtype=torch.float32), "cxcywh", "xyxy").tolist()[0]
                    else:
                        # assume already xyxy
                        bbox_xyxy = [x, y, w, h]
                    add_record(img_file, bbox_xyxy, str(label))
        if index:
            return index

    # --- TODO: Add your custom schema here ---
    # Raise to signal the user to implement mapping.
    raise ValueError("Unrecognized JSON schema. Edit adapter_build_index() to map your fields.")


# ---------------------------
# Dataset
# ---------------------------

class ChessDetDataset(Dataset):
    def __init__(self, frames_dir: Path, index: Dict[str, List[Dict[str, Any]]], label_to_id: Dict[str, int], augment: bool = True):
        self.frames_dir = frames_dir
        self.items = sorted(index.keys())
        self.index = index
        self.label_to_id = label_to_id
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname = self.items[i]
        img_path = self.frames_dir / fname
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        anns = self.index.get(fname, [])
        boxes = []
        labels = []
        areas = []
        for ann in anns:
            x1, y1, x2, y2 = ann["bbox"]
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W-1, x2), min(H-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            lab = self.label_to_id[ann["label"]]
            labels.append(lab)
            areas.append((x2 - x1) * (y2 - y1))

        if self.augment:
            # Minimal safe augmentation for tiny datasets
            if random.random() < 0.5:
                # Horizontal flip
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                new_boxes = []
                for x1, y1, x2, y2 in boxes:
                    new_x1 = W - x2
                    new_x2 = W - x1
                    new_boxes.append([new_x1, y1, new_x2, y2])
                boxes = new_boxes

        # Convert
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([i]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        return to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------
# Model
# ---------------------------

def build_model(num_classes: int) -> nn.Module:
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ---------------------------
# Train / Val split for a single video
# ---------------------------

def split_train_val(files: List[str], val_every_n: int = 10) -> Tuple[List[str], List[str]]:
    train, val = [], []
    for idx, f in enumerate(sorted(files)):
        (val if (idx % val_every_n == 0) else train).append(f)
    return train, val


# ---------------------------
# Training loop
# ---------------------------

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20):
    model.train()
    losses_avg = 0.0
    for step, (images, targets) in enumerate(tqdm(data_loader, desc=f"epoch {epoch}") ):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses_avg += losses.item()
        if (step + 1) % print_freq == 0:
            avg = losses_avg / (step + 1)
            tqdm.write(f"[epoch {epoch} step {step+1}] loss={avg:.4f} "
                       f"cls={loss_dict.get('loss_classifier', torch.tensor(0.)).item():.3f} "
                       f"box={loss_dict.get('loss_box_reg', torch.tensor(0.)).item():.3f} "
                       f"rpn_cls={loss_dict.get('loss_objectness', torch.tensor(0.)).item():.3f} "
                       f"rpn_box={loss_dict.get('loss_rpn_box_reg', torch.tensor(0.)).item():.3f}")


def evaluate_loss(model, data_loader, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()).item()
            total += losses
            n += 1
    return total / max(1, n)


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', type=Path, help='Path to source video')
    src.add_argument('--frames_dir', type=Path, help='Directory with extracted frames')
    parser.add_argument('--json', type=Path, required=True, help='Annotation JSON path')
    parser.add_argument('--out', type=Path, required=True, help='Output dir for checkpoints, logs')
    parser.add_argument('--classes', nargs='+', required=True, help='List of class names (no background)')
    parser.add_argument('--extract_every', type=int, default=1, help='Sample every Nth frame when extracting')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--val_every_n', type=int, default=10, help='Use every Nth frame for val split')
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ensure_dir(args.out)
    frames_dir = args.frames_dir
    if args.video is not None:
        frames_dir = args.out / 'frames'
        if frames_dir.exists():
            # start clean for determinism
            shutil.rmtree(frames_dir)
        print(f"Extracting frames from {args.video} -> {frames_dir}")
        extract_frames_from_video(args.video, frames_dir, every_n=args.extract_every)

    assert frames_dir is not None

    # Build annotation index
    print("Parsing JSON and building index...")
    index = adapter_build_index(args.json, frames_dir, args.classes)

    # Label map
    # Torchvision detectors expect class ids starting at 1; 0 is reserved for background.
    class_names = list(dict.fromkeys(args.classes))  # stable order, unique
    label_to_id = {name: i+1 for i, name in enumerate(class_names)}

    # Train/val split
    files = sorted(index.keys())
    train_files, val_files = split_train_val(files, val_every_n=args.val_every_n)
    print(f"Train images: {len(train_files)} | Val images: {len(val_files)}")

    train_index = {f: index[f] for f in train_files}
    val_index = {f: index[f] for f in val_files}

    train_ds = ChessDetDataset(frames_dir, train_index, label_to_id, augment=True)
    val_ds   = ChessDetDataset(frames_dir, val_index,   label_to_id, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                              num_workers=4, collate_fn=collate_fn)

    # Model
    num_classes = 1 + len(class_names)  # + background
    model = build_model(num_classes)

    # Optional: freeze most of the backbone for tiny datasets
    for name, p in model.backbone.body.named_parameters():
        if not any(k in name for k in ["layer3", "layer4"]):
            p.requires_grad = False

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float('inf')
    ckpt_best = args.out / 'best.pt'

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate_loss(model, val_loader, device)
        print(f"[epoch {epoch}] val_loss={val_loss:.4f}")
        lr_sched.step()
        # Save checkpoint each epoch
        ckpt_path = args.out / f"epoch_{epoch:03d}.pt"
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'classes': class_names,
            'label_to_id': label_to_id,
        }, ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            shutil.copy2(ckpt_path, ckpt_best)
            print(f"Saved new best checkpoint -> {ckpt_best}")

    print("Training complete.")
    print(f"Best val loss: {best_val:.4f}. Best ckpt: {ckpt_best}")


if __name__ == '__main__':
    main()
