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
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_convert
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 1337) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    every_n: int = 1
) -> List[Path]:
    """
    Extract frames using OpenCV.
    
    Args:
        video_path: Path to input video file
        out_dir: Directory to save extracted frames
        every_n: Extract every Nth frame
        
    Returns:
        List of extracted frame paths
    """
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    saved = []
    frame_idx = 0
    read_idx = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        if read_idx % every_n == 0:
            frame_idx += 1
            fpath = out_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(fpath), frame)
            saved.append(fpath)
        
        read_idx += 1
    
    cap.release()
    return saved


# ---------------------------
# Annotation adapter (schema-tolerant)
# ---------------------------

def _norm_to_abs(
    x: float,
    y: float,
    w: float,
    h: float,
    W: int,
    H: int
) -> Tuple[float, float, float, float]:
    """
    Convert normalized xywh to absolute pixel xywh.
    
    Heuristic: if values look like percentages (<= 100), treat as percent;
    else treat as fraction [0,1].
    """
    scale = 100.0 if max(x, y, w, h) > 1.5 else 1.0
    return x/scale*W, y/scale*H, w/scale*W, h/scale*H


class AnnotationAdapter:
    """Handles multiple annotation schema formats."""
    
    def __init__(self, json_path: Path, frames_dir: Path, class_names: List[str]):
        self.json_path = json_path
        self.frames_dir = frames_dir
        self.class_set = set(class_names)
        self.index: Dict[str, List[Dict[str, Any]]] = {}
        
    def build_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build index mapping image filename -> list of annotations."""
        data = json.loads(self.json_path.read_text())
        
        # Try different schema parsers
        if self._try_coco_schema(data):
            return self.index
        if self._try_label_studio_schema(data):
            return self.index
        if self._try_custom_schema(data):
            return self.index
        
        raise ValueError(
            "Unrecognized JSON schema. Edit AnnotationAdapter to map your fields."
        )
    
    def _add_record(self, img_file: str, bbox_xyxy: List[float], label: str) -> None:
        """Add annotation record if label is in our class set."""
        if label not in self.class_set:
            return
        self.index.setdefault(img_file, []).append({
            "bbox": bbox_xyxy,
            "label": label
        })
    
    def _try_coco_schema(self, data: Any) -> bool:
        """Try parsing COCO-like schema."""
        if not isinstance(data, dict) or not all(k in data for k in ("images", "annotations")):
            return False
        
        images = {im["id"]: im for im in data["images"]}
        id_to_cat = {}
        if "categories" in data:
            id_to_cat = {c["id"]: c["name"] for c in data["categories"]}
        
        for ann in data["annotations"]:
            img_info = images.get(ann["image_id"])
            if not img_info:
                continue
            
            file_name = img_info["file_name"]
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            label = id_to_cat.get(
                ann.get("category_id", -1),
                str(ann.get("category_id", "unknown"))
            )
            self._add_record(file_name, [x1, y1, x2, y2], label)
        
        return bool(self.index)
    
    def _try_label_studio_schema(self, data: Any) -> bool:
        """Try parsing Label Studio video export schema."""
        if not isinstance(data, list):
            return False
        
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
                if frame_idx is None:
                    continue
                
                img_file = f"frame_{int(frame_idx)+1:06d}.jpg"
                img_path = self.frames_dir / img_file
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
                
                self._add_record(img_file, bbox_xyxy, str(label))
        
        return bool(self.index)
    
    def _try_custom_schema(self, data: Any) -> bool:
        """Try parsing custom schema with frames list."""
        if not isinstance(data, dict) or "frames" not in data:
            return False
        
        for f in data["frames"]:
            img_file = f.get("file") or f.get("filename")
            if not img_file:
                frame_n = f.get("frame")
                if frame_n is None:
                    continue
                img_file = f"frame_{int(frame_n)+1:06d}.jpg"
            
            objects = f.get("objects", [])
            img_path = self.frames_dir / img_file
            
            W = H = None
            if img_path.exists():
                with Image.open(img_path) as im:
                    W, H = im.size
            
            for obj in objects:
                label = obj.get("label", obj.get("class", "object"))
                bbox = obj.get("bbox") or obj.get("xywh")
                fmt = (obj.get("format") or obj.get("fmt") or "xywh").lower()
                normalized = obj.get("normalized", False)
                
                if not bbox or len(bbox) != 4:
                    continue
                
                x, y, w, h = bbox
                if normalized and W is not None and H is not None:
                    x, y, w, h = _norm_to_abs(x, y, w, h, W, H)
                
                if fmt == "xywh":
                    bbox_xyxy = [x, y, x + w, y + h]
                elif fmt == "cxcywh":
                    bbox_xyxy = box_convert(
                        torch.tensor([[x, y, w, h]], dtype=torch.float32),
                        "cxcywh",
                        "xyxy"
                    ).tolist()[0]
                else:
                    # Assume already xyxy
                    bbox_xyxy = [x, y, w, h]
                
                self._add_record(img_file, bbox_xyxy, str(label))
        
        return bool(self.index)


# ---------------------------
# Dataset
# ---------------------------

class ChessDetDataset(Dataset):
    """Chess piece detection dataset with optional augmentation."""
    
    def __init__(
        self,
        frames_dir: Path,
        index: Dict[str, List[Dict[str, Any]]],
        label_to_id: Dict[str, int],
        augment: bool = True
    ):
        self.frames_dir = frames_dir
        self.items = sorted(index.keys())
        self.index = index
        self.label_to_id = label_to_id
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.label_to_id[ann["label"]])
            areas.append((x2 - x1) * (y2 - y1))
        
        # Minimal augmentation for tiny datasets
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [[W - x2, y1, W - x1, y2] for x1, y1, x2, y2 in boxes]
        
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([i]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        
        return to_tensor(img), target


def collate_fn(batch):
    """Custom collate function for detection."""
    return tuple(zip(*batch))


# ---------------------------
# Model
# ---------------------------

def build_model(num_classes: int) -> nn.Module:
    """Build Faster R-CNN model with custom head."""
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def freeze_backbone_layers(model: nn.Module) -> None:
    """Freeze early backbone layers for tiny datasets."""
    for name, p in model.backbone.body.named_parameters():
        if not any(k in name for k in ["layer3", "layer4"]):
            p.requires_grad = False


# ---------------------------
# Train / Val split
# ---------------------------

def split_train_val(
    files: List[str],
    val_every_n: int = 10
) -> Tuple[List[str], List[str]]:
    """Split files into train and validation sets."""
    train, val = [], []
    for idx, f in enumerate(sorted(files)):
        (val if idx % val_every_n == 0 else train).append(f)
    return train, val


# ---------------------------
# Training loop
# ---------------------------

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 20
) -> None:
    """Train for one epoch."""
    model.train()
    losses_avg = 0.0
    
    for step, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
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
            avg_loss = losses_avg / (step + 1)
            tqdm.write(
                f"[Epoch {epoch} Step {step+1}] "
                f"loss={avg_loss:.4f} "
                f"cls={loss_dict.get('loss_classifier', torch.tensor(0.)).item():.3f} "
                f"box={loss_dict.get('loss_box_reg', torch.tensor(0.)).item():.3f} "
                f"rpn_cls={loss_dict.get('loss_objectness', torch.tensor(0.)).item():.3f} "
                f"rpn_box={loss_dict.get('loss_rpn_box_reg', torch.tensor(0.)).item():.3f}"
            )


def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate average loss on validation set."""
    model.train()  # Keep in train mode to get loss
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()).item()
            total_loss += losses
            num_batches += 1
    
    return total_loss / max(1, num_batches)


# ---------------------------
# Main
# ---------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train chess piece detector from video annotations"
    )
    
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', type=Path, help='Path to source video')
    src.add_argument('--frames_dir', type=Path, help='Directory with extracted frames')
    
    parser.add_argument('--json', type=Path, required=True,
                       help='Annotation JSON path')
    parser.add_argument('--out', type=Path, required=True,
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--classes', nargs='+', required=True,
                       help='List of class names (no background)')
    
    parser.add_argument('--extract_every', type=int, default=1,
                       help='Sample every Nth frame when extracting')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--val_every_n', type=int, default=10,
                       help='Use every Nth frame for validation split')
    parser.add_argument('--seed', type=int, default=1337,
                       help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ensure_dir(args.out)
    
    # Handle frame extraction
    frames_dir = args.frames_dir
    if args.video is not None:
        frames_dir = args.out / 'frames'
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        print(f"Extracting frames from {args.video} -> {frames_dir}")
        extract_frames_from_video(args.video, frames_dir, every_n=args.extract_every)
    
    assert frames_dir is not None, "frames_dir must be set"
    
    # Build annotation index
    print("Parsing JSON and building index...")
    adapter = AnnotationAdapter(args.json, frames_dir, args.classes)
    index = adapter.build_index()
    
    # Create label mapping (torchvision expects class ids starting at 1)
    class_names = list(dict.fromkeys(args.classes))  # Stable order, unique
    label_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    
    # Train/val split
    files = sorted(index.keys())
    train_files, val_files = split_train_val(files, val_every_n=args.val_every_n)
    print(f"Train images: {len(train_files)} | Val images: {len(val_files)}")
    
    train_index = {f: index[f] for f in train_files}
    val_index = {f: index[f] for f in val_files}
    
    # Create datasets
    train_ds = ChessDetDataset(frames_dir, train_index, label_to_id, augment=True)
    val_ds = ChessDetDataset(frames_dir, val_index, label_to_id, augment=False)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Build model
    num_classes = 1 + len(class_names)  # +1 for background
    model = build_model(num_classes)
    freeze_backbone_layers(model)  # Freeze early layers for tiny datasets
    model.to(device)
    
    # Setup optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Training loop
    best_val_loss = float('inf')
    ckpt_best = args.out / 'best.pt'
    
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate_loss(model, val_loader, device)
        print(f"[Epoch {epoch}] val_loss={val_loss:.4f}")
        
        lr_scheduler.step()
        
        # Save checkpoint
        ckpt_path = args.out / f"epoch_{epoch:03d}.pt"
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'classes': class_names,
            'label_to_id': label_to_id,
        }, ckpt_path)
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            shutil.copy2(ckpt_path, ckpt_best)
            print(f"Saved new best checkpoint -> {ckpt_best}")
    
    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {ckpt_best}")


if __name__ == '__main__':
    main()