#!/usr/bin/env python3
"""
Train a chess-piece detector in PyTorch (torchvision Faster R-CNN) from
various annotation formats including YOLO, COCO, and Label Studio.

Usage with YOLO format (Roboflow):
  python chess_detector_training.py \
    --format yolo \
    --data_yaml data/data.yaml \
    --out runs/exp1

Usage with Label Studio JSON:
  python chess_detector_training.py \
    --format labelstudio \
    --video data/chess.mp4 \
    --json data/annotations.json \
    --out runs/exp1 \
    --classes white_pawn white_rook white_knight white_bishop white_queen white_king \
              black_pawn black_rook black_knight black_bishop black_queen black_king

Usage with COCO format:
  python chess_detector_training.py \
    --format coco \
    --json data/annotations.json \
    --frames_dir data/frames \
    --out runs/exp1
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import torch
import yaml
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
    
    def __init__(
        self,
        format_type: str,
        class_names: Optional[List[str]] = None,
        json_path: Optional[Path] = None,
        frames_dir: Optional[Path] = None,
        data_yaml: Optional[Path] = None
    ):
        self.format_type = format_type.lower()
        self.json_path = json_path
        self.frames_dir = frames_dir
        self.data_yaml = data_yaml
        self.class_names = class_names
        self.class_set = set(class_names) if class_names else set()
        self.index: Dict[str, List[Dict[str, Any]]] = {}
        
    def build_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build index mapping image filename -> list of annotations."""
        if self.format_type == 'yolo':
            return self._build_yolo_index()
        elif self.format_type == 'labelstudio':
            return self._build_labelstudio_index()
        elif self.format_type == 'coco':
            return self._build_coco_index()
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")
    
    def _add_record(self, img_file: str, bbox_xyxy: List[float], label: str) -> None:
        """Add annotation record if label is in our class set."""
        if self.class_set and label not in self.class_set:
            return
        self.index.setdefault(img_file, []).append({
            "bbox": bbox_xyxy,
            "label": label
        })
    
    def _build_yolo_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build index from YOLO format (Roboflow style)."""
        if not self.data_yaml:
            raise ValueError("data_yaml required for YOLO format")
        
        # Load YAML config
        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get class names from YAML
        self.class_names = config['names']
        self.class_set = set(self.class_names)
        
        # Get base directory (YAML location)
        base_dir = self.data_yaml.parent
        
        # Process train and val splits
        for split in ['train', 'val', 'test']:
            if split not in config:
                continue
            
            # Get images directory
            img_dir = base_dir / config[split]
            if not img_dir.exists():
                print(f"Warning: {split} images dir not found: {img_dir}")
                continue
            
            # Corresponding labels directory
            labels_dir = img_dir.parent / 'labels'
            if not labels_dir.exists():
                print(f"Warning: {split} labels dir not found: {labels_dir}")
                continue
            
            # Process each image
            for img_path in img_dir.glob('*'):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                # Load image to get dimensions
                with Image.open(img_path) as im:
                    W, H = im.size
                
                # Find corresponding label file
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue
                
                # Parse YOLO format labels
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        
                        # Convert from normalized cxcywh to absolute xyxy
                        cx_abs = cx * W
                        cy_abs = cy * H
                        w_abs = w * W
                        h_abs = h * H
                        
                        x1 = cx_abs - w_abs / 2
                        y1 = cy_abs - h_abs / 2
                        x2 = cx_abs + w_abs / 2
                        y2 = cy_abs + h_abs / 2
                        
                        label = self.class_names[class_id]
                        self._add_record(img_path.name, [x1, y1, x2, y2], label)
        
        return self.index
    
    def _build_labelstudio_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build index from Label Studio video export."""
        if not self.json_path or not self.frames_dir:
            raise ValueError("json_path and frames_dir required for Label Studio format")
        
        data = json.loads(self.json_path.read_text())
        
        if not isinstance(data, list):
            raise ValueError("Label Studio format should be a list of tasks")
        
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
                
                # Handle video rectangle format with sequence
                if "sequence" in val:
                    labels = val.get("labels", ["object"])
                    if isinstance(labels, list):
                        label = labels[0]
                    else:
                        label = str(labels)
                    
                    for seq_item in val["sequence"]:
                        if not seq_item.get("enabled", True):
                            continue
                        
                        frame_idx = seq_item.get("frame")
                        if frame_idx is None:
                            continue
                        
                        img_file = f"frame_{int(frame_idx)+1:06d}.jpg"
                        img_path = self.frames_dir / img_file
                        if not img_path.exists():
                            continue
                        
                        with Image.open(img_path) as im:
                            W, H = im.size
                        
                        x = seq_item.get("x", 0)
                        y = seq_item.get("y", 0)
                        w = seq_item.get("width", 0)
                        h = seq_item.get("height", 0)
                        
                        ax, ay, aw, ah = _norm_to_abs(x, y, w, h, W, H)
                        bbox_xyxy = [ax, ay, ax + aw, ay + ah]
                        
                        self._add_record(img_file, bbox_xyxy, str(label))
                
                # Handle single frame format
                elif {"x", "y", "width", "height"} <= set(val.keys()):
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
        
        return self.index
    
    def _build_coco_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build index from COCO format."""
        if not self.json_path:
            raise ValueError("json_path required for COCO format")
        
        data = json.loads(self.json_path.read_text())
        
        if not isinstance(data, dict) or not all(k in data for k in ("images", "annotations")):
            raise ValueError("Invalid COCO format")
        
        images = {im["id"]: im for im in data["images"]}
        id_to_cat = {}
        if "categories" in data:
            id_to_cat = {c["id"]: c["name"] for c in data["categories"]}
            if not self.class_names:
                self.class_names = [c["name"] for c in sorted(data["categories"], key=lambda x: x["id"])]
                self.class_set = set(self.class_names)
        
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
        
        return self.index


# ---------------------------
# Dataset
# ---------------------------

class ChessDetDataset(Dataset):
    """Chess piece detection dataset with optional augmentation."""
    
    def __init__(
        self,
        image_paths: List[Path],
        index: Dict[str, List[Dict[str, Any]]],
        label_to_id: Dict[str, int],
        augment: bool = True
    ):
        """
        Args:
            image_paths: List of full paths to images
            index: Dict mapping image filename to annotations
            label_to_id: Mapping from label name to class id
            augment: Whether to apply augmentation
        """
        self.image_paths = image_paths
        self.index = index
        self.label_to_id = label_to_id
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_paths[i]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        
        anns = self.index.get(img_path.name, [])
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
    files: List[Path],
    val_ratio: float = 0.1
) -> Tuple[List[Path], List[Path]]:
    """Split files into train and validation sets."""
    random.shuffle(files)
    val_size = int(len(files) * val_ratio)
    return files[val_size:], files[:val_size]


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
        description="Train chess piece detector from various annotation formats"
    )
    
    parser.add_argument('--format', type=str, required=True,
                       choices=['yolo', 'labelstudio', 'coco'],
                       help='Annotation format type')
    
    # YOLO format args
    parser.add_argument('--data_yaml', type=Path,
                       help='Path to YOLO data.yaml (required for YOLO format)')
    
    # Label Studio format args
    parser.add_argument('--video', type=Path,
                       help='Path to source video (for Label Studio format)')
    parser.add_argument('--frames_dir', type=Path,
                       help='Directory with extracted frames')
    parser.add_argument('--json', type=Path,
                       help='Annotation JSON path (for Label Studio/COCO formats)')
    parser.add_argument('--classes', nargs='+',
                       help='List of class names (optional, auto-detected from data)')
    
    parser.add_argument('--out', type=Path, required=True,
                       help='Output directory for checkpoints and logs')
    
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
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation split ratio (for YOLO format, uses existing splits)')
    parser.add_argument('--seed', type=int, default=1337,
                       help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze early backbone layers (recommended for small datasets)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ensure_dir(args.out)
    
    # Handle frame extraction for Label Studio
    frames_dir = args.frames_dir
    if args.format == 'labelstudio':
        if args.video is not None:
            frames_dir = args.out / 'frames'
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            print(f"Extracting frames from {args.video} -> {frames_dir}")
            extract_frames_from_video(args.video, frames_dir, every_n=args.extract_every)
        
        if not frames_dir:
            raise ValueError("Label Studio format requires either --video or --frames_dir")
    
    # Build annotation index
    print("Parsing annotations and building index...")
    adapter = AnnotationAdapter(
        format_type=args.format,
        class_names=args.classes,
        json_path=args.json,
        frames_dir=frames_dir,
        data_yaml=args.data_yaml
    )
    index = adapter.build_index()
    
    if not index:
        raise ValueError("No annotations found! Check your data paths and format.")
    
    # Get class names (from adapter if auto-detected)
    class_names = adapter.class_names or args.classes
    if not class_names:
        raise ValueError("No class names found. Specify --classes or use format with class info.")
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create label mapping (torchvision expects class ids starting at 1)
    class_names = list(dict.fromkeys(class_names))  # Stable order, unique
    label_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    
    # Get all image paths
    all_images = []
    for img_file in index.keys():
        # Try to find the full path
        if args.format == 'yolo' and args.data_yaml:
            base_dir = args.data_yaml.parent
            for split in ['train', 'val', 'test']:
                img_dir = base_dir / f"../{split}/images"
                img_path = img_dir / img_file
                if img_path.exists():
                    all_images.append(img_path.resolve())
                    break
        elif frames_dir:
            img_path = frames_dir / img_file
            if img_path.exists():
                all_images.append(img_path)
    
    print(f"Found {len(all_images)} images with annotations")
    
    # Train/val split
    train_images, val_images = split_train_val(all_images, val_ratio=args.val_ratio)
    print(f"Train images: {len(train_images)} | Val images: {len(val_images)}")
    
    # Create datasets
    train_ds = ChessDetDataset(train_images, index, label_to_id, augment=True)
    val_ds = ChessDetDataset(val_images, index, label_to_id, augment=False)
    
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
    
    if args.freeze_backbone:
        freeze_backbone_layers(model)
        print("Froze early backbone layers")
    
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