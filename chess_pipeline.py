#!/usr/bin/env python3
"""
End-to-end chess-piece detector pipeline.

- Accepts a single .mp4 and a Label Studio JSON (video) export (or COCO/custom).
- Extracts frames (OpenCV if possible, else falls back to ffmpeg).
- Parses annotations and builds an image->boxes index (handles LS videorectangle.sequence).
- Exports a YOLO dataset (images/{train,val}, labels/{train,val}, dataset.yaml).
- (Optional) Trains YOLO in-process via the Ultralytics Python API.

USAGE EXAMPLES
--------------
# A) One-liner: extract frames, export YOLO dataset, then train YOLOv8n
python chess_pipeline.py \
  --video ./data/aca02073-output.mp4 \
  --json  ./data/project-1-at-2025-10-26-16-17-9bfd7434.json \
  --out   runs/exp1 \
  --classes white_pawn white_rook white_knight white_bishop white_queen white_king \
           black_pawn black_rook black_knight black_bishop black_queen black_king \
  --export_yolo --export_dir yolo_export \
  --train_yolo --yolo_model yolov8n.pt --epochs 50 --imgsz 1280 --batch 16

# B) Faster/leaner extraction: only annotated frames (recommended for long videos)
python chess_pipeline.py \
  --video ./data/aca02073-output.mp4 \
  --json  ./data/project-1-at-2025-10-26-16-17-9bfd7434.json \
  --out   runs/exp1 \
  --classes ... \
  --lazy_extract \
  --export_yolo --export_dir yolo_export \
  --train_yolo

# C) If you already extracted frames yourself:
python chess_pipeline.py \
  --frames_dir runs/exp1/frames \
  --json       ./data/project-1-at-2025-10-26-16-17-9bfd7434.json \
  --out        runs/exp1 \
  --classes ... \
  --export_yolo --export_dir yolo_export \
  --train_yolo
"""

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
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


def _open_video_any(path: Path):
    """Try several OpenCV backends; validate by reading one frame."""
    apis = []
    if hasattr(cv2, "CAP_FFMPEG"):
        apis.append(cv2.CAP_FFMPEG)
    if hasattr(cv2, "CAP_GSTREAMER"):
        apis.append(cv2.CAP_GSTREAMER)
    apis.append(cv2.CAP_ANY)

    for api in apis:
        cap = cv2.VideoCapture(str(path), api)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap
            cap.release()
    return None


def extract_frames_from_video(video_path: Path, out_dir: Path, every_n: int = 1) -> List[Path]:
    """Extract frames via OpenCV; fall back to ffmpeg if needed. Names are 1-based: frame_000001.jpg."""
    ensure_dir(out_dir)

    # Try OpenCV path
    cap = _open_video_any(video_path)
    if cap is not None:
        idx = 0
        saved: List[Path] = []
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
        if saved:
            return saved
        # fall through to ffmpeg if OpenCV read nothing

    # Fallback to ffmpeg
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vsync", "0", "-qscale:v", "2"]
        if every_n > 1:
            # keep every Nth frame
            cmd += ["-vf", f"select=not(mod(n\\,{every_n}))", "-vsync", "vfr"]
        cmd += [str(out_dir / "frame_%06d.jpg")]
        print("Falling back to ffmpeg:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        files = sorted(out_dir.glob("frame_*.jpg"))
        if files:
            return [p for p in files]

    raise RuntimeError(f"Failed to open video: {video_path}")


def extract_selected_frames(video_path: Path, out_dir: Path, frame_indices: List[int]) -> List[Path]:
    """
    Extract only the requested zero-based frame indices from video_path into out_dir,
    using names frame_{idx+1:06d}.jpg (1-based filenames).
    """
    ensure_dir(out_dir)
    cap = _open_video_any(video_path)
    saved: List[Path] = []
    if cap is not None:
        for idx in sorted(set(frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            fpath = out_dir / f"frame_{idx + 1:06d}.jpg"
            cv2.imwrite(str(fpath), frame)
            saved.append(fpath)
        cap.release()
        if saved:
            return saved

    # ffmpeg fallback: extract all, then filter (still cheaper than failing)
    all_dir = out_dir / "_tmp_all"
    ensure_dir(all_dir)
    print("Selected-frame extraction fallback: ffmpeg all→filter")
    extract_frames_from_video(video_path, all_dir, every_n=1)
    for idx in sorted(set(frame_indices)):
        src = all_dir / f"frame_{idx + 1:06d}.jpg"
        if src.exists():
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            saved.append(dst)
    shutil.rmtree(all_dir, ignore_errors=True)
    return saved


# ---------------------------
# Annotation adapter (schema-tolerant)
# ---------------------------

def _norm_to_abs(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    """Convert [0,100] or [0,1] normalized xywh to absolute pixel xywh by guessing scale."""
    scale = 100.0 if max(x, y, w, h) > 1.5 else 1.0
    return x/scale*W, y/scale*H, w/scale*W, h/scale*H


def _collect_ls_referenced_frames(ls_data: Any) -> Set[int]:
    """Collect zero-based frame indices referenced in Label Studio video export."""
    referenced: Set[int] = set()
    if not isinstance(ls_data, list):
        return referenced
    for task in ls_data:
        results = []
        anns = task.get("annotations", [])
        if anns:
            for a in anns:
                results.extend(a.get("result", []))
        else:
            results = task.get("result", []) or []
        for r in results:
            val = r.get("value", {})
            seq = val.get("sequence")
            if isinstance(seq, list) and seq:
                for step in seq:
                    if not step.get("enabled", True):
                        continue
                    if "frame" in step:
                        referenced.add(int(step["frame"]))
            elif {"x", "y", "width", "height"} <= set(val.keys()):
                fi = r.get("frame") or val.get("frame")
                if fi is not None:
                    referenced.add(int(fi))
    return referenced


def adapter_build_index(
    json_path: Path,
    frames_dir: Path,
    class_names: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build an index mapping image filename -> list of {bbox:[x1,y1,x2,y2], label:str}
      A) COCO-like
      B) Label Studio video (videorectangle with sequence OR per-frame rectangles)
      C) Simple custom {"frames":[...]}
    """
    data = json.loads(Path(json_path).read_text())

    class_set = set(class_names)
    index: Dict[str, List[Dict[str, Any]]] = {}

    def add_record(img_file: str, bbox_xyxy: List[float], label: str):
        if label not in class_set:
            return
        index.setdefault(img_file, []).append({"bbox": bbox_xyxy, "label": label})

    # --- Try COCO-like ---
    if isinstance(data, dict) and all(k in data for k in ("images", "annotations")):
        images = {im["id"]: im for im in data["images"]}
        id_to_cat = {c["id"]: c["name"] for c in data.get("categories", [])}
        for ann in data["annotations"]:
            img_info = images.get(ann["image_id"])
            if not img_info:
                continue
            file_name = img_info["file_name"]
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            label = id_to_cat.get(ann.get("category_id", -1), str(ann.get("category_id", "unknown")))
            add_record(file_name, [x1, y1, x2, y2], label)
        return index

    # --- Try Label Studio (video) ---
    # Handles:
    #   - videorectangle with value["sequence"] = [{frame,x,y,width,height,enabled}, ...]
    #   - per-frame rectangles with value keys {x,y,width,height} and r["frame"] or value["frame"]
    if isinstance(data, list):
        def _probe_img(frame_idx: int) -> Tuple[str, Path]:
            """Resolve image path for given zero-based frame index. Try 1-based then 0-based names."""
            candidates = [
                f"frame_{int(frame_idx) + 1:06d}.jpg",
                f"frame_{int(frame_idx):06d}.jpg",
            ]
            for name in candidates:
                p = frames_dir / name
                if p.exists():
                    return name, p
            # default to 1-based filename (even if missing)—caller will skip if not exists
            name = candidates[0]
            return name, frames_dir / name

        for task in data:
            results = []
            anns = task.get("annotations", [])
            if anns:
                for a in anns:
                    results.extend(a.get("result", []))
            else:
                results = task.get("result", []) or []

            for r in results:
                val = r.get("value", {})
                labels = val.get("rectanglelabels") or val.get("labels") or ["object"]
                label = labels[0] if isinstance(labels, list) else str(labels)

                seq = val.get("sequence")
                if isinstance(seq, list) and seq:
                    for step in seq:
                        if not step.get("enabled", True):
                            continue
                        frame_idx = step.get("frame")
                        if frame_idx is None:
                            continue
                        img_file, img_path = _probe_img(int(frame_idx))
                        if not img_path.exists():
                            continue
                        try:
                            with Image.open(img_path) as im:
                                W, H = im.size
                        except Exception:
                            continue
                        x = float(step.get("x", 0.0))
                        y = float(step.get("y", 0.0))
                        w = float(step.get("width", 0.0))
                        h = float(step.get("height", 0.0))
                        ax, ay, aw, ah = _norm_to_abs(x, y, w, h, W, H)
                        add_record(img_file, [ax, ay, ax + aw, ay + ah], str(label))
                    continue

                if {"x", "y", "width", "height"} <= set(val.keys()):
                    frame_idx = r.get("frame") or val.get("frame")
                    if frame_idx is None:
                        continue
                    img_file, img_path = _probe_img(int(frame_idx))
                    if not img_path.exists():
                        continue
                    try:
                        with Image.open(img_path) as im:
                            W, H = im.size
                    except Exception:
                        continue
                    x, y, w, h = float(val["x"]), float(val["y"]), float(val["width"]), float(val["height"])
                    ax, ay, aw, ah = _norm_to_abs(x, y, w, h, W, H)
                    add_record(img_file, [ax, ay, ax + aw, ay + ah], str(label))

        if index:
            return index

    # --- Try simple custom schema ---
    if isinstance(data, dict) and "frames" in data:
        for f in data["frames"]:
            img_file = f.get("file") or f.get("filename")
            if not img_file:
                frame_n = f.get("frame")
                if frame_n is None:
                    continue
                img_file = f"frame_{int(frame_n) + 1:06d}.jpg"
            objects = f.get("objects", [])
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
                        x1, y1, x2, y2 = x, y, x + w, y + h
                    elif fmt == "cxcywh":
                        x1, y1, x2, y2 = box_convert(
                            torch.tensor([[x, y, w, h]], dtype=torch.float32),
                            "cxcywh", "xyxy"
                        ).tolist()[0]
                    else:
                        x1, y1, x2, y2 = x, y, w, h  # assume already xyxy
                    add_record(img_file, [x1, y1, x2, y2], str(label))
        if index:
            return index

    raise ValueError("Unrecognized JSON schema. Edit adapter_build_index() to map your fields.")


# ---------------------------
# Minimal TorchVision detector (optional baseline training)
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
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W-1, x2), min(H-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            lab = self.label_to_id[ann["label"]]
            labels.append(lab)
            areas.append((x2 - x1) * (y2 - y1))

        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [[W - x2, y1, W - x1, y2] for x1, y1, x2, y2 in boxes]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([i]), "area": areas, "iscrowd": iscrowd}
        return to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int) -> nn.Module:
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def split_train_val(files: List[str], val_every_n: int = 10) -> Tuple[List[str], List[str]]:
    train, val = [], []
    for idx, f in enumerate(sorted(files)):
        (val if (idx % val_every_n == 0) else train).append(f)
    return train, val


# ---------------------------
# YOLO export + training
# ---------------------------

def export_to_yolo(
    index: Dict[str, List[Dict[str, Any]]],
    frames_dir: Path,
    out_dir: Path,
    class_names: List[str],
) -> Path:
    """
    Create YOLO dataset:
      out_dir/
        images/{train,val}
        labels/{train,val}
        dataset.yaml
    """
    ensure_dir(out_dir)
    class_names = list(dict.fromkeys(class_names))
    name_to_yolo = {n: i for i, n in enumerate(class_names)}

    files = sorted(index.keys())
    train_files, val_files = split_train_val(files, val_every_n=10)

    for split in ("train", "val"):
        (out_dir / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (out_dir / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    def _write_split(split_files: List[str], split: str):
        for fname in split_files:
            img_src = frames_dir / fname
            if not img_src.exists():
                continue
            img_dst = out_dir / f"images/{split}/{fname}"
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_src, img_dst)
            with Image.open(img_src) as im:
                W, H = im.size
            lines: List[str] = []
            for ann in index.get(fname, []):
                label = ann.get("label")
                if label not in name_to_yolo:
                    continue
                x1, y1, x2, y2 = ann["bbox"]
                x1 = max(0.0, float(x1)); y1 = max(0.0, float(y1))
                x2 = min(float(W - 1), float(x2)); y2 = min(float(H - 1), float(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                cx = ((x1 + x2) / 2.0) / float(W)
                cy = ((y1 + y2) / 2.0) / float(H)
                w  = (x2 - x1) / float(W)
                h  = (y2 - y1) / float(H)
                cls_id = name_to_yolo[label]
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            lab_path = (out_dir / f"labels/{split}/{Path(fname).with_suffix('.txt').name}")
            lab_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    _write_split(train_files, "train")
    _write_split(val_files, "val")

    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
        + "\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_text)
    print("YOLO dataset written to:", out_dir)
    return out_dir / "dataset.yaml"


def train_yolo_ultralytics(
    dataset_yaml: Path,
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 1280,
    batch: int = 16,
    lr0: float = 0.001,
):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Ultralytics not installed. Run: pip install ultralytics\n"
            f"Original import error: {e}"
        )

    print(f"Starting YOLO training: model={model_name} epochs={epochs} imgsz={imgsz} batch={batch}")
    model = YOLO(model_name)
    out = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        mosaic=0.0,
        hsv_h=0.0, hsv_s=0.2, hsv_v=0.2,
        degrees=1.0, translate=0.05, scale=0.10,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0, persistent_workers=False,  # safer on low-RAM/VRAM
    )

    # Be tolerant to API differences
    save_dir = None
    best = None
    last = None
    tr = getattr(model, "trainer", None)
    if tr is not None:
        save_dir = getattr(tr, "save_dir", None)
        ckpts = getattr(tr, "ckpt", {}) or {}
        best = ckpts.get("best") if isinstance(ckpts, dict) else None
        # Newer releases store best path separately
        best = best or getattr(tr, "best", None)
        last = getattr(tr, "last", None)

    # Some versions return a Results-like object with save_dir
    if save_dir is None and hasattr(out, "save_dir"):
        save_dir = out.save_dir

    print("YOLO training complete.")
    if save_dir:
        print("Results directory:", save_dir)
    if best:
        print("Best weights:", best)
    if last:
        print("Last weights:", last)

    return {"save_dir": save_dir, "best": best, "last": last}

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', type=Path, help='Path to source video')
    src.add_argument('--frames_dir', type=Path, help='Directory with extracted frames')
    parser.add_argument('--json', type=Path, required=True, help='Annotation JSON path')
    parser.add_argument('--out', type=Path, required=True, help='Output dir (frames/checkpoints/logs)')
    parser.add_argument('--classes', nargs='+', required=True, help='List of class names (no background)')
    parser.add_argument('--extract_every', type=int, default=1, help='Sample every Nth frame when extracting')
    parser.add_argument('--lazy_extract', action='store_true', help='Extract only annotated frames from video')
    parser.add_argument('--seed', type=int, default=1337)

    # YOLO controls
    parser.add_argument('--export_yolo', action='store_true', help='Export YOLO dataset (images/labels + dataset.yaml)')
    parser.add_argument('--export_dir', type=Path, default=Path('yolo_export'), help='YOLO export directory')
    parser.add_argument('--train_yolo', action='store_true', help='Train YOLO after exporting')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt', help='Ultralytics model name/path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr0', type=float, default=0.001)

    args = parser.parse_args()
    set_seed(args.seed)

    # 1) Ensure frames exist
    frames_dir = args.frames_dir
    if args.video is not None:
        frames_dir = args.out / 'frames'
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        if args.lazy_extract:
            # Parse LS JSON to discover referenced frames (zero-based)
            try:
                ls_data = json.loads(Path(args.json).read_text())
            except Exception as e:
                raise RuntimeError(f"Failed to read JSON {args.json}: {e}")
            referenced = _collect_ls_referenced_frames(ls_data)
            if referenced:
                print(f"Extracting {len(referenced)} annotated frames from {args.video} -> {frames_dir}")
                extract_selected_frames(args.video, frames_dir, sorted(referenced))
            else:
                print(f"No explicit frame refs found; extracting every {args.extract_every} frame from {args.video} -> {frames_dir}")
                extract_frames_from_video(args.video, frames_dir, every_n=args.extract_every)
        else:
            print(f"Extracting frames from {args.video} -> {frames_dir}")
            extract_frames_from_video(args.video, frames_dir, every_n=args.extract_every)

    assert frames_dir is not None and frames_dir.exists(), "frames_dir must exist after extraction"

    # 2) Build annotation index (schema tolerant)
    print("Parsing JSON and building index...")
    index = adapter_build_index(args.json, frames_dir, args.classes)
    if not index:
        raise RuntimeError("No annotations matched any frames. Check frame naming or extraction settings.")

    # 3) Export YOLO dataset
    dataset_yaml: Optional[Path] = None
    if args.export_yolo or args.train_yolo:
        dataset_yaml = export_to_yolo(index, frames_dir, args.export_dir, args.classes)

    # 4) Train YOLO (optional)
    if args.train_yolo:
        assert dataset_yaml is not None and dataset_yaml.exists(), "dataset.yaml missing; export step failed."
        train_yolo_ultralytics(
            dataset_yaml=dataset_yaml,
            model_name=args.yolo_model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
        )

    # 5) Done
    if not (args.export_yolo or args.train_yolo):
        print("Index built. Use --export_yolo to create a YOLO dataset, or --train_yolo to train directly.")


if __name__ == '__main__':
    main()

