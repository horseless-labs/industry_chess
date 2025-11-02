#!/usr/bin/env python3
"""
Test script for chess piece detector (YOLO or Faster R-CNN).

Usage:
    # Test on single image
    python test_chess_detector.py \
        --model runs/detect/train/weights/best.pt \
        --source test_image.jpg \
        --detector yolo

    # Test on video
    python test_chess_detector.py \
        --model runs/detect/train/weights/best.pt \
        --source test_video.mp4 \
        --detector yolo \
        --save_video

    # Test on directory of images
    python test_chess_detector.py \
        --model runs/detect/train/weights/best.pt \
        --source test_images/ \
        --detector yolo

    # Test with Faster R-CNN
    python test_chess_detector.py \
        --model best.pt \
        --source test_image.jpg \
        --detector fasterrcnn
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image


# ---------------------------
# Label fixes and refinements
# ---------------------------

# Fix king/queen confusion in dataset
LABEL_FIXES = {
    'white_king': 'white_queen',
    'white_queen': 'white_king',
    'black_king': 'black_queen',
    'black_queen': 'black_king',
}

# Enable/disable fixes (set to False to see original labels)
APPLY_KING_QUEEN_FIX = True
APPLY_BISHOP_PAWN_FIX = True


def fix_label(label: str) -> str:
    """Apply label fixes (e.g., swap king/queen)."""
    if APPLY_KING_QUEEN_FIX:
        return LABEL_FIXES.get(label, label)
    return label


def refine_detection(det: Dict, img_height: int) -> Dict:
    """
    Refine detection using confidence and position heuristics.
    
    Args:
        det: Detection dict with 'bbox', 'confidence', 'class'
        img_height: Image height for position calculations
    
    Returns:
        Refined detection dict
    """
    if not APPLY_BISHOP_PAWN_FIX:
        return det
    
    label = det['class']
    conf = det['confidence']
    bbox = det['bbox']
    
    # Calculate vertical position (0 = top, 1 = bottom)
    y_center = (bbox[1] + bbox[3]) / 2
    y_ratio = y_center / img_height
    
    # Only refine low-confidence bishop/pawn detections
    if conf < 0.75 and ('bishop' in label or 'pawn' in label):
        color = 'white' if 'white' in label else 'black'
        
        # For white pieces (bottom of image from white's perspective)
        if 'white' in label:
            if 'pawn' in label and y_ratio < 0.35:
                # White pawn in top third? Probably a bishop
                det['class'] = f'{color}_bishop'
                det['refined'] = True
            elif 'bishop' in label and y_ratio > 0.75:
                # White bishop in bottom area? Probably a pawn
                det['class'] = f'{color}_pawn'
                det['refined'] = True
        
        # For black pieces (top of image from white's perspective)
        else:
            if 'pawn' in label and y_ratio < 0.25:
                # Black pawn very high? Probably a bishop
                det['class'] = f'{color}_bishop'
                det['refined'] = True
            elif 'bishop' in label and y_ratio > 0.65:
                # Black bishop moving forward? Might be pawn
                det['class'] = f'{color}_pawn'
                det['refined'] = True
    
    return det


def should_filter_detection(det: Dict) -> bool:
    """
    Filter out uncertain bishop/pawn detections.
    
    Returns:
        True if detection should be filtered out
    """
    if not APPLY_BISHOP_PAWN_FIX:
        return False
    
    label = det['class']
    conf = det['confidence']
    
    # Filter very low confidence bishop/pawn detections
    if ('bishop' in label or 'pawn' in label) and conf < 0.50:
        return True
    
    return False


def load_yolo_model(model_path: Path):
    """Load YOLO model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install ultralytics: pip install ultralytics")
        raise
    
    model = YOLO(model_path)
    return model


def load_fasterrcnn_model(model_path: Path, device: torch.device):
    """Load Faster R-CNN model."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    ckpt = torch.load(model_path, map_location=device)
    
    class_names = ckpt['classes']
    label_to_id = ckpt['label_to_id']
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    num_classes = 1 + len(class_names)
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    
    return model, id_to_label


def test_yolo_image(model, image_path: Path, conf_threshold: float = 0.25, save_path: Optional[Path] = None):
    """Test YOLO model on single image."""
    results = model(image_path, conf=conf_threshold)[0]
    
    # Get image
    img = cv2.imread(str(image_path))
    img_height = img.shape[0]
    
    # Draw detections
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = results.names[cls]
        
        # Create detection dict
        det = {
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'class': label,
            'refined': False
        }
        
        # Apply label fixes
        det['class'] = fix_label(det['class'])
        
        # Refine using heuristics
        det = refine_detection(det, img_height)
        
        # Filter uncertain detections
        if should_filter_detection(det):
            continue
        
        detections.append(det)
        
        # Draw box
        label = det['class']
        color = (0, 255, 0) if 'white' in label else (255, 0, 0)
        
        # Use orange color if refined by heuristics
        if det.get('refined', False):
            color = (0, 165, 255)  # Orange = auto-corrected
        
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        text = f"{label} {conf:.2f}"
        if det.get('refined', False):
            text += " *"  # Mark refined detections
        
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (int(x1), int(y1) - text_h - 10), (int(x1) + text_w, int(y1)), color, -1)
        cv2.putText(img, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save or display
    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"Saved to {save_path}")
    else:
        cv2.imshow('Detections', img)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detections


def test_fasterrcnn_image(
    model,
    id_to_label: Dict[int, str],
    image_path: Path,
    device: torch.device,
    conf_threshold: float = 0.25,
    save_path: Optional[Path] = None
):
    """Test Faster R-CNN model on single image."""
    from torchvision.transforms.functional import to_tensor
    
    # Load and process image
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = to_tensor(img_pil).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    # Convert to numpy for drawing
    img = cv2.imread(str(image_path))
    
    # Draw detections
    detections = []
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    for box, label_id, conf in zip(boxes, labels, scores):
        if conf < conf_threshold:
            continue
        
        x1, y1, x2, y2 = box
        label = id_to_label[label_id]
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': float(conf),
            'class': label
        })
        
        # Draw box
        color = (0, 255, 0) if 'white' in label else (255, 0, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        text = f"{label} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (int(x1), int(y1) - text_h - 10), (int(x1) + text_w, int(y1)), color, -1)
        cv2.putText(img, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save or display
    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"Saved to {save_path}")
    else:
        cv2.imshow('Detections', img)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detections


def test_video(model, video_path: Path, detector_type: str, device: torch.device, 
               conf_threshold: float = 0.25, save_video: bool = False, 
               id_to_label: Optional[Dict[int, str]] = None):
    """Test model on video."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer if saving
    writer = None
    if save_video:
        output_path = video_path.parent / f"{video_path.stem}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    frame_idx = 0
    total_detections = 0
    refined_count = 0
    
    print("Processing video... (press 'q' to quit)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        img_height = frame.shape[0]
        
        if detector_type == 'yolo':
            # YOLO inference
            results = model(frame, conf=conf_threshold, verbose=False)[0]
            
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = results.names[cls]
                
                # Create detection dict
                det = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': label,
                    'refined': False
                }
                
                # Apply label fixes
                det['class'] = fix_label(det['class'])
                
                # Refine using heuristics
                det = refine_detection(det, img_height)
                
                # Filter uncertain detections
                if should_filter_detection(det):
                    continue
                
                total_detections += 1
                if det.get('refined', False):
                    refined_count += 1
                
                label = det['class']
                
                # Draw
                color = (0, 255, 0) if 'white' in label else (255, 0, 0)
                if det.get('refined', False):
                    color = (0, 165, 255)  # Orange for refined
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                text = f"{label} {conf:.2f}"
                if det.get('refined', False):
                    text += " *"
                cv2.putText(frame, text, (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        else:  # fasterrcnn
            from torchvision.transforms.functional import to_tensor
            
            # Convert frame to tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            img_tensor = to_tensor(frame_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = model(img_tensor)[0]
            
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            
            for box, label_id, conf in zip(boxes, labels, scores):
                if conf < conf_threshold:
                    continue
                
                x1, y1, x2, y2 = box
                label = id_to_label[label_id]
                
                # Create detection dict
                det = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class': label,
                    'refined': False
                }
                
                # Apply label fixes
                det['class'] = fix_label(det['class'])
                
                # Refine using heuristics
                det = refine_detection(det, img_height)
                
                # Filter uncertain detections
                if should_filter_detection(det):
                    continue
                
                total_detections += 1
                if det.get('refined', False):
                    refined_count += 1
                
                label = det['class']
                
                # Draw
                color = (0, 255, 0) if 'white' in label else (255, 0, 0)
                if det.get('refined', False):
                    color = (0, 165, 255)
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                text = f"{label} {conf:.2f}"
                if det.get('refined', False):
                    text += " *"
                cv2.putText(frame, text, (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Detection', frame)
        
        # Write frame
        if writer:
            writer.write(frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessed {frame_idx} frames")
    print(f"Total detections: {total_detections}")
    print(f"Auto-corrected detections: {refined_count} ({refined_count/max(1,total_detections)*100:.1f}%)")
    print(f"Average detections per frame: {total_detections/frame_idx:.1f}")


def test_directory(model, dir_path: Path, detector_type: str, device: torch.device,
                  conf_threshold: float = 0.25, save_output: bool = False,
                  id_to_label: Optional[Dict[int, str]] = None):
    """Test model on directory of images."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in dir_path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {dir_path}")
        return
    
    print(f"Found {len(image_files)} images")
    
    output_dir = None
    if save_output:
        output_dir = dir_path / 'detections'
        output_dir.mkdir(exist_ok=True)
        print(f"Saving outputs to: {output_dir}")
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        save_path = output_dir / f"{img_path.stem}_detected{img_path.suffix}" if save_output else None
        
        if detector_type == 'yolo':
            detections = test_yolo_image(model, img_path, conf_threshold, save_path)
        else:
            detections = test_fasterrcnn_image(model, id_to_label, img_path, device, 
                                               conf_threshold, save_path)
        
        print(f"  Found {len(detections)} pieces")
        
        # Print detection summary
        class_counts = {}
        for det in detections:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for cls, count in sorted(class_counts.items()):
            print(f"    {cls}: {count}")


def print_summary(detections: List[Dict]):
    """Print detection summary statistics."""
    if not detections:
        print("No detections found!")
        return
    
    print("\n=== Detection Summary ===")
    print(f"Total detections: {len(detections)}")
    
    # Class distribution
    class_counts = {}
    for det in detections:
        cls = det['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("\nPiece counts:")
    for cls in sorted(class_counts.keys()):
        print(f"  {cls}: {class_counts[cls]}")
    
    # Confidence statistics
    confidences = [det['confidence'] for det in detections]
    print(f"\nConfidence stats:")
    print(f"  Mean: {np.mean(confidences):.3f}")
    print(f"  Min:  {np.min(confidences):.3f}")
    print(f"  Max:  {np.max(confidences):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Test chess piece detector")
    parser.add_argument('--model', type=Path, required=True,
                       help='Path to model weights')
    parser.add_argument('--source', type=Path, required=True,
                       help='Path to image, video, or directory')
    parser.add_argument('--detector', type=str, required=True,
                       choices=['yolo', 'fasterrcnn'],
                       help='Detector type')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--save', action='store_true',
                       help='Save output images/video')
    parser.add_argument('--save_video', action='store_true',
                       help='Save output video (for video source)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {args.detector} model from {args.model}...")
    
    if args.detector == 'yolo':
        model = load_yolo_model(args.model)
        id_to_label = None
    else:
        model, id_to_label = load_fasterrcnn_model(args.model, device)
    
    print("Model loaded successfully!")
    
    # Determine source type
    if args.source.is_file():
        if args.source.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video
            test_video(model, args.source, args.detector, device, args.conf, 
                      args.save_video, id_to_label)
        else:
            # Single image
            print(f"\nTesting on image: {args.source}")
            save_path = args.source.parent / f"{args.source.stem}_detected{args.source.suffix}" if args.save else None
            
            if args.detector == 'yolo':
                detections = test_yolo_image(model, args.source, args.conf, save_path)
            else:
                detections = test_fasterrcnn_image(model, id_to_label, args.source, 
                                                   device, args.conf, save_path)
            
            print_summary(detections)
    
    elif args.source.is_dir():
        # Directory of images
        test_directory(model, args.source, args.detector, device, args.conf, 
                      args.save, id_to_label)
    
    else:
        print(f"Source not found: {args.source}")


if __name__ == '__main__':
    main()