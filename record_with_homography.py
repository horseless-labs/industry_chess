#!/usr/bin/env python3
"""
Chess board homography + FEN detection and validation.

Usage:
    # Webcam with auto-selection
    python chess_homography_fen.py --model best.pt
    
    # Specific camera
    python chess_homography_fen.py --model best.pt --source 0
    
    # Video file
    python chess_homography_fen.py --model best.pt --source game.mp4
    
    # With target FEN for validation
    python chess_homography_fen.py --model best.pt --target-fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

Controls:
    a = auto-detect board corners
    m = manual corner selection (click 4 corners: h8, a8, a1, h1)
    p = pause/unpause
    r = reset homography
    q = quit
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# --------------------------------------------
# CONFIG
# --------------------------------------------
BOARD_SIZE = 800  # warped board will be 800x800
SQUARE_SIZE = BOARD_SIZE // 8

# Label fixes (if your dataset has king/queen swapped)
LABEL_FIXES = {
    'white_king': 'white_queen',
    'white_queen': 'white_king',
    'black_king': 'black_queen',
    'black_queen': 'black_king',
}
APPLY_LABEL_FIX = True  # Set to False if your labels are correct

# Map model class names to FEN piece letters
CLASS_TO_FEN = {
    "white_king": "K",
    "white_queen": "Q",
    "white_rook": "R",
    "white_bishop": "B",
    "white_knight": "N",
    "white_pawn": "P",
    "black_king": "k",
    "black_queen": "q",
    "black_rook": "r",
    "black_bishop": "b",
    "black_knight": "n",
    "black_pawn": "p",
}

# --------------------------------------------
# CAMERA / SOURCE
# --------------------------------------------
def list_cameras(max_index=10):
    """Find available cameras."""
    cams = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            cams.append(i)
            cap.release()
    return cams


def open_source(source):
    """Open camera or video file."""
    if source is None:
        cams = list_cameras()
        if not cams:
            print("No webcams found. Provide a video file with --source /path/to/video.mp4")
            sys.exit(1)
        print("Available cameras:")
        for c in cams:
            print(f"  [{c}]")
        idx = int(input("Select camera index: ").strip())
        cap = cv2.VideoCapture(idx)
        return cap
    else:
        # Try int first
        try:
            idx = int(source)
            cap = cv2.VideoCapture(idx)
            return cap
        except ValueError:
            # Assume file
            p = Path(source)
            if not p.exists():
                print(f"Source file not found: {p}")
                sys.exit(1)
            cap = cv2.VideoCapture(str(p))
            return cap


# --------------------------------------------
# BOARD DETECTION (AUTO)
# --------------------------------------------
def detect_board_corners_auto(frame):
    """
    Try to find chessboard using contour detection.
    Returns corners as [h8, a8, a1, h1] or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours[:5]:  # Check top 5 largest
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            # Check if it's roughly square-shaped
            pts = approx.reshape(4, 2).astype(np.float32)
            ordered = order_points(pts)
            
            # Calculate aspect ratio
            w1 = np.linalg.norm(ordered[0] - ordered[1])
            w2 = np.linalg.norm(ordered[2] - ordered[3])
            h1 = np.linalg.norm(ordered[0] - ordered[3])
            h2 = np.linalg.norm(ordered[1] - ordered[2])
            
            aspect_ratio = max(w1, w2) / max(h1, h2)
            
            # Should be roughly square (0.7 to 1.3 ratio)
            if 0.7 < aspect_ratio < 1.3:
                return ordered
    
    return None


# --------------------------------------------
# MANUAL CORNER SELECTION
# --------------------------------------------
manual_points = []

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for manual corner selection."""
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(manual_points) < 4:
            manual_points.append((x, y))
            print(f"Point {len(manual_points)}/4: ({x}, {y})")


def get_manual_corners(frame):
    """
    Let user click 4 corners in order: h8 (top-left), a8 (top-right), 
    a1 (bottom-right), h1 (bottom-left).
    Returns ordered corners or None.
    """
    global manual_points
    manual_points = []
    temp = frame.copy()
    
    window_name = "Click 4 corners: h8, a8, a1, h1 (ESC to cancel)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    labels = ["h8 (top-left)", "a8 (top-right)", "a1 (bottom-right)", "h1 (bottom-left)"]
    
    while True:
        disp = temp.copy()
        
        # Draw existing points
        for i, p in enumerate(manual_points):
            color = (0, 255, 0) if i < 4 else (0, 0, 255)
            cv2.circle(disp, p, 5, color, -1)
            cv2.putText(disp, labels[i], (p[0] + 10, p[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show instruction
        next_label = labels[len(manual_points)] if len(manual_points) < 4 else "Done"
        cv2.putText(disp, f"Click: {next_label}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("Manual selection canceled")
            break
        if len(manual_points) == 4:
            print("All 4 corners selected")
            break
    
    cv2.destroyWindow(window_name)
    
    if len(manual_points) != 4:
        return None
    
    # Return as h8, a8, a1, h1 (already in correct order)
    return np.array(manual_points, dtype=np.float32)


# --------------------------------------------
# HOMOGRAPHY / WARP
# --------------------------------------------
def order_points(pts):
    """
    Order points as: TL, TR, BR, BL.
    This will be mapped to: h8, a8, a1, h1
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and diff
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # Top-left (smallest sum)
    rect[2] = pts[np.argmax(s)]      # Bottom-right (largest sum)
    rect[1] = pts[np.argmin(diff)]   # Top-right (smallest diff)
    rect[3] = pts[np.argmax(diff)]   # Bottom-left (largest diff)
    
    return rect


def compute_homography(src_corners):
    """
    Compute homography from source corners to canonical board.
    src_corners should be [h8, a8, a1, h1]
    """
    # Destination: canonical 800x800 board
    # h8 at (0,0), a8 at (800,0), a1 at (800,800), h1 at (0,800)
    dst_corners = np.array([
        [0, 0],                      # h8
        [BOARD_SIZE, 0],             # a8
        [BOARD_SIZE, BOARD_SIZE],    # a1
        [0, BOARD_SIZE]              # h1
    ], dtype=np.float32)
    
    H, _ = cv2.findHomography(src_corners, dst_corners)
    return H


def warp_board(frame, H):
    """Warp the board to canonical view."""
    warped = cv2.warpPerspective(frame, H, (BOARD_SIZE, BOARD_SIZE))
    return warped


# --------------------------------------------
# YOLO DETECTION
# --------------------------------------------
def load_yolo_model(model_path):
    """Load YOLO model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install ultralytics")
        sys.exit(1)
    
    model = YOLO(model_path)
    return model


def detect_pieces(model, frame, conf_threshold=0.3):
    """
    Run YOLO detection on frame.
    Returns list of detections with bbox in frame coordinates.
    """
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    
    detections = []
    if not results:
        return detections
    
    r = results[0]
    if r.boxes is None:
        return detections
    
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_name = r.names[cls_id]
        
        # Apply label fix if enabled
        if APPLY_LABEL_FIX:
            cls_name = LABEL_FIXES.get(cls_name, cls_name)
        
        detections.append({
            "class": cls_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
            "center": [(x1 + x2) / 2, (y1 + y2) / 2]
        })
    
    return detections


# --------------------------------------------
# MAPPING TO SQUARES
# --------------------------------------------
def point_to_square(x, y):
    """
    Convert warped board coordinates to chess square.
    (0, 0) = h8
    (800, 0) = a8
    (800, 800) = a1
    (0, 800) = h1
    
    Returns square name like "e4" or None if out of bounds.
    """
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return None
    
    # Calculate file (a-h) and rank (1-8)
    file_idx = int(x // SQUARE_SIZE)  # 0=h, 7=a
    rank_idx = int(y // SQUARE_SIZE)  # 0=8, 7=1
    
    file_char = chr(ord('h') - file_idx)  # h, g, f, ..., a
    rank_num = 8 - rank_idx               # 8, 7, 6, ..., 1
    
    return f"{file_char}{rank_num}"


def assign_detections_to_warped(detections, use_bottom=True, nms_threshold=0.3):
    """
    Map detections (already in warped board space) directly to squares.
    
    Args:
        detections: List of detection dicts
        use_bottom: If True, use bottom-center of bbox instead of center
                   (better for tall pieces viewed from low angles)
        nms_threshold: IoU threshold for non-maximum suppression
    
    Returns dict: {square_name: detection}
    """
    # First, apply NMS to remove duplicate detections of same piece
    detections = non_maximum_suppression(detections, nms_threshold)
    
    square_map = {}
    
    for det in detections:
        if use_bottom:
            # Use bottom-center of bounding box (where piece base is)
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = y2  # Bottom of box (piece base)
        else:
            cx, cy = det["center"]
        
        square = point_to_square(cx, cy)
        
        if square is None:
            continue
        
        # If multiple pieces map to same square, keep highest confidence
        if square not in square_map or det["confidence"] > square_map[square]["confidence"]:
            square_map[square] = det
    
    return square_map


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def non_maximum_suppression(detections, iou_threshold=0.5):
    """
    Apply NMS to remove duplicate detections of the same piece.
    Keeps detection with highest confidence when IoU > threshold.
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    
    keep = []
    
    while sorted_dets:
        # Keep highest confidence detection
        best = sorted_dets.pop(0)
        keep.append(best)
        
        # Remove detections that overlap significantly with best
        filtered = []
        for det in sorted_dets:
            iou = compute_iou(best["bbox"], det["bbox"])
            if iou < iou_threshold:
                filtered.append(det)
        
        sorted_dets = filtered
    
    return keep


def assign_detections_to_squares(detections, H):
    """
    Map detections from frame space to board squares using homography.
    Returns dict: {square_name: detection}
    (DEPRECATED: Use assign_detections_to_warped when detecting on warped image)
    """
    square_map = {}
    
    for det in detections:
        cx, cy = det["center"]
        
        # Transform center point through homography
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        warped_pt = cv2.perspectiveTransform(pt, H)[0][0]
        
        wx, wy = warped_pt
        square = point_to_square(wx, wy)
        
        if square is None:
            continue
        
        # If multiple pieces map to same square, keep highest confidence
        if square not in square_map or det["confidence"] > square_map[square]["confidence"]:
            square_map[square] = det
    
    return square_map


# --------------------------------------------
# FEN GENERATION
# --------------------------------------------
def square_map_to_fen(square_map):
    """
    Convert square mapping to FEN notation.
    Returns full FEN string.
    """
    # Build 8x8 board (rank 8 to rank 1)
    board = [['' for _ in range(8)] for _ in range(8)]
    
    for square, det in square_map.items():
        file_char = square[0]
        rank_num = int(square[1])
        
        file_idx = ord(file_char) - ord('a')  # 0=a, 7=h
        rank_idx = 8 - rank_num               # 0=rank 8, 7=rank 1
        
        cls_name = det["class"]
        piece_char = CLASS_TO_FEN.get(cls_name, '')
        
        if piece_char:
            board[rank_idx][file_idx] = piece_char
    
    # Convert to FEN
    fen_rows = []
    for rank in range(8):
        row_fen = ""
        empty_count = 0
        
        for file_idx in range(8):
            if board[rank][file_idx] == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    row_fen += str(empty_count)
                    empty_count = 0
                row_fen += board[rank][file_idx]
        
        if empty_count > 0:
            row_fen += str(empty_count)
        
        fen_rows.append(row_fen)
    
    placement = '/'.join(fen_rows)
    
    # Add standard suffixes (can be customized)
    return f"{placement} w KQkq - 0 1"


# --------------------------------------------
# FEN COMPARISON
# --------------------------------------------
def compare_fen(detected_fen, target_fen):
    """
    Compare two FEN strings (placement part only).
    Returns list of differences.
    """
    def expand_fen_row(row):
        """Expand FEN row to 8 characters."""
        result = []
        for ch in row:
            if ch.isdigit():
                result.extend([''] * int(ch))
            else:
                result.append(ch)
        return result + [''] * (8 - len(result))
    
    detected_rows = detected_fen.split()[0].split('/')
    target_rows = target_fen.split()[0].split('/')
    
    differences = []
    
    for rank_idx, (det_row, tgt_row) in enumerate(zip(detected_rows, target_rows)):
        det_expanded = expand_fen_row(det_row)
        tgt_expanded = expand_fen_row(tgt_row)
        
        rank_num = 8 - rank_idx
        
        for file_idx, (det_piece, tgt_piece) in enumerate(zip(det_expanded, tgt_expanded)):
            if det_piece != tgt_piece:
                file_char = chr(ord('a') + file_idx)
                square = f"{file_char}{rank_num}"
                differences.append({
                    "square": square,
                    "detected": det_piece or '(empty)',
                    "expected": tgt_piece or '(empty)'
                })
    
    return differences


# --------------------------------------------
# VISUALIZATION
# --------------------------------------------
def draw_board_overlay_warped(frame, warped, square_map, H):
    """Draw detected pieces from warped space back onto original frame."""
    overlay = frame.copy()
    H_inv = np.linalg.inv(H)
    
    # Draw board outline
    corners = np.array([
        [0, 0],
        [BOARD_SIZE, 0],
        [BOARD_SIZE, BOARD_SIZE],
        [0, BOARD_SIZE]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    frame_corners = cv2.perspectiveTransform(corners, H_inv).astype(np.int32)
    cv2.polylines(overlay, [frame_corners], True, (0, 255, 0), 3)
    
    # Draw grid
    for i in range(1, 8):
        # Vertical lines
        line_start = np.array([[[i * SQUARE_SIZE, 0]]], dtype=np.float32)
        line_end = np.array([[[i * SQUARE_SIZE, BOARD_SIZE]]], dtype=np.float32)
        start_frame = cv2.perspectiveTransform(line_start, H_inv)[0][0].astype(int)
        end_frame = cv2.perspectiveTransform(line_end, H_inv)[0][0].astype(int)
        cv2.line(overlay, tuple(start_frame), tuple(end_frame), (0, 255, 0), 1)
        
        # Horizontal lines
        line_start = np.array([[[0, i * SQUARE_SIZE]]], dtype=np.float32)
        line_end = np.array([[[BOARD_SIZE, i * SQUARE_SIZE]]], dtype=np.float32)
        start_frame = cv2.perspectiveTransform(line_start, H_inv)[0][0].astype(int)
        end_frame = cv2.perspectiveTransform(line_end, H_inv)[0][0].astype(int)
        cv2.line(overlay, tuple(start_frame), tuple(end_frame), (0, 255, 0), 1)
    
    # Draw square labels (transform centers back to frame space)
    for square, det in square_map.items():
        color = (0, 255, 0) if 'white' in det['class'] else (255, 0, 0)
        
        # Get center in warped space
        cx_warped, cy_warped = det['center']
        
        # Transform back to frame space
        pt_warped = np.array([[[cx_warped, cy_warped]]], dtype=np.float32)
        pt_frame = cv2.perspectiveTransform(pt_warped, H_inv)[0][0]
        
        cx_frame, cy_frame = int(pt_frame[0]), int(pt_frame[1])
        
        cv2.circle(overlay, (cx_frame, cy_frame), 5, color, -1)
        cv2.putText(overlay, square, (cx_frame + 10, cy_frame),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay


def draw_board_overlay(frame, square_map, H):
    """Draw detected pieces on the original frame."""
    overlay = frame.copy()
    
    # Draw board outline
    corners = np.array([
        [0, 0],
        [BOARD_SIZE, 0],
        [BOARD_SIZE, BOARD_SIZE],
        [0, BOARD_SIZE]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    H_inv = np.linalg.inv(H)
    frame_corners = cv2.perspectiveTransform(corners, H_inv).astype(np.int32)
    cv2.polylines(overlay, [frame_corners], True, (0, 255, 0), 3)
    
    # Draw grid
    for i in range(1, 8):
        # Vertical lines
        line_start = np.array([[[i * SQUARE_SIZE, 0]]], dtype=np.float32)
        line_end = np.array([[[i * SQUARE_SIZE, BOARD_SIZE]]], dtype=np.float32)
        start_frame = cv2.perspectiveTransform(line_start, H_inv)[0][0].astype(int)
        end_frame = cv2.perspectiveTransform(line_end, H_inv)[0][0].astype(int)
        cv2.line(overlay, tuple(start_frame), tuple(end_frame), (0, 255, 0), 1)
        
        # Horizontal lines
        line_start = np.array([[[0, i * SQUARE_SIZE]]], dtype=np.float32)
        line_end = np.array([[[BOARD_SIZE, i * SQUARE_SIZE]]], dtype=np.float32)
        start_frame = cv2.perspectiveTransform(line_start, H_inv)[0][0].astype(int)
        end_frame = cv2.perspectiveTransform(line_end, H_inv)[0][0].astype(int)
        cv2.line(overlay, tuple(start_frame), tuple(end_frame), (0, 255, 0), 1)
    
    # Draw square labels
    for square, det in square_map.items():
        color = (0, 255, 0) if 'white' in det['class'] else (255, 0, 0)
        cx, cy = det['center']
        cv2.circle(overlay, (int(cx), int(cy)), 5, color, -1)
        cv2.putText(overlay, square, (int(cx) + 10, int(cy)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay


# --------------------------------------------
# MAIN LOOP
# --------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Chess board FEN detection with homography")
    parser.add_argument("--source", type=str, default=None,
                       help="Camera index or video file (default: select from list)")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to YOLO model weights")
    parser.add_argument("--target-fen", type=str, default=None,
                       help="Target FEN for validation")
    parser.add_argument("--conf", type=float, default=0.3,
                       help="Detection confidence threshold")
    parser.add_argument("--use-bottom", action="store_true", default=True,
                       help="Use bottom of bbox for square assignment (better for low angles)")
    parser.add_argument("--nms", type=float, default=0.5,
                       help="NMS IoU threshold for removing duplicate detections")
    parser.add_argument("--no-auto", action="store_true",
                       help="Skip automatic board detection")
    
    args = parser.parse_args()
    
    # Load model and source
    print("Loading model...")
    model = load_yolo_model(args.model)
    print("Opening video source...")
    cap = open_source(args.source)
    
    H = None
    src_corners = None
    paused = False
    last_fen = None
    
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  a = auto-detect board corners")
    print("  m = manual corner selection")
    print("  p = pause/unpause")
    print("  r = reset homography")
    print("  q = quit")
    print("="*60 + "\n")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video/stream")
                break
        
        disp = frame.copy()
        
        # Try auto-detection once at start (if allowed)
        if H is None and not args.no_auto and src_corners is None:
            auto_corners = detect_board_corners_auto(frame)
            if auto_corners is not None:
                src_corners = auto_corners
                H = compute_homography(src_corners)
                print("✓ Auto-detected board corners")
        
        # Process frame if we have homography
        if H is not None:
            # WARP FIRST, then detect on warped image
            warped = warp_board(frame, H)
            
            # Detect pieces on warped board (better for model)
            detections = detect_pieces(model, warped, args.conf)
            
            # Map to squares (detections already in warped space)
            square_map = assign_detections_to_warped(
                detections, 
                use_bottom=args.use_bottom,
                nms_threshold=args.nms
            )
            
            # Generate FEN
            fen = square_map_to_fen(square_map)
            
            # Print FEN if changed
            if fen != last_fen:
                print(f"\nFEN: {fen}")
                print(f"Pieces detected: {len(square_map)}")
                # Print piece positions for debugging
                if square_map:
                    squares_sorted = sorted(square_map.keys())
                    print(f"Squares: {', '.join(squares_sorted)}")
                last_fen = fen
            
            # Draw overlay on original frame
            disp = draw_board_overlay_warped(frame, warped, square_map, H)
            
            # Draw detections on warped view for debugging
            warped_display = warped.copy()
            for square, det in square_map.items():
                # Draw bbox
                x1, y1, x2, y2 = det['bbox']
                color = (0, 255, 0) if 'white' in det['class'] else (255, 0, 0)
                cv2.rectangle(warped_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw center point used for square assignment
                if args.use_bottom:
                    cx = (x1 + x2) / 2
                    cy = y2  # Bottom
                else:
                    cx, cy = det['center']
                
                cv2.circle(warped_display, (int(cx), int(cy)), 5, color, -1)
                cv2.putText(warped_display, square, (int(cx) + 10, int(cy)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw grid on warped view
            for i in range(9):
                # Vertical lines
                x = i * SQUARE_SIZE
                cv2.line(warped_display, (x, 0), (x, BOARD_SIZE), (0, 255, 255), 1)
                # Horizontal lines  
                y = i * SQUARE_SIZE
                cv2.line(warped_display, (0, y), (BOARD_SIZE, y), (0, 255, 255), 1)
            
            cv2.putText(warped_display, f"Pieces: {len(square_map)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Warped Board", warped_display)
            
            # Compare with target if provided
            if args.target_fen:
                diffs = compare_fen(fen, args.target_fen)
                if not diffs:
                    cv2.putText(disp, "CORRECT!", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                else:
                    cv2.putText(disp, f"{len(diffs)} errors", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    print(f"\nMismatches ({len(diffs)}):")
                    for d in diffs[:10]:  # Show first 10
                        print(f"  {d['square']}: got '{d['detected']}' expected '{d['expected']}'")
        
        else:
            cv2.putText(disp, "Press 'a' for auto or 'm' for manual", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw corner markers if we have them
        if src_corners is not None:
            labels = ['h8', 'a8', 'a1', 'h1']
            for i, pt in enumerate(src_corners):
                cv2.circle(disp, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
                cv2.putText(disp, labels[i], (int(pt[0]) + 10, int(pt[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Chess Board Detection", disp)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord('a'):
            auto_corners = detect_board_corners_auto(frame)
            if auto_corners is not None:
                src_corners = auto_corners
                H = compute_homography(src_corners)
                print("✓ Auto-detected board corners")
            else:
                print("✗ Auto-detection failed. Try manual (press 'm')")
        elif key == ord('m'):
            manual_corners = get_manual_corners(frame)
            if manual_corners is not None:
                src_corners = manual_corners
                H = compute_homography(src_corners)
                print("✓ Manual corners set")
            else:
                print("✗ Manual selection canceled")
        elif key == ord('r'):
            H = None
            src_corners = None
            last_fen = None
            print("Reset homography")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if last_fen:
        print(f"\nFinal FEN: {last_fen}")


if __name__ == "__main__":
    main()