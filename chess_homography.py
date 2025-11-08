#!/usr/bin/env python3
"""
Chess board homography + FEN detection and validation.

Usage:
    # Webcam with auto-selection
    python chess_homography.py --model best.pt
    
    # Specific camera
    python chess_homography.py --model best.pt --source 0
    
    # Video file
    python chess_homography.py --model best.pt --source game.mp4
    
    # With target FEN for validation
    python chess_homography.py --model best.pt --target-fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

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

# Try to import chess libraries
# TODO: figure out C++ port situation
try:
    import chess
    import chess.pgn
    import chess.svg
    from cairosvg import svg2png
    HAS_CHESS_RENDERING = True
except ImportError:
    try:
        import chess
        import chess.pgn
        HAS_CHESS_RENDERING = False
        print("Note: cairosvg not available. Using fallback board rendering.")
    except ImportError:
        HAS_CHESS_RENDERING = False
        print("Note: python-chess not available. Board visualization will be limited.")

# --------------------------------------------
# CONFIG
# --------------------------------------------
BOARD_SIZE = 800  # warped board will be 800x800
SQUARE_SIZE = BOARD_SIZE // 8

# Label fixes (if your dataset has king/queen swapped)
# Problem also came up *with* label fixes
# TODO: double-check labeling in original dataset; this could be a detection
#       problem instead of a labeling one.
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
# Currently has user guess with the index
# TODO: rewrite to display data relating name of device with index
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
# Severe problems with automatic detection in testing; unsure if this is a
# result of poor lighting conditions.
# TODO: test with different lighting, other boards, etc.
# TODO: consider removal.
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
# TODO: consider making this the default option
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
    
    labels = ["h8 (top right of board)", "a8 (top left of board)", "a1 (bottom left of board)", "h1 (bottom right of board)"]
    
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

# Further reading for cv2.findHomography
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
# and projective transformations
# https://www.youtube.com/watch?v=2BIzmFD_pRQ
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
# TEMPORAL SMOOTHING
# --------------------------------------------

class TemporalSmoother:
    """
    Smooth detections over time using a sliding window.
    Helps stabilize flickering piece classifications.
    """
    
    def __init__(self, window_size=10, confidence_threshold=0.6):
        """
        Args:
            window_size: Number of frames to consider for smoothing
            confidence_threshold: Minimum agreement ratio to accept a classification
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.history = {}  # square -> list of (class_name, confidence) tuples
    
    def update(self, square_map):
        """
        Update history with new detections and return smoothed square map.
        
        Args:
            square_map: Dict of {square: detection_dict}
        
        Returns:
            Smoothed square_map with stabilized classifications
        """
        # Update history for each square
        current_squares = set(square_map.keys())
        
        # Add new detections to history
        for square, det in square_map.items():
            if square not in self.history:
                self.history[square] = []
            
            self.history[square].append({
                'class': det['class'],
                'confidence': det['confidence']
            })
            
            # Keep only last N frames
            if len(self.history[square]) > self.window_size:
                self.history[square].pop(0)
        
        # Remove history for squares that haven't been detected recently
        all_squares = set(self.history.keys())
        for square in all_squares:
            # If square not detected in current frame, add a "None" entry
            if square not in current_squares:
                self.history[square].append(None)
                if len(self.history[square]) > self.window_size:
                    self.history[square].pop(0)
                
                # Remove history if piece hasn't been seen for a while
                none_count = sum(1 for x in self.history[square] if x is None)
                if none_count > self.window_size // 2:
                    del self.history[square]
        
        # Create smoothed square map
        smoothed_map = {}
        
        for square, det in square_map.items():
            if square not in self.history or len(self.history[square]) < 3:
                # Not enough history, use current detection
                smoothed_map[square] = det
                continue
            
            # Count class occurrences in history (ignore None entries)
            class_votes = {}
            total_votes = 0
            confidence_sum = {}
            
            for entry in self.history[square]:
                if entry is not None:
                    cls = entry['class']
                    conf = entry['confidence']
                    class_votes[cls] = class_votes.get(cls, 0) + 1
                    confidence_sum[cls] = confidence_sum.get(cls, 0) + conf
                    total_votes += 1
            
            if total_votes == 0:
                smoothed_map[square] = det
                continue
            
            # Find most common class
            most_common_class = max(class_votes.items(), key=lambda x: x[1])[0]
            vote_ratio = class_votes[most_common_class] / total_votes
            
            # If confidence is high enough, use the most common class
            if vote_ratio >= self.confidence_threshold:
                # Use smoothed class but keep current bbox and other info
                smoothed_det = det.copy()
                smoothed_det['class'] = most_common_class
                smoothed_det['smoothed'] = True
                smoothed_det['vote_ratio'] = vote_ratio
                # Use average confidence for the smoothed class
                smoothed_det['confidence'] = confidence_sum[most_common_class] / class_votes[most_common_class]
                smoothed_map[square] = smoothed_det
            else:
                # Not confident enough, use current detection
                smoothed_map[square] = det
        
        return smoothed_map
    
    def reset(self):
        """Clear all history."""
        self.history = {}


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


def assign_detections_from_original(detections, H, use_bottom=True, nms_threshold=0.5):
    """
    Map detections from original frame space to board squares.
    
    Args:
        detections: List of detection dicts (in original frame coordinates)
        H: Homography matrix (frame -> warped board)
        use_bottom: If True, use bottom-center of bbox instead of center
        nms_threshold: IoU threshold for non-maximum suppression
    
    Returns dict: {square_name: detection}
    """
    # First, apply NMS to remove duplicate detections in original frame space
    detections = non_maximum_suppression(detections, nms_threshold)
    
    square_map = {}
    
    for det in detections:
        if use_bottom:
            # Use bottom-center of bounding box (where piece base is)
            # Reduces accidental detections across multiple squares from low
            # angles.
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = y2  # Bottom of box (piece base)
        else:
            cx, cy = det["center"]
        
        # Transform this point to warped board space
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        warped_pt = cv2.perspectiveTransform(pt, H)[0][0]
        
        wx, wy = warped_pt
        square = point_to_square(wx, wy)
        
        if square is None:
            continue
        
        # Store detection with warped coordinates for visualization
        det_copy = det.copy()
        det_copy["warped_center"] = (wx, wy)
        
        # If multiple pieces map to same square, keep highest confidence
        if square not in square_map or det["confidence"] > square_map[square]["confidence"]:
            square_map[square] = det_copy
    
    return square_map


def assign_detections_to_warped(detections, use_bottom=True, nms_threshold=0.5):
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
            # Reduces accidental detections across multiple squares from low
            # angles.
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

"""
Read that Soft-NMS variants are better for handling very crowded scenes
# TODO: search for Soft-NMS
# TODO: run tests to determine whether those are more performant, or if density
        is still too low to be relevant here.
"""
def non_maximum_suppression(detections, iou_threshold=0.5): # IoU is intersection/union
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
# BOARD VISUALIZATION
# --------------------------------------------

def render_board_from_fen(fen_string, size=400):
    """
    Render a chess board from FEN notation as an image.
    Returns numpy array (BGR format for OpenCV) or None if rendering unavailable.
    
    Args:
        fen_string: FEN notation string
        size: Size of the board in pixels (square)
    """
    if not HAS_CHESS_RENDERING:
        return None
    
    try:
        # Parse FEN
        board = chess.Board(fen_string)
        
        # Generate SVG
        svg_data = chess.svg.board(
            board,
            size=size,
            coordinates=True,
            colors={
                "square light": "#f0d9b5",
                "square dark": "#b58863",
                "square dark lastmove": "#aaa23a",
                "square light lastmove": "#cdd26a"
            }
        )
        
        # Convert SVG to PNG
        png_data = svg2png(bytestring=svg_data.encode('utf-8'))
        
        # Convert to numpy array
        nparr = np.frombuffer(png_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    
    except Exception as e:
        print(f"Error rendering board: {e}")
        return None


def create_fallback_board_image(fen_string, size=400):
    """
    Create a simple text-based board visualization as fallback.
    
    Args:
        fen_string: FEN notation string
        size: Size of the board in pixels
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    # Parse FEN placement
    try:
        board = chess.Board(fen_string)
        square_size = size // 8
        
        # Unicode chess pieces
        piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        # Draw checkerboard
        for rank in range(8):
            for file in range(8):
                x = file * square_size
                y = rank * square_size
                
                # Alternate colors
                if (rank + file) % 2 == 0:
                    color = (240, 217, 181)  # Light square
                else:
                    color = (181, 136, 99)   # Dark square
                
                cv2.rectangle(img, (x, y), (x + square_size, y + square_size), color, -1)
        
        # Draw pieces
        font = cv2.FONT_HERSHEY_SIMPLEX
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)  # Flip rank for display
                piece = board.piece_at(square)
                
                if piece:
                    symbol = piece_symbols.get(piece.symbol(), piece.symbol())
                    x = file * square_size + square_size // 4
                    y = rank * square_size + square_size * 3 // 4
                    
                    # Choose color based on piece color
                    text_color = (50, 50, 50) if piece.color == chess.WHITE else (10, 10, 10)
                    
                    cv2.putText(img, symbol, (x, y), font, 1.2, text_color, 2)
        
        # Add file/rank labels
        font_scale = 0.4
        for i in range(8):
            # Files (a-h)
            file_label = chr(ord('a') + i)
            cv2.putText(img, file_label, (i * square_size + square_size // 2 - 5, size - 5),
                       font, font_scale, (100, 100, 100), 1)
            
            # Ranks (8-1)
            rank_label = str(8 - i)
            cv2.putText(img, rank_label, (5, i * square_size + square_size // 2 + 5),
                       font, font_scale, (100, 100, 100), 1)
        
        return img
    
    except Exception as e:
        # Ultimate fallback - just show FEN text
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Board Display Error", (20, size // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, fen_string.split()[0], (10, size // 2 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        return img


def get_board_visualization(fen_string, size=400):
    """
    Get board visualization from FEN, with fallback options.
    
    Returns:
        numpy array (BGR image) of the board
    """
    # Try high-quality SVG rendering first
    if HAS_CHESS_RENDERING:
        board_img = render_board_from_fen(fen_string, size)
        if board_img is not None:
            return board_img
    
    # Fallback to simple rendering
    return create_fallback_board_image(fen_string, size)


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
# Draws grids on their respective boards; currently not in use
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
    parser.add_argument("--smooth-window", type=int, default=10,
                       help="Number of frames for temporal smoothing (0 to disable)")
    parser.add_argument("--smooth-threshold", type=float, default=0.6,
                       help="Confidence threshold for temporal smoothing (0.5-1.0)")
    parser.add_argument("--save-video", action="store_true",
                       help="Save output video with overlays")
    parser.add_argument("--output-dir", type=Path, default=Path("output_videos"),
                       help="Directory to save output videos")
    parser.add_argument("--no-auto", action="store_true",
                       help="Skip automatic board detection")
    
    args = parser.parse_args()
    
    # Load model and source
    print("Loading model...")
    model = load_yolo_model(args.model)
    print("Opening video source...")
    cap = open_source(args.source)
    
    # Setup video writer if saving
    writer = None
    if args.save_video:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Generate output filename
        if args.source:
            try:
                source_name = Path(args.source).stem
            except:
                source_name = f"camera_{args.source}"
        else:
            source_name = "camera"
        
        output_path = args.output_dir / f"{source_name}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Will save video to: {output_path}")
    
    H = None
    src_corners = None
    paused = False
    last_fen = None
    
    # Initialize temporal smoother
    smoother = None
    if args.smooth_window > 0:
        smoother = TemporalSmoother(
            window_size=args.smooth_window,
            confidence_threshold=args.smooth_threshold
        )
        print(f"Temporal smoothing enabled: window={args.smooth_window}, threshold={args.smooth_threshold}")
    
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  a = auto-detect board corners")
    print("  m = manual corner selection")
    print("  p = pause/unpause")
    print("  r = reset homography")
    print("  q = quit")
    if not HAS_CHESS_RENDERING:
        print("\nNote: Install python-chess and cairosvg for better board rendering:")
        print("  pip install python-chess cairosvg")
    print("="*60 + "\n")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video/stream")
                break
        
        disp = frame.copy()
        
        # Try auto-detection once at start (if allowed)
        # TODO: disable until board can more reliably be detected
        # TODO: find better board detection method
        if H is None and not args.no_auto and src_corners is None:
            auto_corners = detect_board_corners_auto(frame)
            if auto_corners is not None:
                src_corners = auto_corners
                H = compute_homography(src_corners)
                print("✓ Auto-detected board corners")
        
        # Process frame if we have homography
        if H is not None:
            # Detect on ORIGINAL frame (better - avoids warping artifacts)
            detections = detect_pieces(model, frame, args.conf)
            
            # Map to squares using homography transformation
            square_map = assign_detections_from_original(
                detections, 
                H,
                use_bottom=args.use_bottom,
                nms_threshold=args.nms
            )
                        
            # Apply temporal smoothing if enabled
            if smoother is not None:
                square_map = smoother.update(square_map)
            
            # Generate warped view for visualization
            # Might remove later; not as good as board_viz for this purpose
            # warped = warp_board(frame, H)
            
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
            
            # Create board visualization window
            board_viz = get_board_visualization(fen, size=500)
            
            # Add FEN text below board
            viz_height = board_viz.shape[0]
            viz_width = board_viz.shape[1]
            
            # Create extended image with space for text
            extended_viz = np.ones((viz_height + 100, viz_width, 3), dtype=np.uint8) * 255
            extended_viz[0:viz_height, :] = board_viz
            
            # Add title
            # TODO: change fonts to be more readable
            cv2.putText(extended_viz, "Detected Board State", (10, viz_height + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add piece count
            cv2.putText(extended_viz, f"Pieces: {len(square_map)}", (10, viz_height + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)
            
            # Add FEN notation (truncated if too long)
            fen_display = fen if len(fen) < 60 else fen[:57] + "..."
            cv2.putText(extended_viz, f"FEN: {fen_display}", (10, viz_height + 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            cv2.imshow("Board Visualization", extended_viz)
            
            # Draw overlay on original frame with detections
            disp = frame.copy()
            
            # Draw board grid
            H_inv = np.linalg.inv(H)
            corners = np.array([
                [0, 0],
                [BOARD_SIZE, 0],
                [BOARD_SIZE, BOARD_SIZE],
                [0, BOARD_SIZE]
            ], dtype=np.float32).reshape(-1, 1, 2)
            frame_corners = cv2.perspectiveTransform(corners, H_inv).astype(np.int32)
            cv2.polylines(disp, [frame_corners], True, (0, 255, 0), 3)
            
            # Draw detections and labels on original frame
            for square, det in square_map.items():
                x1, y1, x2, y2 = det['bbox']
                # Keep color based on piece color (don't change for smoothing)
                color = (0, 255, 0) if 'white' in det['class'] else (255, 0, 0)
                
                # Draw bounding box
                cv2.rectangle(disp, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw point used for assignment
                if args.use_bottom:
                    cx = (x1 + x2) / 2
                    cy = y2
                else:
                    cx, cy = det['center']
                
                cv2.circle(disp, (int(cx), int(cy)), 5, color, -1)
                
                # Draw label with piece name and square
                piece_name = det['class'].replace('_', ' ').title()
                label = f"{piece_name} @ {square}"
                conf_pct = int(det['confidence'] * 100)
                
                # Add smoothing indicator to label only (not color)
                if det.get('smoothed', False):
                    vote_pct = int(det.get('vote_ratio', 0) * 100)
                    full_label = f"{label} ({conf_pct}% | S:{vote_pct}%)"
                else:
                    full_label = f"{label} ({conf_pct}%)"
                
                # Create background for text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(full_label, font, font_scale, thickness)
                
                # Position label above bbox
                label_y = int(y1) - 10
                if label_y < text_h + 10:
                    label_y = int(y2) + text_h + 10  # Put below if too close to top
                
                # Draw background rectangle
                cv2.rectangle(disp, 
                             (int(x1), label_y - text_h - 5),
                             (int(x1) + text_w + 10, label_y + 5),
                             color, -1)
                
                # Draw text
                cv2.putText(disp, full_label, (int(x1) + 5, label_y),
                           font, font_scale, (255, 255, 255), thickness)
                
            # Draws the grid on top fo the chess board.
            # disp = draw_board_overlay(frame, square_map, H)
            
            # Draw warped view with square assignments for debugging
            # Might remove later; board_viz is a general improvement over warped
            # warped_display = warped.copy()
            
            # # Draw grid on warped view
            # for i in range(9):
            #     # Vertical lines
            #     x = i * SQUARE_SIZE
            #     cv2.line(warped_display, (x, 0), (x, BOARD_SIZE), (0, 255, 255), 1)
            #     # Horizontal lines  
            #     y = i * SQUARE_SIZE
            #     cv2.line(warped_display, (0, y), (BOARD_SIZE, y), (0, 255, 255), 1)
            
            # # Draw square labels at transformed positions
            # for square, det in square_map.items():
            #     if "warped_center" in det:
            #         wx, wy = det["warped_center"]
            #         color = (0, 255, 0) if 'white' in det['class'] else (255, 0, 0)
            #         cv2.circle(warped_display, (int(wx), int(wy)), 5, color, -1)
            #         cv2.putText(warped_display, square, (int(wx) + 10, int(wy)),
            #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # cv2.putText(warped_display, f"Pieces: {len(square_map)}", (10, 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # cv2.imshow("Warped Board", warped_display)
            
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
        
        # Save frame to video if enabled
        if writer is not None:
            writer.write(disp)
        
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
            if smoother is not None:
                smoother.reset()
            print("Reset homography and smoothing history")
    
    cap.release()
    if writer is not None:
        writer.release()
        print(f"\n✓ Video saved to: {output_path}")
    cv2.destroyAllWindows()
    
    if last_fen:
        print(f"\nFinal FEN: {last_fen}")


if __name__ == "__main__":
    main()