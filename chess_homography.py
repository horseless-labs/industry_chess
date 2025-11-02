#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# --------------------------------------------
# CONFIG
# --------------------------------------------
BOARD_SIZE = 800  # warped board will be 800x800
SQUARE_SIZE = BOARD_SIZE // 8

# Map your model's class names -> FEN piece letters
# Adjust to match your trained YOLO classes.
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
def list_cameras(max_index=6):
    cams = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            cams.append(i)
            cap.release()
    return cams


def open_source(source):
    """
    source can be:
    - int or str number => webcam index
    - path to video file
    """
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
        # try int first
        try:
            idx = int(source)
            cap = cv2.VideoCapture(idx)
            return cap
        except ValueError:
            # assume file
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
    Try to find the chessboard polygon automatically.
    Basic strategy: edge -> contour -> biggest 4-point contour.
    This is intentionally simple; you can upgrade to line-based later.
    Returns np.array shape (4,2) or None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            # order them
            return order_points(pts)
    return None

# --------------------------------------------
# MANUAL CORNER SELECTION
# --------------------------------------------
manual_points = []
def mouse_callback(event, x, y, flags, param):
    global manual_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(manual_points) < 4:
            manual_points.append((x, y))
            print(f"Point {len(manual_points)}: {x}, {y}")

def get_manual_corners(frame):
    """
    Lets the user click 4 points.
    Returns np.array(4,2) float32 ordered TL,TR,BR,BL
    """
    global manual_points
    manual_points = []
    temp = frame.copy()
    cv2.namedWindow("Select 4 corners (TL,TR,BR,BL)")
    cv2.setMouseCallback("Select 4 corners (TL,TR,BR,BL)", mouse_callback)

    while True:
        disp = temp.copy()
        for i, p in enumerate(manual_points):
            cv2.circle(disp, p, 5, (0, 0, 255), -1)
            cv2.putText(disp, str(i+1), (p[0]+5, p[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("Select 4 corners (TL,TR,BR,BL)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if len(manual_points) == 4:
            break
    cv2.destroyWindow("Select 4 corners (TL,TR,BR,BL)")

    if len(manual_points) != 4:
        return None

    pts = np.array(manual_points, dtype=np.float32)
    return order_points(pts)

# --------------------------------------------
# HOMOGRAPHY / WARP
# --------------------------------------------
def order_points(pts):
    """
    Order a set of 4 points: TL, TR, BR, BL.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL

    return rect


def compute_homography(src_pts, board_size=BOARD_SIZE):
    dst_pts = np.array([
        [0, 0],
        [board_size - 1, 0],
        [board_size - 1, board_size - 1],
        [0, board_size - 1]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H


def warp_board(frame, H, board_size=BOARD_SIZE):
    warped = cv2.warpPerspective(frame, H, (board_size, board_size))
    return warped

# --------------------------------------------
# YOLO DETECTION (ULTRALYTICS STYLE)
# --------------------------------------------
def load_yolo_model(model_path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("You need 'ultralytics' installed: pip install ultralytics")
        sys.exit(1)
    model = YOLO(model_path)
    return model


def detect_pieces_on_warped(model, warped_img):
    """
    warped_img: 800x800 BGR
    returns list of detections:
    [
      {
        'cls_name': str,
        'conf': float,
        'bbox': [x1,y1,x2,y2]
      },
      ...
    ]
    """
    results = model.predict(source=warped_img, verbose=False)
    out = []
    if not results:
        return out
    r = results[0]

    if r.boxes is None:
        return out

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_name = r.names[cls_id]
        out.append({
            "cls_name": cls_name,
            "conf": conf,
            "bbox": [x1, y1, x2, y2]
        })
    return out

# --------------------------------------------
# MAPPING DETECTIONS -> SQUARES
# --------------------------------------------
def point_to_square(x, y, board_size=BOARD_SIZE):
    """
    x,y are in warped space.
    Returns (file_idx, rank_idx) where
    file_idx: 0..7 -> a..h
    rank_idx: 0..7 -> 8..1  (0 is top, which is rank 8)
    """
    if x < 0 or x >= board_size or y < 0 or y >= board_size:
        return None
    file_idx = 7 - int(x // SQUARE_SIZE)   # mirror horizontally
    rank_idx = int(y // SQUARE_SIZE)
    # file_idx: 0->h, 7->a
    # rank_idx: 0->8, 7->1
    return file_idx, rank_idx


def detections_to_board(detections, board_size=BOARD_SIZE):
    """
    Build 8x8 array, [rank][file], where rank 0 is 8th rank.
    If multiple detections land on the same square, keep the one with higher confidence.
    """
    board = [[None for _ in range(8)] for _ in range(8)]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        sq = point_to_square(cx, cy, board_size)
        if sq is None:
            continue
        file_idx, rank_idx = sq
        current = board[rank_idx][file_idx]
        if current is None or det["conf"] > current["conf"]:
            board[rank_idx][file_idx] = det
    return board

# --------------------------------------------
# BOARD -> FEN
# --------------------------------------------
def board_to_fen(board):
    """
    board: 8x8, board[rank][file], rank 0 is 8th rank
    Returns FEN placement only, e.g. "rnbqkbnr/pppppppp/8/...".
    We'll default to "w KQkq - 0 1" for the rest,
    but you can override.
    """
    rows = []
    for rank in range(8):
        empty = 0
        row_fen = ""
        for file_idx in range(8):
            cell = board[rank][file_idx]
            if cell is None:
                empty += 1
            else:
                if empty > 0:
                    row_fen += str(empty)
                    empty = 0
                cls_name = cell["cls_name"]
                piece = CLASS_TO_FEN.get(cls_name, None)
                if piece is None:
                    # unknown class, treat as empty for now
                    empty += 1
                else:
                    row_fen += piece
        if empty > 0:
            row_fen += str(empty)
        rows.append(row_fen)
    placement = "/".join(rows)
    # return full FEN
    fen = placement + " w - - 0 1"
    return fen

# --------------------------------------------
# FEN COMPARISON
# --------------------------------------------
def compare_fen(pred_fen, target_fen):
    """
    Very simple comparison: compare placement fields.
    Returns dict with mismatches.
    """
    pred_place = pred_fen.split()[0]
    target_place = target_fen.split()[0]

    pred_rows = pred_place.split("/")
    targ_rows = target_fen.split()[0].split("/")

    diffs = []
    for r in range(8):
        # expand rows to 8 squares
        pr = expand_fen_row(pred_rows[r])
        tr = expand_fen_row(targ_rows[r])
        for f in range(8):
            if pr[f] != tr[f]:
                rank_name = 8 - r
                file_name = chr(ord('a') + f)
                diffs.append({
                    "square": f"{file_name}{rank_name}",
                    "pred": pr[f],
                    "target": tr[f]
                })
    return diffs

def expand_fen_row(row):
    """
    'rnbqkbnr' -> ['r','n','b','q','k','b','n','r']
    '3p4' -> ['','','','p','','','','']
    we return list of length 8, empty is ''
    """
    out = []
    for ch in row:
        if ch.isdigit():
            for _ in range(int(ch)):
                out.append('')
        else:
            out.append(ch)
    # pad if needed
    while len(out) < 8:
        out.append('')
    return out[:8]

# --------------------------------------------
# MAIN LOOP
# --------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Chess homography + FEN checker")
    parser.add_argument("--source", type=str, default=None,
                        help="camera index or video file. If omitted, will list cameras.")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="Path to YOLO model (Ultralytics)")
    parser.add_argument("--target-fen", type=str, default=None,
                        help="Target FEN to compare against")
    parser.add_argument("--no-auto", action="store_true",
                        help="Skip auto detection and force manual")
    args = parser.parse_args()

    cap = open_source(args.source)
    model = load_yolo_model(args.model)

    H = None
    src_corners = None

    print("Controls:")
    print("  a = try auto-detect board")
    print("  m = manual corner select")
    print("  q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames.")
            break

        disp = frame.copy()

        # try auto once if we don't have homography and auto is allowed
        if H is None and not args.no_auto:
            auto = detect_board_corners_auto(frame)
            if auto is not None:
                src_corners = auto
                H = compute_homography(src_corners)
                print("Auto board detection successful.")

        # draw corners if we have them
        if src_corners is not None:
            for p in src_corners:
                cv2.circle(disp, (int(p[0]), int(p[1])), 8, (0,255,0), -1)

        cv2.imshow("Input", disp)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            auto = detect_board_corners_auto(frame)
            if auto is not None:
                src_corners = auto
                H = compute_homography(src_corners)
                print("Auto board detection successful.")
            else:
                print("Auto board detection failed. Try manual (press m).")
        elif key == ord('m'):
            manual = get_manual_corners(frame)
            if manual is not None:
                src_corners = manual
                H = compute_homography(src_corners)
                print("Manual board corners set.")
            else:
                print("Manual selection failed / canceled.")

        # if we have homography, do the full pipeline on current frame
        if H is not None:
            warped = warp_board(frame, H)
            detections = detect_pieces_on_warped(model, warped)
            board = detections_to_board(detections)

            fen = board_to_fen(board)
            cv2.putText(warped, fen, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Warped board", warped)

            if args.target_fen:
                diffs = compare_fen(fen, args.target_fen)
                if not diffs:
                    text = "Board correct"
                else:
                    text = f"{len(diffs)} mismatches"
                cv2.putText(disp, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                # print diffs to console
                if diffs:
                    print("Mismatches:")
                    for d in diffs:
                        print(f"  {d['square']}: got '{d['pred']}' expected '{d['target']}'")
                else:
                    print("Board matches target.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
