# chessboard_rectify_noprinter_menu.py
# Requires: pip install opencv-python numpy

import cv2 as cv
import numpy as np
from collections import deque
import os
import sys
import glob
import platform

# ---------- Config ----------
CANONICAL_SIZE = 512               # rectified board size (pixels)
BOARD_INNER = (7, 7)               # 7x7 inner corners for an 8x8 chessboard
EDGE_TH = (60, 180)                # Canny thresholds for Hough fallback
MIN_LINE_LEN = 120                 # HoughP min line length
MAX_LINE_GAP = 10                  # HoughP max gap
CONF_HISTORY = 10                  # rolling quality history
UPDATE_MIN_CONF = 0.35             # don't update H if confidence lower than this
SHOW = True

# How far to scan numeric camera indices if OS-specific paths aren't found
SCAN_MAX_INDEX = 10

# ---------- Utilities ----------
def order_quad(pts):
    """Order 4 points as [tl, tr, br, bl]."""
    pts = np.array(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    idx = np.argsort(angles)
    pts = pts[idx]
    # Now enforce tl as smallest x+y
    s = pts.sum(axis=1)
    tl = np.argmin(s)
    pts = np.roll(pts, -tl, axis=0)
    return pts

def homography_from_pts4(pts4, size=CANONICAL_SIZE):
    dst = np.float32([[0,0],[size-1,0],[size-1,size-1],[0,size-1]])
    return cv.getPerspectiveTransform(pts4, dst)

def square_bbox(file_idx, rank_idx, size=CANONICAL_SIZE):
    tile = size // 8
    x0 = file_idx * tile
    y0 = (7 - rank_idx) * tile  # rank 1 at bottom
    return x0, y0, tile, tile

def average_luma(img):
    if len(img.shape)==3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return float(img.mean())

def orientation_fix(rectified):
    """Rotate 180° if a1 isn't dark (relative to h1)."""
    tile = CANONICAL_SIZE // 8
    pad = max(2, tile//10)
    a1 = rectified[(7*tile)+pad:(8*tile)-pad, 0+pad:tile-pad]
    h1 = rectified[(7*tile)+pad:(8*tile)-pad, (7*tile)+pad:(8*tile)-pad]
    if average_luma(a1) > average_luma(h1):  # a1 should be darker than h1
        return cv.rotate(rectified, cv.ROTATE_180)
    return rectified

# ---------- Detection: Grid-first ----------
def detect_grid_homography(frame):
    """Try to detect the 7x7 inner-corner grid to compute a robust H."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    flags = cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY
    ret, corners = cv.findChessboardCornersSB(gray, BOARD_INNER, flags=flags)
    if not ret:
        return None, 0.0

    img_pts = corners.reshape(-1, 2).astype(np.float32)

    xs = np.linspace((0.5/8)*CANONICAL_SIZE, (7.5/8)*CANONICAL_SIZE, BOARD_INNER[0])
    ys = np.linspace((0.5/8)*CANONICAL_SIZE, (7.5/8)*CANONICAL_SIZE, BOARD_INNER[1])
    XX, YY = np.meshgrid(xs, ys)
    canon_pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)

    H, mask = cv.findHomography(img_pts, canon_pts, cv.RANSAC, 3.0)
    if H is None:
        return None, 0.0
    inlier_ratio = float(mask.sum()) / len(mask)
    return H, inlier_ratio

# ---------- Detection: 4-corner fallback via Hough ----------
def detect_outer_corners_hough(frame):
    """Detect outer 4 board corners from dominant orthogonal lines."""
    work = cv.GaussianBlur(frame, (5,5), 0)
    gray = cv.cvtColor(work, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, EDGE_TH[0], EDGE_TH[1])

    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                           minLineLength=MIN_LINE_LEN, maxLineGap=MAX_LINE_GAP)
    if lines is None:
        return None, 0.0

    horiz, vert = [], []
    for l in lines[:,0,:]:
        x1,y1,x2,y2 = l
        dx, dy = x2-x1, y2-y1
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) < 20 or abs(angle) > 160:
            horiz.append(l)
        elif 70 < abs(angle) < 110:
            vert.append(l)

    if len(horiz) < 2 or len(vert) < 2:
        return None, 0.0

    horiz = np.array(horiz)
    vert  = np.array(vert)

    def midpoints(lines):
        return (lines[:,0:2] + lines[:,2:4]) / 2.0

    hm = midpoints(horiz); vm = midpoints(vert)
    top_idx = np.argmin(hm[:,1]); bottom_idx = np.argmax(hm[:,1])
    left_idx = np.argmin(vm[:,0]); right_idx  = np.argmax(vm[:,0])

    top_line = horiz[top_idx]
    bot_line = horiz[bottom_idx]
    left_line = vert[left_idx]
    right_line = vert[right_idx]

    def line_params(l):
        x1,y1,x2,y2 = l
        A = y2 - y1
        B = x1 - x2
        C = A*x1 + B*y1
        return A, B, C

    def intersect(l1, l2):
        A1,B1,C1 = line_params(l1)
        A2,B2,C2 = line_params(l2)
        det = A1*B2 - A2*B1
        if abs(det) < 1e-6: return None
        x = (B2*C1 - B1*C2) / det
        y = (A1*C2 - A2*C1) / det
        return np.array([x,y], dtype=np.float32)

    pts = [
        intersect(top_line, left_line),
        intersect(top_line, right_line),
        intersect(bot_line, right_line),
        intersect(bot_line, left_line),
    ]
    if any(p is None for p in pts):
        return None, 0.0

    quad = order_quad(pts)
    h, w = frame.shape[:2]
    if np.any(quad[:,0] < -5) or np.any(quad[:,0] > w+5) or np.any(quad[:,1] < -5) or np.any(quad[:,1] > h+5):
        return None, 0.0

    def poly_area(q):
        return 0.5*abs(np.dot(q[:,0], np.roll(q[:,1], -1)) - np.dot(q[:,1], np.roll(q[:,0], -1)))
    area = poly_area(quad)
    area_norm = area / (w*h + 1e-6)

    v1 = quad[1] - quad[0]
    v2 = quad[3] - quad[0]
    cosang = abs(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6))
    ortho = 1 - cosang

    conf = 0.5*area_norm + 0.5*max(0.0, ortho)
    return quad, conf

# ---------- Main rectification step ----------
class HomographyKeeper:
    def __init__(self):
        self.H = None
        self.conf_hist = deque(maxlen=CONF_HISTORY)

    def update(self, H, conf):
        if H is None: return False
        self.conf_hist.append(conf)
        avg = sum(self.conf_hist)/len(self.conf_hist)
        if conf >= UPDATE_MIN_CONF or (self.H is None and conf > 0):
            self.H = H
            return True
        return False

def rectify_frame(frame, keeper):
    H, inlier_ratio = detect_grid_homography(frame)
    conf_grid = float(inlier_ratio)

    quad, conf_hough = (None, 0.0)
    if H is None or conf_grid < 0.5:
        res = detect_outer_corners_hough(frame)
        if res[0] is not None:
            quad, conf_hough = res
            H = homography_from_pts4(quad)

    conf = max(conf_grid, conf_hough)
    keeper.update(H, conf)

    if keeper.H is None:
        return None, conf, (conf_grid, conf_hough), None

    rectified = cv.warpPerspective(frame, keeper.H, (CANONICAL_SIZE, CANONICAL_SIZE))
    rectified = orientation_fix(rectified)
    return rectified, conf, (conf_grid, conf_hough), quad

# ---------- NEW: Camera selection helpers ----------
def _probe_source(open_arg, api_pref=None):
    """Try to open a source (index or path). Returns (cap, (w,h)) or (None, None)."""
    cap = cv.VideoCapture(open_arg, api_pref) if api_pref else cv.VideoCapture(open_arg)
    if not cap.isOpened():
        cap.release()
        return None, None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None, None
    h, w = frame.shape[:2]
    return cap, (w, h)

def list_cameras():
    """Enumerate likely camera sources and return a list of available devices."""
    cams = []
    key_counter = 0

    # On Linux, prefer explicit device paths for stability
    if platform.system() == 'Linux':
        for path in sorted(glob.glob('/dev/video*')):
            cap, res = _probe_source(path, api_pref=cv.CAP_V4L2)
            if cap is not None:
                cap.release()
                key = str(key_counter)
                key_counter += 1
                label = f"{path} (V4L2, {res[0]}x{res[1]})"
                cams.append({'key': key, 'label': label, 'open_arg': path, 'api': cv.CAP_V4L2})

    # Fallback for other OSes or if V4L2 fails: numeric indices
    for idx in range(SCAN_MAX_INDEX):
        cap, res = _probe_source(idx)
        if cap is not None:
            cap.release()
            key = str(key_counter)
            key_counter += 1
            label = f"Device Index {idx} ({res[0]}x{res[1]})"
            cams.append({'key': key, 'label': label, 'open_arg': idx, 'api': None})

    return cams

def select_camera_interactive():
    """Print a menu, let the user select a camera, and return the opened device."""
    while True:
        cams = list_cameras()
        print("\n=== Select a Camera ===")
        if not cams:
            print("No cameras found. Connect a camera and press 'r' to rescan, or 'q' to quit.")
        else:
            for c in cams:
                print(f"[{c['key']}] {c['label']}")
        print("\n[r] Refresh List")
        print("[q] Quit")
        choice = input("Enter selection: ").strip().lower()

        if choice == 'q':
            return None, None
        if choice == 'r':
            continue

        match = next((c for c in cams if c['key'] == choice), None)
        if not match:
            print("Invalid selection.")
            continue

        api = match['api']
        open_arg = match['open_arg']
        cap = cv.VideoCapture(open_arg, api) if api else cv.VideoCapture(open_arg)
        
        if not cap.isOpened():
            print("Error: Failed to open the selected camera.")
            continue
        
        print(f"Successfully opened camera: {match['label']}")
        return cap, match['label']

# ---------- MODIFIED: Demo / glue ----------
def draw_debug(frame, rectified, confs, quad):
    vis = frame.copy()
    conf_grid, conf_hough = confs
    if quad is not None:
        q = quad.astype(int)
        cv.polylines(vis, [q], isClosed=True, color=(0,255,0), thickness=2)
    cv.putText(vis, f"grid:{conf_grid:.2f}  hough:{conf_hough:.2f}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv.LINE_AA)
    if rectified is not None:
        tile = CANONICAL_SIZE // 8
        for i in range(9):
            cv.line(rectified, (0, i*tile), (CANONICAL_SIZE, i*tile), (128,128,128), 1)
            cv.line(rectified, (i*tile, 0), (i*tile, CANONICAL_SIZE), (128,128,128), 1)
    return vis, rectified

def run_loop(cap, cam_label):
    """Main video processing loop."""
    keeper = HomographyKeeper()
    win_name = "Board Rectification"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        rectified, conf, confs, quad = rectify_frame(frame, keeper)
        if SHOW:
            vis, rect = draw_debug(frame, None if rectified is None else rectified.copy(), confs, quad)
            if rect is not None:
                side = CANONICAL_SIZE
                h = max(vis.shape[0], side)
                canvas = np.zeros((h, vis.shape[1]+side, 3), dtype=np.uint8)
                canvas[:vis.shape[0], :vis.shape[1]] = vis
                canvas[:side, vis.shape[1]:vis.shape[1]+side] = rect
                cv.putText(canvas, cam_label, (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
                cv.imshow(win_name, canvas)
            else:
                cv.putText(vis, cam_label, (10, vis.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
                cv.imshow(win_name, vis)

        key = cv.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

def main():
    """Main function to handle camera selection and processing loop."""
    cap, label = select_camera_interactive()

    if cap is None:
        print("No camera selected. Exiting.")
        return

    run_loop(cap, label)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()