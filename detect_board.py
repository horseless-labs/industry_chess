# chessboard_rectify_noprinter_menu.py
# Requires: pip install opencv-python numpy

import cv2 as cv
import numpy as np
from collections import deque
import os
import sys
import glob
import platform
import time

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

# ---------- Detection: Grid-first (CLAHE) ----------
def detect_grid_homography(frame):
    """Try to detect the 7x7 inner-corner grid to compute a robust H."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
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

# ---------- Tracking (KLT) ----------
class BoardTracker:
    def __init__(self):
        self.img_pts = None    # last-tracked 2D points (Nx2)
        self.canon_pts = None  # matching canonical 2D points (Nx2)
        self.prev_gray = None
        self.failed_frames = 0

    def seed_from_H(self, H, size=CANONICAL_SIZE, inner=BOARD_INNER):
        xs = np.linspace((0.5/8)*size, (7.5/8)*size, inner[0], dtype=np.float32)
        ys = np.linspace((0.5/8)*size, (7.5/8)*size, inner[1], dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys)
        canon = np.stack([XX.ravel(), YY.ravel()], axis=1)

        Hinv = np.linalg.inv(H)
        pts3 = np.concatenate([canon, np.ones((canon.shape[0],1), np.float32)], axis=1)
        proj = (Hinv @ pts3.T).T
        proj = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)

        self.img_pts = proj
        self.canon_pts = canon.astype(np.float32)
        self.prev_gray = None
        self.failed_frames = 0

    def track_and_refit(self, frame_bgr):
        if self.img_pts is None or self.canon_pts is None:
            return None, 0.0

        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, 0.0

        next_pts, st, err = cv.calcOpticalFlowPyrLK(
            self.prev_gray, gray,
            self.img_pts.reshape(-1,1,2),
            None, winSize=(21,21), maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        if next_pts is None or st is None:
            self.failed_frames += 1
            self.prev_gray = gray
            return None, 0.0

        st = st.reshape(-1).astype(bool)
        tracked_img = next_pts.reshape(-1,2)[st]
        tracked_can = self.canon_pts[st]

        if len(tracked_img) < 12:
            self.failed_frames += 1
            self.prev_gray = gray
            return None, 0.0

        H, mask = cv.findHomography(tracked_img, tracked_can, cv.RANSAC, 3.0)
        if H is None or mask is None:
            self.failed_frames += 1
            self.prev_gray = gray
            return None, 0.0

        inlier_mask = mask.reshape(-1).astype(bool)
        inlier_ratio = float(inlier_mask.sum()) / len(inlier_mask)

        # Update stored points for next round
        self.img_pts = tracked_img[inlier_mask]
        self.canon_pts = tracked_can[inlier_mask]
        self.prev_gray = gray
        self.failed_frames = 0
        return H, inlier_ratio

def crop_to_quad(frame, quad, pad=20):
    if quad is None:
        return frame, (0,0)
    x0 = max(int(min(quad[:,0]))-pad, 0)
    y0 = max(int(min(quad[:,1]))-pad, 0)
    x1 = min(int(max(quad[:,0]))+pad, frame.shape[1])
    y1 = min(int(max(quad[:,1]))+pad, frame.shape[0])
    roi = frame[y0:y1, x0:x1]
    return roi, (x0, y0)

# ---------- Main rectification step ----------
class HomographyKeeper:
    def __init__(self):
        self.H = None
        self.conf_hist = deque(maxlen=CONF_HISTORY)
        self.tracker = BoardTracker()

    def update(self, H, conf):
        if H is None:
            return False
        self.conf_hist.append(conf)
        avg = sum(self.conf_hist)/len(self.conf_hist)
        if conf >= UPDATE_MIN_CONF or (self.H is None and conf > 0):
            self.H = H
            return True
        return False

def rectify_frame(frame, keeper):
    # 0) predictive tracking if we already have an H
    H = None
    conf_track = 0.0
    quad_est = None

    if keeper.H is not None:
        Ht, r = keeper.tracker.track_and_refit(frame)
        if Ht is not None and r >= 0.4:
            H = Ht
            conf_track = r

        # estimate quad for ROI
        canon_corners = np.float32([[0,0],[CANONICAL_SIZE-1,0],[CANONICAL_SIZE-1,CANONICAL_SIZE-1],[0,CANONICAL_SIZE-1]])
        Hinv = np.linalg.inv(keeper.H)
        pts3 = np.concatenate([canon_corners, np.ones((4,1), np.float32)], axis=1)
        proj = (Hinv @ pts3.T).T
        proj = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
        quad_est = order_quad(proj)

    # 1) detectors (possibly in ROI)
    search_frame = frame
    offset = (0, 0)
    if quad_est is not None:
        search_frame, offset = crop_to_quad(frame, quad_est, pad=40)

    conf_grid = 0.0
    conf_hough = 0.0
    quad = None

    if H is None:
        Hg, inlier_ratio = detect_grid_homography(search_frame)
        if Hg is not None:
            # bake translation (search_frame is cropped)
            T = np.array([[1,0,-offset[0]],
                          [0,1,-offset[1]],
                          [0,0,1]], dtype=np.float32)
            H = Hg @ T
            conf_grid = float(inlier_ratio)

    if (H is None) or (conf_grid < 0.5 and conf_track < 0.5):
        res = detect_outer_corners_hough(search_frame)
        if res[0] is not None:
            quad = res[0] + np.array(offset, dtype=np.float32)  # un-crop
            H = homography_from_pts4(quad)
            conf_hough = res[1]
        else:
            quad = None
    else:
        quad = quad_est

    # 2) choose confidence, update keeper
    conf = max(conf_track, conf_grid, conf_hough)
    keeper.update(H, conf)

    if keeper.H is None:
        return None, conf, (conf_grid, conf_hough), None

    # 3) refresh tracker when we have a good H
    if conf >= 0.55 and (keeper.tracker.canon_pts is None or len(keeper.tracker.canon_pts) < 20 or keeper.tracker.failed_frames > 0):
        keeper.tracker.seed_from_H(keeper.H)

    rectified = cv.warpPerspective(frame, keeper.H, (CANONICAL_SIZE, CANONICAL_SIZE))
    rectified = orientation_fix(rectified)
    return rectified, conf, (conf_grid, conf_hough), quad

# ---------- Safer camera enumeration & opening ----------
def list_cameras():
    """List likely camera nodes without stress-testing them."""
    cams = []
    key_counter = 0

    if platform.system() == 'Linux':
        for path in sorted(glob.glob('/dev/video*')):
            key = str(key_counter); key_counter += 1
            label = f"{path} (V4L2)"
            cams.append({'key': key, 'label': label, 'open_arg': path, 'api': cv.CAP_V4L2})

    # Numeric indices as a fallback
    for idx in range(min(SCAN_MAX_INDEX, 6)):
        key = str(key_counter); key_counter += 1
        label = f"Device Index {idx}"
        cams.append({'key': key, 'label': label, 'open_arg': idx, 'api': None})

    return cams

def _open_camera(open_arg, api_pref):
    cap = cv.VideoCapture(open_arg, api_pref) if api_pref else cv.VideoCapture(open_arg)
    if not cap.isOpened():
        return None
    # stabilize the stream
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    # prefer MJPG if available
    try:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    # prime a frame
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def select_camera_interactive():
    """Print a menu, let the user select a camera, and return the opened device + args."""
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
            return None, None, None
        if choice == 'r':
            continue

        match = next((c for c in cams if c['key'] == choice), None)
        if not match:
            print("Invalid selection.")
            continue

        cap = _open_camera(match['open_arg'], match['api'])
        if cap is None:
            print("Error: Failed to open the selected camera.")
            continue

        print(f"Successfully opened camera: {match['label']}")
        return cap, match['label'], (match['open_arg'], match['api'])

# ---------- Demo / glue ----------
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

def run_loop(cap, cam_label, open_arg_and_api):
    """Main video processing loop with auto-reconnect."""
    keeper = HomographyKeeper()
    win_name = "Board Rectification"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    consecutive_fail = 0
    reopen_delay = 0.5

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            consecutive_fail += 1
            print("Camera read failed.")
            if consecutive_fail >= 3 and open_arg_and_api is not None:
                print("Attempting to reopen camera...")
                cap.release()
                time.sleep(reopen_delay)
                open_arg, api = open_arg_and_api
                cap = _open_camera(open_arg, api)
                if cap is None:
                    print("Reopen failed; will retry.")
                    time.sleep(0.5)
                    if (cv.waitKey(1) & 0xFF) in (27, ord('q')):
                        break
                    continue
                print("Reconnected.")
                consecutive_fail = 0
                reopen_delay = min(reopen_delay * 1.5, 2.0)
                continue

            time.sleep(0.05)
            if (cv.waitKey(1) & 0xFF) in (27, ord('q')):
                break
            continue

        consecutive_fail = 0

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
    cap, label, open_pack = select_camera_interactive()
    if cap is None:
        print("No camera selected. Exiting.")
        return
    run_loop(cap, label, open_pack)
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
