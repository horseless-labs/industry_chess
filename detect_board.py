# chessboard_rectify_noprinter_menu.py
# Requires: pip install opencv-python numpy

import cv2 as cv
import numpy as np
from collections import deque
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
SCAN_MAX_INDEX = 10                # numeric index scan limit

# ---------- Globals for manual calibration & status ----------
_manual_clicks = []
_manual_H = None
_manual_active = False
_last_detector = ""

# ---------- Utilities ----------
def order_quad(pts):
    """Order 4 points as [tl, tr, br, bl]."""
    pts = np.array(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    pts = pts[np.argsort(ang)]
    tl = np.argmin(pts.sum(axis=1))
    return np.roll(pts, -tl, axis=0)

def homography_from_pts4(pts4, size=CANONICAL_SIZE):
    dst = np.float32([[0,0],[size-1,0],[size-1,size-1],[0,size-1]])
    return cv.getPerspectiveTransform(np.float32(pts4), dst)

def average_luma(img):
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return float(img.mean())

def orientation_fix(rectified):
    """Rotate 180° if a1 isn't dark (relative to h1)."""
    tile = CANONICAL_SIZE // 8
    pad = max(2, tile//10)
    a1 = rectified[(7*tile)+pad:(8*tile)-pad, 0+pad:tile-pad]
    h1 = rectified[(7*tile)+pad:(8*tile)-pad, (7*tile)+pad:(8*tile)-pad]
    if average_luma(a1) > average_luma(h1):
        return cv.rotate(rectified, cv.ROTATE_180)
    return rectified

def shrink_quad_toward_center(quad, frac=0.06):
    """Move each corner toward centroid by 'frac' of its distance."""
    q = quad.astype(np.float32)
    c = q.mean(axis=0, keepdims=True)
    return c + (q - c) * (1.0 - float(frac))

def checker_energy(rectified, tiles=8):
    """Cheap checker-ness: alternating tile mean contrast."""
    gray = cv.cvtColor(rectified, cv.COLOR_BGR2GRAY) if rectified.ndim == 3 else rectified
    S = gray.shape[0]
    t = S // tiles
    if t < 4:
        return 0.0
    acc, cnt = 0.0, 0
    for r in range(tiles):
        for c in range(tiles):
            y0, y1 = r*t, (r+1)*t
            x0, x1 = c*t, (c+1)*t
            m = float(gray[y0:y1, x0:x1].mean())
            acc += (1 if ((r+c) % 2 == 0) else -1) * m
            cnt += 1
    return abs(acc) / (cnt + 1e-6)

def poly_area(q):
    return 0.5*abs(np.dot(q[:,0], np.roll(q[:,1], -1)) - np.dot(q[:,1], np.roll(q[:,0], -1)))

# ---------- Manual calibration mouse callback ----------
def _mouse_cb(event, x, y, flags, userdata):
    global _manual_clicks, _manual_H, _manual_active
    if not _manual_active:
        return
    if event == cv.EVENT_LBUTTONDOWN:
        _manual_clicks.append((x, y))
        if len(_manual_clicks) == 4:
            q = order_quad(np.array(_manual_clicks, np.float32))
            _manual_H = homography_from_pts4(q)
            _manual_active = False
            print("Manual calibration captured.")

# ---------- Detection: Grid-first (CLAHE) ----------
def detect_grid_homography(frame):
    """Try to detect the 7x7 inner-corner grid to compute a robust H."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
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
    if H is None or mask is None:
        return None, 0.0
    inlier_ratio = float(mask.sum()) / len(mask)
    return H, inlier_ratio

# ---------- Detection: Contour (largest near-rectangle) ----------
def detect_outer_corners_contour(frame):
    """Find the largest near-rectangular 4-point contour (outer board/tape)."""
    work = cv.GaussianBlur(frame, (5,5), 0)
    gray = cv.cvtColor(work, cv.COLOR_BGR2GRAY)
    gray = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, 31, 5)
    th = cv.morphologyEx(th, cv.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    cnts, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0

    h, w = frame.shape[:2]
    best_quad, best_score, best_area = None, 0.0, 0.0
    for c in cnts:
        area = cv.contourArea(c)
        if area < 0.02 * w * h:
            continue
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        approx = approx.reshape(-1, 2).astype(np.float32)
        if not cv.isContourConvex(approx.astype(np.int32)):
            continue
        rect = cv.minAreaRect(approx)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        aspect = min(rw, rh) / max(rw, rh)
        squareness = aspect
        score = (area / (w*h + 1e-6)) * 0.8 + squareness * 0.2
        if score > best_score:
            best_score, best_area, best_quad = score, area, order_quad(approx)
    if best_quad is None:
        return None, 0.0
    area_norm = best_area / (w*h + 1e-6)
    conf = 0.6 * area_norm + 0.4 * best_score
    return best_quad, float(conf)

# ---------- Detection: Hough (fused with contour choice) ----------
def detect_outer_corners_hough(frame):
    """Try Hough lines; if small/weak, fall back to largest 4-pt contour."""
    work = cv.GaussianBlur(frame, (5,5), 0)
    gray = cv.cvtColor(work, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, EDGE_TH[0], EDGE_TH[1])
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                           minLineLength=MIN_LINE_LEN, maxLineGap=MAX_LINE_GAP)

    quad_h, conf_h = None, 0.0
    if lines is not None:
        horiz, vert = [], []
        for x1,y1,x2,y2 in lines[:,0,:]:
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            if abs(angle) < 20 or abs(angle) > 160: horiz.append([x1,y1,x2,y2])
            elif 70 < abs(angle) < 110:             vert.append([x1,y1,x2,y2])
        if len(horiz) >= 2 and len(vert) >= 2:
            horiz = np.array(horiz); vert = np.array(vert)
            def mids(L): return (L[:,0:2] + L[:,2:4]) / 2.0
            hm, vm = mids(horiz), mids(vert)
            top    = horiz[np.argmin(hm[:,1])]
            bottom = horiz[np.argmax(hm[:,1])]
            left   = vert [np.argmin(vm[:,0])]
            right  = vert [np.argmax(vm[:,0])]
            def params(l): x1,y1,x2,y2=l; A=y2-y1; B=x1-x2; C=A*x1+B*y1; return A,B,C
            def inter(l1,l2):
                A1,B1,C1 = params(l1); A2,B2,C2 = params(l2)
                det = A1*B2 - A2*B1
                if abs(det) < 1e-6: return None
                x = (B2*C1 - B1*C2)/det; y = (A1*C2 - A2*C1)/det
                return np.array([x,y], np.float32)
            pts = [inter(top,left), inter(top,right), inter(bottom,right), inter(bottom,left)]
            if all(p is not None for p in pts):
                q = order_quad(pts)
                h,w = frame.shape[:2]
                area = poly_area(q); area_norm = area/(w*h + 1e-6)
                v1, v2 = q[1]-q[0], q[3]-q[0]
                cosang = abs(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6))
                ortho = 1 - cosang
                conf_h = 0.5*area_norm + 0.5*max(0.0, ortho)
                quad_h = q

    quad_c, conf_c = detect_outer_corners_contour(frame)

    def area_norm(q):
        if q is None: return 0.0
        h,w = frame.shape[:2]
        return poly_area(q)/(w*h + 1e-6)

    if quad_h is None and quad_c is None:
        return None, 0.0
    if quad_h is None: return quad_c, conf_c
    if quad_c is None: return quad_h, conf_h

    ah, ac = area_norm(quad_h), area_norm(quad_c)
    if ac > ah * 1.05:
        return quad_c, max(conf_c, conf_h*0.9)
    return quad_h, max(conf_h, conf_c*0.9)

# ---------- Detection: Blue tape (HSV) ----------
def detect_board_from_blue_tape(frame, expect_inset_frac=0.06):
    """
    Detect board via blue painter's tape. Returns (quad_inset, conf).
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Broad blue range; tweak if your tape skews greener/bluer
    lower = np.array([90,  60,  40], np.uint8)
    upper = np.array([135, 255, 255], np.uint8)
    mask = cv.inRange(hsv, lower, upper)

    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  k, iterations=1)

    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0

    h, w = frame.shape[:2]
    best = max(cnts, key=cv.contourArea)
    area = cv.contourArea(best)
    if area < 0.02 * w * h:
        return None, 0.0

    rect = cv.minAreaRect(best)
    box  = cv.boxPoints(rect).astype(np.float32)
    quad_tape = order_quad(box)

    # Try a few insets and pick the most "checker-ish"
    candidates = [shrink_quad_toward_center(quad_tape, f) for f in (0.04, 0.06, 0.08, expect_inset_frac)]
    best_q, best_s = None, -1.0
    for q in candidates:
        Hq = homography_from_pts4(q)
        rectified = cv.warpPerspective(frame, Hq, (CANONICAL_SIZE, CANONICAL_SIZE))
        s = checker_energy(rectified)
        if s > best_s:
            best_s, best_q = s, q

    conf = 0.5 * (area / (w*h + 1e-6)) + 0.5 * min(best_s / 40.0, 1.0)  # gentle scale
    return best_q, float(conf)

# ---------- Tracking (KLT) ----------
class BoardTracker:
    def __init__(self):
        self.img_pts = None
        self.canon_pts = None
        self.prev_gray = None
        self.failed_frames = 0

    def seed_from_H(self, H, size=CANONICAL_SIZE, inner=BOARD_INNER):
        xs = np.linspace((0.5/8)*size, (7.5/8)*size, inner[0], dtype=np.float32)
        ys = np.linspace((0.5/8)*size, (7.5/8)*size, inner[1], dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys)
        canon = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)

        Hinv = np.linalg.inv(H)
        pts3 = np.concatenate([canon, np.ones((canon.shape[0],1), np.float32)], axis=1)
        proj = (Hinv @ pts3.T).T
        proj = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)

        self.img_pts = proj
        self.canon_pts = canon
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

        inlier = mask.reshape(-1).astype(bool)
        self.img_pts = tracked_img[inlier]
        self.canon_pts = tracked_can[inlier]
        self.prev_gray = gray
        self.failed_frames = 0
        inlier_ratio = float(inlier.sum()) / len(inlier)
        return H, inlier_ratio

def crop_to_quad(frame, quad, pad=20, min_area_norm=0.06):
    """Crop around quad with padding; if quad tiny, skip cropping."""
    if quad is None:
        return frame, (0,0), False, 1.0
    h, w = frame.shape[:2]
    area_norm = poly_area(quad) / (w*h + 1e-6)
    if area_norm < min_area_norm:
        return frame, (0,0), False, area_norm
    x0 = max(int(min(quad[:,0]))-pad, 0)
    y0 = max(int(min(quad[:,1]))-pad, 0)
    x1 = min(int(max(quad[:,0]))+pad, w)
    y1 = min(int(max(quad[:,1]))+pad, h)
    roi = frame[y0:y1, x0:x1]
    return roi, (x0, y0), True, area_norm

# ---------- Main rectification ----------
class HomographyKeeper:
    def __init__(self):
        self.H = None
        self.conf_hist = deque(maxlen=CONF_HISTORY)
        self.tracker = BoardTracker()
        self.weak_frames = 0

    def update(self, H, conf):
        if H is None:
            return False
        self.conf_hist.append(conf)
        if conf >= UPDATE_MIN_CONF or (self.H is None and conf > 0):
            self.H = H
            return True
        return False

def rectify_frame(frame, keeper):
    global _manual_H, _last_detector

    # 0) manual homography has top priority
    if _manual_H is not None:
        keeper.update(_manual_H, 1.0)
        rectified = cv.warpPerspective(frame, keeper.H, (CANONICAL_SIZE, CANONICAL_SIZE))
        rectified = orientation_fix(rectified)
        _last_detector = "manual"
        return rectified, 1.0, (0.0, 0.0), None

    # 1) try blue-tape detector (fast and robust)
    quad_tape, conf_tape = detect_board_from_blue_tape(frame)
    if quad_tape is not None and conf_tape >= 0.35:
        Htape = homography_from_pts4(quad_tape)
        keeper.update(Htape, conf_tape)
        if keeper.tracker.canon_pts is None:
            keeper.tracker.seed_from_H(Htape)
        rectified = cv.warpPerspective(frame, Htape, (CANONICAL_SIZE, CANONICAL_SIZE))
        rectified = orientation_fix(rectified)
        _last_detector = "tape"
        return rectified, conf_tape, (0.0, conf_tape), quad_tape

    # 2) predictive tracking if we already have an H
    H = None
    conf_track = 0.0
    quad_est = None

    if keeper.H is not None:
        Ht, r = keeper.tracker.track_and_refit(frame)
        if Ht is not None and r >= 0.4:
            H = Ht
            conf_track = r
        # estimate quad from previous H, then nudge inward (avoid tape bezel)
        canon_corners = np.float32([[0,0],[CANONICAL_SIZE-1,0],[CANONICAL_SIZE-1,CANONICAL_SIZE-1],[0,CANONICAL_SIZE-1]])
        Hinv = np.linalg.inv(keeper.H)
        pts3 = np.concatenate([canon_corners, np.ones((4,1), np.float32)], axis=1)
        proj = (Hinv @ pts3.T).T
        quad_est = shrink_quad_toward_center(order_quad((proj[:, :2] / proj[:, 2:3]).astype(np.float32)), frac=0.03)

    # ROI (self-correcting)
    search_frame, offset, used_roi, area_est = (frame, (0,0), False, 0.0)
    if quad_est is not None:
        pad = 40 if conf_track < 0.6 else 30
        search_frame, offset, used_roi, area_est = crop_to_quad(frame, quad_est, pad=pad, min_area_norm=0.06)

    conf_grid = 0.0
    conf_hough = 0.0
    quad = None

    # 3) grid detector
    if H is None:
        Hg, r = detect_grid_homography(search_frame)
        if Hg is not None:
            T = np.array([[1,0,-offset[0]],[0,1,-offset[1]],[0,0,1]], np.float32)
            H = Hg @ T
            conf_grid = float(r)

    # 4) corner fallback(s)
    if (H is None) or (conf_grid < 0.5 and conf_track < 0.5):
        res = detect_outer_corners_hough(search_frame)
        if res[0] is not None:
            quad = res[0] + np.array(offset, dtype=np.float32)  # un-crop
            # If quad is very large (likely bezel), try insets and choose most checker-y
            h,w = frame.shape[:2]
            area_norm = poly_area(quad) / (w*h + 1e-6)
            quad_candidates = [quad]
            if area_norm > 0.30:
                for f in [0.04, 0.06, 0.08]:
                    quad_candidates.append(shrink_quad_toward_center(quad, frac=f))
            best_quad, best_score = None, -1.0
            for q in quad_candidates:
                Hq = homography_from_pts4(q)
                rect = cv.warpPerspective(frame, Hq, (CANONICAL_SIZE, CANONICAL_SIZE))
                score = checker_energy(rect)
                if score > best_score:
                    best_score, best_quad = score, q
            quad = best_quad
            H = homography_from_pts4(quad)
            conf_hough = res[1]
        else:
            quad = None
    else:
        quad = quad_est

    # 5) If ROI used but quad tiny/weak, retry on full frame once
    tiny_or_weak = False
    if quad is not None:
        h,w = frame.shape[:2]
        if (poly_area(quad)/(w*h + 1e-6) < 0.06) and (max(conf_grid, conf_track, conf_hough) < 0.6):
            tiny_or_weak = True
    if used_roi and tiny_or_weak:
        Hg2, r2 = detect_grid_homography(frame)
        H2, conf2_grid = (Hg2, r2) if Hg2 is not None else (None, 0.0)
        res2 = detect_outer_corners_hough(frame) if (H2 is None or conf2_grid < 0.5) else (None, 0.0)
        if res2[0] is not None and (H2 is None or conf2_grid < 0.5):
            quad = res2[0]
            H2 = homography_from_pts4(quad)
            conf_hough = max(conf_hough, res2[1])
        if H2 is not None and (H is None or conf2_grid > conf_grid):
            H, conf_grid = H2, max(conf_grid, conf2_grid)

    # 6) confidence & keeper update
    conf = max(conf_track, conf_grid, conf_hough)
    if conf < 0.35:
        keeper.weak_frames += 1
    else:
        keeper.weak_frames = 0
    if keeper.weak_frames >= 10:
        keeper.H = None
        keeper.tracker = BoardTracker()
        keeper.weak_frames = 0
        _last_detector = "reset"
        return None, conf, (conf_grid, conf_hough), None

    keeper.update(H, conf)
    if keeper.H is None:
        _last_detector = "none"
        return None, conf, (conf_grid, conf_hough), None

    if conf >= 0.55 and (keeper.tracker.canon_pts is None or len(keeper.tracker.canon_pts) < 20 or keeper.tracker.failed_frames > 0):
        keeper.tracker.seed_from_H(keeper.H)

    rectified = cv.warpPerspective(frame, keeper.H, (CANONICAL_SIZE, CANONICAL_SIZE))
    rectified = orientation_fix(rectified)
    _last_detector = "grid/hough" if conf_grid >= conf_hough else "hough/grid"
    return rectified, conf, (conf_grid, conf_hough), quad

# ---------- Safer camera enumeration & opening ----------
def list_cameras():
    cams, key = [], 0
    if platform.system() == 'Linux':
        for path in sorted(glob.glob('/dev/video*')):
            cams.append({'key': str(key), 'label': f"{path} (V4L2)", 'open_arg': path, 'api': cv.CAP_V4L2}); key += 1
    for idx in range(min(SCAN_MAX_INDEX, 6)):
        cams.append({'key': str(key), 'label': f"Device Index {idx}", 'open_arg': idx, 'api': None}); key += 1
    return cams

def _open_camera(open_arg, api_pref):
    cap = cv.VideoCapture(open_arg, api_pref) if api_pref else cv.VideoCapture(open_arg)
    if not cap.isOpened():
        return None
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

def select_camera_interactive():
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
    cv.putText(vis, f"grid:{conf_grid:.2f}  hough:{conf_hough:.2f}  mode:{_last_detector}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)
    if rectified is not None:
        tile = CANONICAL_SIZE // 8
        for i in range(9):
            cv.line(rectified, (0, i*tile), (CANONICAL_SIZE, i*tile), (128,128,128), 1)
            cv.line(rectified, (i*tile, 0), (i*tile, CANONICAL_SIZE), (128,128,128), 1)
    return vis, rectified

def run_loop(cap, cam_label, open_arg_and_api):
    global _manual_clicks, _manual_H, _manual_active
    win = "Board Rectification"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.setMouseCallback(win, _mouse_cb)

    consecutive_fail = 0
    reopen_delay = 0.5
    keeper = HomographyKeeper()

    print("Press 'c' to manually click 4 corners (TL→TR→BR→BL). ESC/q to quit.")

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
                cv.imshow(win, canvas)
            else:
                cv.putText(vis, cam_label, (10, vis.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
                cv.imshow(win, vis)

        key = cv.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if key == ord('c'):
            _manual_clicks = []
            _manual_H = None
            _manual_active = True
            print("Manual calibration: click 4 board corners clockwise from top-left.")

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
