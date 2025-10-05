# chessboard_rectify_slim_record.py
# pip install opencv-python numpy
import cv2 as cv
import numpy as np
from collections import deque
import glob, platform, time, argparse, os
from datetime import datetime

# ---------- Config ----------
CANONICAL_SIZE = 512
BOARD_INNER = (7, 7)
EDGE_TH = (60, 180)
MIN_LINE_LEN = 120
MAX_LINE_GAP = 10
CONF_HISTORY = 10
UPDATE_MIN_CONF = 0.35
SCAN_MAX_INDEX = 10
SHOW = True

# ---------- Globals ----------
_manual_clicks = []
_manual_H = None
_manual_active = False
_last_mode = "none"

# ---------- Small geometry / image utils ----------
def to_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img

def order_quad(pts4):
    pts = np.array(pts4, np.float32)
    c = pts.mean(0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    pts = pts[np.argsort(ang)]
    tl = np.argmin(pts.sum(1))
    return np.roll(pts, -tl, 0)

def poly_area(q):
    return 0.5 * abs(np.dot(q[:,0], np.roll(q[:,1], -1)) - np.dot(q[:,1], np.roll(q[:,0], -1)))

def shrink_toward_center(q, frac=0.06):
    q = q.astype(np.float32)
    c = q.mean(0, keepdims=True)
    return c + (q - c) * (1.0 - float(frac))

def H_from_quad(q, size=CANONICAL_SIZE):
    dst = np.float32([[0,0],[size-1,0],[size-1,size-1],[0,size-1]])
    return cv.getPerspectiveTransform(np.float32(q), dst)

def warp_rectify(frame, H, size=CANONICAL_SIZE):
    return cv.warpPerspective(frame, H, (size, size))

def avg_luma(img):
    g = to_gray(img)
    return float(g.mean())

def orientation_fix(rectified):
    S = rectified.shape[0]
    t = S // 8
    pad = max(2, t//10)
    a1 = rectified[(7*t)+pad:(8*t)-pad, 0+pad:t-pad]
    h1 = rectified[(7*t)+pad:(8*t)-pad, (7*t)+pad:(8*t)-pad]
    if avg_luma(a1) > avg_luma(h1):
        return cv.rotate(rectified, cv.ROTATE_180)
    return rectified

def checker_energy(rectified, tiles=8):
    g = to_gray(rectified)
    S = g.shape[0]
    t = S // tiles
    if t < 4: return 0.0
    acc, cnt = 0.0, 0
    for r in range(tiles):
        for c in range(tiles):
            y0, y1 = r*t, (r+1)*t
            x0, x1 = c*t, (c+1)*t
            m = float(g[y0:y1, x0:x1].mean())
            acc += (1 if ((r+c) % 2 == 0) else -1) * m
            cnt += 1
    return abs(acc) / (cnt + 1e-6)

def crop_to_quad(frame, quad, pad=24, min_area_norm=0.06):
    if quad is None: return frame, (0,0), False, 1.0
    h, w = frame.shape[:2]
    area_norm = poly_area(quad) / (w*h + 1e-6)
    if area_norm < min_area_norm:
        return frame, (0,0), False, area_norm
    x0 = max(int(min(quad[:,0]))-pad, 0)
    y0 = max(int(min(quad[:,1]))-pad, 0)
    x1 = min(int(max(quad[:,0]))+pad, w)
    y1 = min(int(max(quad[:,1]))+pad, h)
    return frame[y0:y1, x0:x1], (x0,y0), True, area_norm

# ---------- Manual calibration mouse callback ----------
def _mouse_cb(event, x, y, flags, userdata):
    global _manual_clicks, _manual_H, _manual_active
    if not _manual_active: return
    if event == cv.EVENT_LBUTTONDOWN:
        _manual_clicks.append((x,y))
        if len(_manual_clicks) == 4:
            q = order_quad(np.array(_manual_clicks, np.float32))
            _manual_H = H_from_quad(q)
            _manual_active = False
            print("Manual calibration captured.")

# ---------- Detectors ----------
def detect_grid_H(frame):
    gray = to_gray(frame)
    gray = cv.createCLAHE(2.0, (8,8)).apply(gray)
    flags = cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY
    ok, corners = cv.findChessboardCornersSB(gray, BOARD_INNER, flags=flags)
    if not ok: return None, 0.0
    img_pts = corners.reshape(-1,2).astype(np.float32)
    xs = np.linspace((0.5/8)*CANONICAL_SIZE, (7.5/8)*CANONICAL_SIZE, BOARD_INNER[0])
    ys = np.linspace((0.5/8)*CANONICAL_SIZE, (7.5/8)*CANONICAL_SIZE, BOARD_INNER[1])
    XX, YY = np.meshgrid(xs, ys)
    canon_pts = np.stack([XX.ravel(), YY.ravel()], 1).astype(np.float32)
    H, mask = cv.findHomography(img_pts, canon_pts, cv.RANSAC, 3.0)
    if H is None or mask is None: return None, 0.0
    return H, float(mask.sum()) / len(mask)

def detect_outer_quad_contour(frame):
    work = cv.GaussianBlur(frame, (5,5), 0)
    gray = cv.createCLAHE(2.0, (8,8)).apply(to_gray(work))
    th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, 31, 5)
    th = cv.morphologyEx(th, cv.MORPH_CLOSE, np.ones((5,5), np.uint8), 2)
    cnts, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, 0.0
    h, w = frame.shape[:2]
    best_q, best_score = None, -1
    for c in cnts:
        area = cv.contourArea(c)
        if area < 0.02 * w * h: continue
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4: continue
        approx = approx.reshape(-1,2).astype(np.float32)
        if not cv.isContourConvex(approx.astype(np.int32)): continue
        rw, rh = cv.minAreaRect(approx)[1]
        if rw < 1 or rh < 1: continue
        squareness = min(rw, rh) / max(rw, rh)
        score = 0.8 * (area / (w*h+1e-6)) + 0.2 * squareness
        if score > best_score:
            best_q, best_score = order_quad(approx), score
    if best_q is None: return None, 0.0
    area_norm = poly_area(best_q) / (w*h + 1e-6)
    conf = 0.6 * area_norm + 0.4 * best_score
    return best_q, float(conf)

def detect_outer_quad_hough(frame):
    gray = to_gray(cv.GaussianBlur(frame, (5,5), 0))
    edges = cv.Canny(gray, EDGE_TH[0], EDGE_TH[1])
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                           minLineLength=MIN_LINE_LEN, maxLineGap=MAX_LINE_GAP)
    if lines is None: return None, 0.0
    horiz, vert = [], []
    for x1,y1,x2,y2 in lines[:,0,:]:
        ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if ang < 20 or ang > 160: horiz.append([x1,y1,x2,y2])
        elif 70 < ang < 110:      vert.append([x1,y1,x2,y2])
    if len(horiz) < 2 or len(vert) < 2: return None, 0.0
    horiz, vert = np.array(horiz), np.array(vert)
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
    if any(p is None for p in pts): return None, 0.0
    q = order_quad(pts)
    h,w = frame.shape[:2]
    area_norm = poly_area(q) / (w*h + 1e-6)
    v1, v2 = q[1]-q[0], q[3]-q[0]
    cosang = abs(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6))
    ortho = 1 - cosang
    conf = 0.5*area_norm + 0.5*max(0.0, ortho)
    return q, float(conf)

def detect_blue_tape_quad(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, np.array([90,60,40], np.uint8), np.array([135,255,255], np.uint8))
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, 2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  k, 1)
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, 0.0
    h, w = frame.shape[:2]
    best = max(cnts, key=cv.contourArea)
    area = cv.contourArea(best)
    if area < 0.02 * w * h: return None, 0.0
    rect = cv.minAreaRect(best)
    quad = order_quad(cv.boxPoints(rect).astype(np.float32))
    candidates = [shrink_toward_center(quad, f) for f in (0.04, 0.06, 0.08)]
    best_q, best_s = None, -1.0
    for q in candidates:
        Hq = H_from_quad(q)
        rectified = warp_rectify(frame, Hq)
        s = checker_energy(rectified)
        if s > best_s:
            best_s, best_q = s, q
    conf = 0.5 * (area/(w*h+1e-6)) + 0.5 * min(best_s/40.0, 1.0)
    return best_q, float(conf)

# ---------- Tracking ----------
class BoardTracker:
    def __init__(self):
        self.img_pts = None
        self.canon_pts = None
        self.prev_gray = None
        self.failed = 0

    def seed(self, H, size=CANONICAL_SIZE, inner=BOARD_INNER):
        xs = np.linspace((0.5/8)*size, (7.5/8)*size, inner[0], dtype=np.float32)
        ys = np.linspace((0.5/8)*size, (7.5/8)*size, inner[1], dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys)
        canon = np.stack([XX.ravel(), YY.ravel()], 1).astype(np.float32)
        Hinv = np.linalg.inv(H)
        pts3 = np.hstack([canon, np.ones((canon.shape[0],1), np.float32)])
        proj = (Hinv @ pts3.T).T
        proj = (proj[:,:2] / proj[:,2:3]).astype(np.float32)
        self.img_pts, self.canon_pts = proj, canon
        self.prev_gray, self.failed = None, 0

    def estimate_quad_from(self, H):
        canon_corners = np.float32([[0,0],[CANONICAL_SIZE-1,0],[CANONICAL_SIZE-1,CANONICAL_SIZE-1],[0,CANONICAL_SIZE-1]])
        Hinv = np.linalg.inv(H)
        pts3 = np.hstack([canon_corners, np.ones((4,1), np.float32)])
        proj = (Hinv @ pts3.T).T
        return order_quad((proj[:,:2] / proj[:,2:3]).astype(np.float32))

    def step(self, frame_bgr):
        if self.img_pts is None: return None, 0.0
        gray = to_gray(frame_bgr)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, 0.0
        next_pts, st, err = cv.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.img_pts.reshape(-1,1,2),
            None, winSize=(21,21), maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.prev_gray = gray
        if next_pts is None or st is None:
            self.failed += 1; return None, 0.0
        st = st.reshape(-1).astype(bool)
        img_t = next_pts.reshape(-1,2)[st]
        can_t = self.canon_pts[st]
        if len(img_t) < 12:
            self.failed += 1; return None, 0.0
        H, mask = cv.findHomography(img_t, can_t, cv.RANSAC, 3.0)
        if H is None or mask is None:
            self.failed += 1; return None, 0.0
        inl = mask.reshape(-1).astype(bool)
        self.img_pts, self.canon_pts = img_t[inl], can_t[inl]
        self.failed = 0
        return H, float(inl.sum()) / len(inl)

# ---------- Keeper ----------
class HomographyKeeper:
    def __init__(self):
        self.H = None
        self.conf_hist = deque(maxlen=CONF_HISTORY)
        self.tracker = BoardTracker()
        self.weak = 0
    def update(self, H, conf):
        if H is None: return False
        self.conf_hist.append(conf)
        if conf >= UPDATE_MIN_CONF or (self.H is None and conf > 0):
            self.H = H
            return True
        return False
    def reset(self):
        self.H = None
        self.tracker = BoardTracker()
        self.weak = 0

# ---------- Unified rectification ----------
def rectify_frame(frame, keeper):
    global _manual_H, _last_mode
    if _manual_H is not None:
        keeper.update(_manual_H, 1.0)
        rect = orientation_fix(warp_rectify(frame, keeper.H))
        _last_mode = "manual"
        return rect, 1.0, (1.0, 0.0), None

    quad_tape, conf_tape = detect_blue_tape_quad(frame)
    if quad_tape is not None and conf_tape >= 0.35:
        Ht = H_from_quad(quad_tape)
        keeper.update(Ht, conf_tape)
        if keeper.tracker.canon_pts is None:
            keeper.tracker.seed(Ht)
        rect = orientation_fix(warp_rectify(frame, Ht))
        _last_mode = "tape"
        return rect, conf_tape, (0.0, conf_tape), quad_tape

    Hpred, rtrack = (None, 0.0)
    quad_est = None
    if keeper.H is not None:
        Ht, r = keeper.tracker.step(frame)
        if Ht is not None and r >= 0.4:
            Hpred, rtrack = Ht, r
        quad_est = shrink_toward_center(keeper.tracker.estimate_quad_from(keeper.H), 0.03)

    search, off, used_roi, _ = (frame, (0,0), False, 0.0)
    if quad_est is not None:
        pad = 40 if rtrack < 0.6 else 28
        search, off, used_roi, _ = crop_to_quad(frame, quad_est, pad=pad)

    H, rgrid = (None, 0.0)
    if Hpred is None:
        Hg, rg = detect_grid_H(search)
        if Hg is not None:
            T = np.array([[1,0,-off[0]],[0,1,-off[1]],[0,0,1]], np.float32)
            H, rgrid = Hg @ T, float(rg)

    quad, rhough = (None, 0.0)
    if (H is None) or (rgrid < 0.5 and rtrack < 0.5):
        qh, ch = detect_outer_quad_hough(search)
        qc, cc = detect_outer_quad_contour(search)
        def area_norm(q):
            if q is None: return 0.0
            h,w = frame.shape[:2]
            return poly_area(q)/(h*w+1e-6)
        cand = max(((qh,ch),(qc,cc)), key=lambda qc: area_norm(qc[0]))
        quad, rhough = cand
        if quad is not None:
            quad = quad + np.array(off, np.float32)
            tests = [quad] + ([shrink_toward_center(quad, f) for f in (0.04,0.06,0.08)] if area_norm(quad) > 0.30 else [])
            best_q, best_s = None, -1.0
            for q in tests:
                Hq = H_from_quad(q)
                s = checker_energy(warp_rectify(frame, Hq))
                if s > best_s:
                    best_q, best_s = q, s
            quad = best_q
            H = H_from_quad(quad)

    conf = max(rtrack, rgrid, rhough)
    keeper.weak = (keeper.weak + 1) if conf < 0.35 else 0
    if keeper.weak >= 10:
        keeper.reset()
        _last_mode = "reset"
        return None, conf, (rgrid, rhough), None

    keeper.update(H, conf)
    if keeper.H is None:
        _last_mode = "none"
        return None, conf, (rgrid, rhough), None

    if conf >= 0.55 and (keeper.tracker.canon_pts is None or len(keeper.tracker.canon_pts) < 20 or keeper.tracker.failed):
        keeper.tracker.seed(keeper.H)

    rectified = orientation_fix(warp_rectify(frame, keeper.H))
    _last_mode = "grid" if rgrid >= rhough else "hough"
    return rectified, conf, (rgrid, rhough), quad

# ---------- Camera helpers ----------
import os, glob

def list_cameras():
    """
    Prefer stable /dev/v4l/by-id symlinks; fall back to /dev/videoN.
    Returns a list of dicts: {key,label,open_arg,api,stable_id}
    """
    cams, key = [], 0

    # Prefer stable symlinks (Linux)
    by_id = sorted(glob.glob('/dev/v4l/by-id/*'))
    for p in by_id:
        try:
            real = os.path.realpath(p)   # e.g. /dev/video4
            label = f"{os.path.basename(p)} -> {os.path.basename(real)}"
            cams.append({
                'key': str(key),
                'label': label,
                'open_arg': p,            # open via stable symlink
                'api': cv.CAP_V4L2,
                'stable_id': os.path.basename(p),
            })
            key += 1
        except Exception:
            pass

    # Fallback: volatile /dev/videoN
    for path in sorted(glob.glob('/dev/video*')):
        cams.append({
            'key': str(key),
            'label': f"{path} (V4L2)",
            'open_arg': path,
            'api': cv.CAP_V4L2,
            'stable_id': None,
        })
        key += 1

    # As an absolute last fallback, try numeric indices 0..5 (non-Linux or exotic)
    if not cams:
        for idx in range(6):
            cams.append({
                'key': str(key),
                'label': f"Device Index {idx}",
                'open_arg': idx,
                'api': None,
                'stable_id': None,
            })
            key += 1
    return cams

def _open_camera(open_arg, api_pref):
    cap = cv.VideoCapture(open_arg, api_pref) if api_pref else cv.VideoCapture(open_arg)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    try:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv.CAP_PROP_FPS, 30)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    except Exception as e:
        print("Warning: failed to set some camera properties:", e)
    print(f"Camera opened at {int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv.CAP_PROP_FPS):.1f} FPS")
    ok, frame = cap.read()
    if not ok or frame is None:
        print("Error: failed to read initial frame.")
        cap.release()
        return None
    return cap


def select_camera_interactive():
    while True:
        cams = list_cameras()
        print("\n=== Select a Camera ===")
        for c in cams:
            print(f"[{c['key']}] {c['label']}")
        print("\n[r] Refresh   [q] Quit")
        choice = input("Enter selection: ").strip().lower()
        if choice == 'q':
            return None, None, None
        if choice == 'r':
            continue
        sel = next((c for c in cams if c['key'] == choice), None)
        if not sel:
            print("Invalid selection."); continue
        cap = _open_camera(sel['open_arg'], sel['api'])
        if cap is None:
            print("Open failed."); continue
        print(f"Opened: {sel['label']}")
        return cap, sel['label'], (sel['open_arg'], sel['api'], sel.get('stable_id'))


# ---------- Video recording helpers ----------
def make_timestamped(base, ext):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.splitext(base)[0]
    return f"{root}_{stamp}{ext}"

def create_writer(path, size, fps, fourcc_str="MJPG"):
    fourcc = cv.VideoWriter_fourcc(*fourcc_str)
    return cv.VideoWriter(path, fourcc, fps, size)

class DualRecorder:
    def __init__(self, orig_path=None, rect_path=None, fps=30, codec="MJPG"):
        self.codec = codec
        self.fps = fps
        self.orig_path = orig_path
        self.rect_path = rect_path
        self.orig_writer = None
        self.rect_writer = None
        self.enabled = False
        self._sizes_ready = False
        self._orig_size = None  # (w,h)
        self._rect_size = (CANONICAL_SIZE, CANONICAL_SIZE)

    def ensure_open(self, frame_bgr):
        if not self.enabled: return
        if not self._sizes_ready:
            h, w = frame_bgr.shape[:2]
            self._orig_size = (w, h)
            self._sizes_ready = True
        if self.orig_writer is None:
            path = self.orig_path or make_timestamped("orig.avi", ".avi")
            self.orig_writer = create_writer(path, self._orig_size, self.fps, self.codec)
            print(f"[REC] Writing original to {path} @ {self._orig_size} {self.fps}fps {self.codec}")
        if self.rect_writer is None:
            path = self.rect_path or make_timestamped("rect.avi", ".avi")
            self.rect_writer = create_writer(path, self._rect_size, self.fps, self.codec)
            print(f"[REC] Writing rectified to {path} @ {self._rect_size} {self.fps}fps {self.codec}")

    def write(self, frame_bgr, rectified_bgr):
        if not self.enabled: return
        if frame_bgr is None: return
        self.ensure_open(frame_bgr)
        if self.orig_writer: self.orig_writer.write(frame_bgr)
        if rectified_bgr is not None and self.rect_writer:
            # Ensure rectified matches writer size
            if rectified_bgr.shape[1] != self._rect_size[0] or rectified_bgr.shape[0] != self._rect_size[1]:
                rectified_bgr = cv.resize(rectified_bgr, self._rect_size)
            self.rect_writer.write(rectified_bgr)

    def toggle(self):
        self.enabled = not self.enabled
        print(f"[REC] {'Started' if self.enabled else 'Stopped'} recording.")
        if not self.enabled:
            self.release()

    def release(self):
        if self.orig_writer:
            self.orig_writer.release()
            self.orig_writer = None
        if self.rect_writer:
            self.rect_writer.release()
            self.rect_writer = None

def _reopen_camera_or_rescan(open_pack, recorder):
    """
    Try to reopen the same camera. If it no longer exists (errno 19),
    rescan /dev/v4l/by-id to find a device with the same stable_id.
    Returns (cap, new_open_pack or None). Either may be None on failure.
    """
    if open_pack is None:
        return None, None

    # Pause recording while device is down (avoids writing black frames)
    if recorder and recorder.enabled:
        print("[REC] Pausing due to camera failure...")
        recorder.toggle()  # stops & releases writers

    prev_open_arg, prev_api, prev_stable = open_pack

    # 1) Try the same open_arg first
    cap = _open_camera(prev_open_arg, prev_api)
    if cap is not None:
        print("Reconnected (same path).")
        return cap, open_pack

    # 2) If we had a stable_id, try to find it again in by-id
    if prev_stable:
        cams = list_cameras()
        match = next((c for c in cams if c.get('stable_id') == prev_stable), None)
        if match:
            cap = _open_camera(match['open_arg'], match['api'])
            if cap is not None:
                print(f"Reconnected via by-id: {match['label']}")
                return cap, (match['open_arg'], match['api'], match['stable_id'])

    print("Reopen failed; will retry...")
    return None, open_pack


# ---------- UI / loop ----------
def draw_debug(frame, rectified, confs, quad, rec_on=False):
    vis = frame.copy()
    rgrid, rhough = confs
    if quad is not None:
        cv.polylines(vis, [quad.astype(int)], True, (0,255,0), 2)
    status = f"grid:{rgrid:.2f}  hough:{rhough:.2f}  mode:{_last_mode}"
    if rec_on: status += "  [REC]"
    cv.putText(vis, status, (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)
    if rectified is not None:
        t = CANONICAL_SIZE // 8
        for i in range(9):
            cv.line(rectified, (0, i*t), (CANONICAL_SIZE, i*t), (128,128,128), 1)
            cv.line(rectified, (i*t, 0), (i*t, CANONICAL_SIZE), (128,128,128), 1)
    return vis, rectified

def run_loop(cap, cam_label, open_pack, recorder):
    global _manual_clicks, _manual_H, _manual_active
    win = "Board Rectification"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.setMouseCallback(win, _mouse_cb)
    print("Press 'r' to start/stop recording (both streams).")
    print("Press 'c' to click 4 corners (TL→TR→BR→BL). ESC/q quits.")

    keeper = HomographyKeeper()
    consecutive_fail, reopen_delay = 0, 0.5

    while True:
        # If we currently have no camera, sleep and attempt reopen soon
        if cap is None:
            time.sleep(0.25)
            cap, open_pack = _reopen_camera_or_rescan(open_pack, recorder)
            blank = np.zeros((480, 640, 3), np.uint8)
            cv.putText(blank, "Camera not available... trying to reconnect",
                       (20, 240), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
            cv.imshow(win, blank)
            if (cv.waitKey(1) & 0xFF) in (27, ord('q')):
                break
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            consecutive_fail += 1
            print("Camera read failed.")
            if consecutive_fail >= 3:
                print("Attempting to reopen camera...")
                try: cap.release()
                except Exception: pass
                cap = None
                cap, open_pack = _reopen_camera_or_rescan(open_pack, recorder)
                consecutive_fail = 0
                reopen_delay = min(reopen_delay * 1.5, 2.0)
            if (cv.waitKey(1) & 0xFF) in (27, ord('q')):
                break
            continue

        consecutive_fail = 0
        rectified, conf, confs, quad = rectify_frame(frame, keeper)

        # --- recording ---
        recorder.write(frame, rectified)

        if SHOW:
            vis, rect = draw_debug(frame, None if rectified is None else rectified.copy(), confs, quad, rec_on=recorder.enabled)
            if rect is not None:
                side = CANONICAL_SIZE
                h = max(vis.shape[0], side)
                canvas = np.zeros((h, vis.shape[1]+side, 3), dtype=np.uint8)
                canvas[:vis.shape[0], :vis.shape[1]] = vis
                canvas[:side, vis.shape[1]:vis.shape[1]+side] = rect
                cv.putText(canvas, cam_label, (10, h-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
                cv.imshow(win, canvas)
            else:
                cv.putText(vis, cam_label, (10, vis.shape[0]-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
                cv.imshow(win, vis)

        key = cv.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('c'):
            _manual_clicks, _manual_H, _manual_active = [], None, True
            print("Manual calibration: click 4 board corners clockwise from top-left.")
        if key == ord('r'):
            recorder.toggle()

    recorder.release()

# ---------- Main ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Chessboard rectification with dual recording")
    ap.add_argument("--orig", type=str, default=None, help="Output path for original video (e.g., orig.avi)")
    ap.add_argument("--rect", type=str, default=None, help="Output path for rectified video (e.g., rect.avi)")
    ap.add_argument("--fps", type=int, default=30, help="Recording FPS")
    ap.add_argument("--codec", type=str, default="MJPG", help="FOURCC codec (e.g., MJPG, XVID, H264)")
    return ap.parse_args()

def main():
    args = parse_args()
    cap, label, pack = select_camera_interactive()
    if cap is None:
        print("No camera selected. Exiting."); return
    recorder = DualRecorder(orig_path=args.orig, rect_path=args.rect, fps=args.fps, codec=args.codec)
    try:
        run_loop(cap, label, pack, recorder)
    finally:
        recorder.release()
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
