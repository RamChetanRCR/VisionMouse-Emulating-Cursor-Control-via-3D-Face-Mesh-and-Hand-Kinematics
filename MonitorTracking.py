
"""
=============================================================================
  COMBINED 3D EYE TRACKER + HAND GESTURE CONTROLLER  v4.0
=============================================================================
  

=============================================================================
  CALIBRATION (every session):
    S  → Look at monitor centre → press S  (sets screen plane)
    C  → Press C and follow the 16-point dot
    R  → Look at screen centre → press R   (re-anchor drift)
    E  → Nudge edge-stretch up (repeat to reach corners better)
    Shift+E → Nudge edge-stretch down

  MODES:  1=Eye  2=Hand  3=Dual(default)
  MOUSE:  M=toggle
  FILTER: F/G=slower/faster   =/- sensitivity   I/K/J/L bias
  DEBUG:  A/D/W/S/Z/X orbit   Q=quit
=============================================================================
  HAND GESTURES (hold 0.4 s):
    ✌  Index+Middle up        → LEFT CLICK
    ☝  Index only             → RIGHT CLICK
    👌  Pinch (thumb+index)   → DOUBLE CLICK
    ✋  Open hand (back)       → SCROLL UP
    ✊  Fist                   → SCROLL DOWN
    🖐  Open palm (facing cam) → SCROLL UP   ← NEW, works palm-side

  EYE GESTURES (eye/dual):
    Both closed 0.6 s → LEFT CLICK
    Left wink          → RIGHT CLICK
    Right wink         → DOUBLE CLICK
=============================================================================
"""

import cv2
import numpy as np
import os
import sys
import time
import math
import threading
from scipy.spatial.transform import Rotation as Rscipy
import mediapipe as mp
import pyautogui

pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    _SKL = True
except ImportError:
    _SKL = False
    print("[WARN] scikit-learn not found — install it: pip install scikit-learn")

# ============================================================================
# 1.  CONFIG
# ============================================================================
MONITOR_W, MONITOR_H = pyautogui.size()

filter_alpha = 0.18
sensitivity_x = 1.0
sensitivity_y = 1.0
bias_x = 0.0
bias_y = 0.0
invert_x = False
invert_y = False
DEADZONE_PX = 8
control_mode = "dual"
edge_stretch = 1.08   # >1.0 expands the mapped range so edges are reachable

home_dir = os.path.expanduser("~")
screen_position_file = os.path.join(
    home_dir, "Documents", "screen_position.txt")

# ============================================================================
# 2.  UTILITIES
# ============================================================================


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _focal_px(width, fov_deg=50.0):
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


def _rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], float)


def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], float)


def compute_scale(pts):
    n, tot, cnt = len(pts), 0, 0
    for i in range(n):
        for j in range(i+1, n):
            tot += np.linalg.norm(pts[i]-pts[j])
            cnt += 1
    return tot / cnt if cnt else 1.0


def write_pos(x, y):
    try:
        with open(screen_position_file, 'w') as f:
            f.write(f"{x},{y}\n")
    except Exception:
        pass

# ============================================================================
# 3.  ADAPTIVE DOUBLE-EMA GAZE FILTER
# ============================================================================


class GazeFilter:
    def __init__(self, alpha=0.18):
        self.base_alpha = alpha
        self.alpha = alpha
        self.s1 = self.s2 = None
        self._prev = None

    def update(self, pt):
        pt = np.array(pt, float)
        if self._prev is not None:
            spd = np.linalg.norm(pt - self._prev)
            t = np.clip((spd - 0.01) / 0.14, 0.0, 1.0)
            self.alpha = self.base_alpha * (1.0 + 1.2 * t)
        self._prev = pt.copy()
        if self.s1 is None:
            self.s1 = self.s2 = pt.copy()
            return pt
        self.s1 = self.alpha * pt + (1 - self.alpha) * self.s1
        self.s2 = self.alpha * self.s1 + (1 - self.alpha) * self.s2
        trend = (self.alpha / (1 - self.alpha + 1e-9)) * (self.s1 - self.s2)
        return self.s1 + trend

    def reset(self):
        self.s1 = self.s2 = self._prev = None
        self.alpha = self.base_alpha

    def set_alpha(self, a):
        self.base_alpha = self.alpha = max(0.02, min(0.5, a))


gaze_filter = GazeFilter(filter_alpha)

# ============================================================================
# 4.  EAR
# ============================================================================
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]


def ear(pts):
    p = [np.array(q) for q in pts]
    A = np.linalg.norm(p[1]-p[5])
    B = np.linalg.norm(p[2]-p[4])
    C = np.linalg.norm(p[0]-p[3])
    return (A+B)/(2.0*C) if C > 1e-6 else 0.0


BLINK_RATIO = 0.72
open_ear_avg = 0.28
blink_thresh = open_ear_avg * BLINK_RATIO

# ============================================================================
# 5.  CALIBRATION — 16 points with pushed corners + auto range normalisation
# ============================================================================
# Corners pushed to 0.02/0.98 so the polynomial has data AT the screen edge.
# Middle rows stay at 0.35/0.65 for good interior coverage.
CAL_PTS_NORM = [
    (0.02, 0.02), (0.35, 0.02), (0.65, 0.02), (0.98, 0.02),
    (0.02, 0.35), (0.35, 0.35), (0.65, 0.35), (0.98, 0.35),
    (0.02, 0.65), (0.35, 0.65), (0.65, 0.65), (0.98, 0.65),
    (0.02, 0.98), (0.35, 0.98), (0.65, 0.98), (0.98, 0.98),
]
CAL_SETTLE_SEC = 1.0
CAL_COLLECT_SEC = 2.2
CAL_DOT_RADIUS = 10


class GazeCalibrator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.pt_idx = 0
        self.buf = []
        self.raw_pts = []   # collected raw (rx, ry) per point
        self.scr_pts = []   # corresponding screen targets
        self.model_x = None
        self.model_y = None
        self.ready = False
        self.t0 = None
        self.anchor_offset = np.zeros(2)
        # Auto-measured raw gaze range — corrects for compressed input range
        self.raw_x_min = self.raw_x_max = None
        self.raw_y_min = self.raw_y_max = None

    def start(self):
        self.reset()
        self.active = True
        self.t0 = time.time()
        total_t = len(CAL_PTS_NORM) * (CAL_SETTLE_SEC + CAL_COLLECT_SEC)
        print(
            f"[Cal] 16-point calibration — follow the dot (~{total_t:.0f} s total)")

    def feed(self, rx, ry):
        if not self.active:
            return
        if self.pt_idx >= len(CAL_PTS_NORM):
            self._finish()
            return
        elapsed = time.time() - self.t0
        if elapsed < CAL_SETTLE_SEC:
            return
        self.buf.append((rx, ry))
        if elapsed >= CAL_SETTLE_SEC + CAL_COLLECT_SEC:
            if len(self.buf) >= 5:
                bx = sorted(s[0] for s in self.buf)
                by = sorted(s[1] for s in self.buf)
                trim = max(1, len(bx) // 5)
                mx = float(np.median(bx[trim:-trim]))
                my = float(np.median(by[trim:-trim]))
                self.raw_pts.append((mx, my))
                self.scr_pts.append(CAL_PTS_NORM[self.pt_idx])
            self.buf = []
            self.pt_idx += 1
            self.t0 = time.time()
            if self.pt_idx < len(CAL_PTS_NORM):
                print(f"[Cal] Point {self.pt_idx+1}/{len(CAL_PTS_NORM)}")
            else:
                self._finish()

    # ------------------------------------------------------------------
    def _finish(self):
        self.active = False
        n = len(self.raw_pts)
        if n < 8:
            print("[Cal] Not enough points.")
            return

        # ── Auto measure actual raw gaze range ──────────────────────────
        # The raw projection typically only spans e.g. [-0.6, +0.6] not [-1,+1].
        # Normalise so that the full collected range maps to [-1, +1] before
        # feeding the polynomial — this is the key fix for edge clipping.
        all_rx = [p[0] for p in self.raw_pts]
        all_ry = [p[1] for p in self.raw_pts]
        self.raw_x_min, self.raw_x_max = min(all_rx), max(all_rx)
        self.raw_y_min, self.raw_y_max = min(all_ry), max(all_ry)
        print(f"[Cal] Raw range X:[{self.raw_x_min:.3f},{self.raw_x_max:.3f}]"
              f"  Y:[{self.raw_y_min:.3f},{self.raw_y_max:.3f}]")

        # Normalise raw points to [-1, +1] based on observed range
        def norm_raw(rx, ry):
            nx = 2.0 * (rx - self.raw_x_min) / \
                max(1e-6, self.raw_x_max - self.raw_x_min) - 1.0
            ny = 2.0 * (ry - self.raw_y_min) / \
                max(1e-6, self.raw_y_max - self.raw_y_min) - 1.0
            return nx, ny

        X_norm = np.array([norm_raw(p[0], p[1]) for p in self.raw_pts])
        Yx = np.array([p[0] for p in self.scr_pts])
        Yy = np.array([p[1] for p in self.scr_pts])

        if _SKL:
            best_x, best_y, best_res = None, None, 1e9
            for deg in (2, 3):
                if n < (deg+1)*(deg+2)//2:
                    continue
                px = Pipeline([('p', PolynomialFeatures(deg)),
                              ('r', Ridge(alpha=0.5))])
                py = Pipeline([('p', PolynomialFeatures(deg)),
                              ('r', Ridge(alpha=0.5))])
                px.fit(X_norm, Yx)
                py.fit(X_norm, Yy)
                res = (np.mean((px.predict(X_norm)-Yx)**2) +
                       np.mean((py.predict(X_norm)-Yy)**2))
                if res < best_res:
                    best_res, best_x, best_y = res, px, py
            self.model_x, self.model_y = best_x, best_y
            print(f"[Cal] Polynomial fitted (residual={best_res:.5f})")
        else:
            self.model_x = np.polyfit(X_norm[:, 0], Yx, 1)
            self.model_y = np.polyfit(X_norm[:, 1], Yy, 1)
            print("[Cal] Linear correction fitted.")

        self.anchor_offset = np.zeros(2)
        self.ready = True
        print(
            "[Cal] Done! Press R any time to re-anchor, E/Shift+E to tune edge stretch.")

    # ------------------------------------------------------------------
    def _to_norm(self, rx, ry):
        """Convert raw projection coords to the normalised range the model was trained on."""
        nx = 2.0*(rx - self.raw_x_min) / \
            max(1e-6, self.raw_x_max - self.raw_x_min) - 1.0
        ny = 2.0*(ry - self.raw_y_min) / \
            max(1e-6, self.raw_y_max - self.raw_y_min) - 1.0
        return nx, ny

    def _predict_norm(self, rx, ry):
        nx, ny = self._to_norm(rx, ry)
        X = np.array([[nx, ny]])
        if _SKL and hasattr(self.model_x, 'predict'):
            cx = float(self.model_x.predict(X)[0])
            cy = float(self.model_y.predict(X)[0])
        else:
            cx = float(np.polyval(self.model_x, nx))
            cy = float(np.polyval(self.model_y, ny))
        return cx, cy

    def correct(self, rx, ry):
        """Return screen-normalised [0,1] gaze position with edge stretch applied."""
        if not self.ready:
            return rx, ry
        cx, cy = self._predict_norm(rx, ry)
        cx += self.anchor_offset[0]
        cy += self.anchor_offset[1]
        # Edge stretch: expand range so corners are actually reachable.
        # Maps [0,1] through a gentle sigmoid-like expansion centred at 0.5.
        cx = self._stretch(cx)
        cy = self._stretch(cy)
        return float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0))

    @staticmethod
    def _stretch(v):
        """Linearly expand values outward from 0.5 by edge_stretch factor."""
        global edge_stretch
        return (v - 0.5) * edge_stretch + 0.5

    def recenter(self, rx, ry):
        if not self.ready:
            return
        pred_x, pred_y = self._predict_norm(rx, ry)
        self.anchor_offset = np.array([0.5 - pred_x, 0.5 - pred_y])
        print(f"[Cal] Re-anchored  offset={self.anchor_offset}")

    # ------------------------------------------------------------------
    def draw_dot(self, scr):
        if not self.active or self.pt_idx >= len(CAL_PTS_NORM):
            return
        nx, ny = CAL_PTS_NORM[self.pt_idx]
        sx = int(
            np.clip(nx * scr.shape[1], CAL_DOT_RADIUS+4, scr.shape[1]-CAL_DOT_RADIUS-4))
        sy = int(
            np.clip(ny * scr.shape[0], CAL_DOT_RADIUS+4, scr.shape[0]-CAL_DOT_RADIUS-4))
        elapsed = time.time() - self.t0

        if elapsed < CAL_SETTLE_SEC:
            pulse = int(CAL_DOT_RADIUS + 5 *
                        abs(math.sin(elapsed * math.pi * 3)))
            cv2.circle(scr, (sx, sy), pulse,         (120, 120, 120), 1)
            cv2.circle(scr, (sx, sy), CAL_DOT_RADIUS, (255, 255, 255), -1)
        else:
            prog = np.clip((elapsed - CAL_SETTLE_SEC) / CAL_COLLECT_SEC, 0, 1)
            cv2.circle(scr, (sx, sy), CAL_DOT_RADIUS+8, (40, 40, 40), -1)
            cv2.circle(scr, (sx, sy), CAL_DOT_RADIUS+8, (70, 70, 70), 1)
            cv2.ellipse(scr, (sx, sy), (CAL_DOT_RADIUS+8, CAL_DOT_RADIUS+8),
                        -90, 0, int(360*prog), (0, 220, 100), 3)
            cv2.circle(scr, (sx, sy), CAL_DOT_RADIUS, (255, 255, 255), -1)

        lbl_y = min(sy + CAL_DOT_RADIUS + 26, scr.shape[0] - 10)
        cv2.putText(scr, f"Point {self.pt_idx+1}/{len(CAL_PTS_NORM)}",
                    (sx - 55, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)


calibrator = GazeCalibrator()

# ============================================================================
# 6.  HEAD POSE + GAZE MODEL
# ============================================================================
NOSE_IDX = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 461,
            125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248]
IRIS_L_IDX = [468, 469, 470, 471]
IRIS_R_IDX = [473, 474, 475, 476]


def compute_head_pose(frame, lm, indices, ref_c):
    w, h = frame.shape[1], frame.shape[0]
    pts = np.array([[lm[i].x*w, lm[i].y*h, lm[i].z*w] for i in indices])
    ctr = np.mean(pts, axis=0)
    for i in indices:
        cv2.circle(frame, (int(lm[i].x*w), int(lm[i].y*h)), 2, (0, 200, 0), -1)
    cov = np.cov((pts-ctr).T)
    ev, evec = np.linalg.eigh(cov)
    evec = evec[:, np.argsort(-ev)]
    if np.linalg.det(evec) < 0:
        evec[:, 2] *= -1
    r = Rscipy.from_matrix(evec)
    ro, pi, ya = r.as_euler('zyx', degrees=False)
    R = Rscipy.from_euler('zyx', [ro, pi, ya]).as_matrix()
    if ref_c[0] is None:
        ref_c[0] = R.copy()
    else:
        Rr = ref_c[0]
        for i in range(3):
            if np.dot(R[:, i], Rr[:, i]) < 0:
                R[:, i] *= -1
    return ctr, R, pts


def iris_centre_3d(lm, idx_list, w, h):
    pts = np.array([[lm[i].x*w, lm[i].y*h, lm[i].z*w] for i in idx_list])
    return np.mean(pts, axis=0)


def iris_radius_px(lm, idx_list, w, h):
    pts = np.array([[lm[i].x*w, lm[i].y*h] for i in idx_list])
    c = np.mean(pts, axis=0)
    return float(np.mean(np.linalg.norm(pts - c, axis=1)))


virtual_monitor = dict(center=None, right=None, up=None,
                       width=0.0, height=0.0, ready=False)


def estimate_monitor_dist(lm, w, h):
    il = np.array([lm[468].x*w, lm[468].y*h])
    ir = np.array([lm[473].x*w, lm[473].y*h])
    ipd_px = np.linalg.norm(il - ir)
    return float(np.clip(ipd_px * 4.0, 150.0, 600.0)) if ipd_px > 10 else 250.0


def gaze_to_raw_norm(origin, direction, mon):
    if not mon["ready"]:
        return None
    nrm = np.cross(mon["right"], mon["up"])
    denom = np.dot(direction, nrm)
    if abs(denom) < 1e-6:
        return None
    t = np.dot(mon["center"]-origin, nrm) / denom
    if t < 0:
        return None
    pt = origin + t * direction
    vl = pt - mon["center"]
    rx = np.dot(vl, mon["right"])
    ry = np.dot(vl, mon["up"])
    return rx / (mon["width"]/2.0), ry / (mon["height"]/2.0)


# ============================================================================
# 7.  HAND TRACKING — palm-aware finger detection
# ============================================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
_hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.70,
    min_tracking_confidence=0.55,
)
TIP_IDS = [4, 8, 12, 16, 20]

# Force-orientation override: None=auto, "back"=back-of-hand, "palm"=palm-facing
_hand_orient_override = None   # toggled by T key


def _is_palm_facing(lm_list, handedness_label):
    """
    Determine if the palm is facing the camera using the wrist-to-index-MCP
    and wrist-to-pinky-MCP vectors and their cross product (normal vector).
    A positive z component of that cross product means palm faces camera
    (after accounting for left/right hand mirroring).

    lm_list: list of (idx, x, y) tuples
    handedness_label: "Left" or "Right" (MediaPipe label after horizontal flip)
    """
    if _hand_orient_override == "palm":
        return True
    if _hand_orient_override == "back":
        return False

    # Use landmarks: 0=wrist, 5=index MCP, 17=pinky MCP
    wrist = np.array([lm_list[0][1],  lm_list[0][2],  0], float)
    idx_mcp = np.array([lm_list[5][1],  lm_list[5][2],  0], float)
    pinky_mcp = np.array([lm_list[17][1], lm_list[17][2], 0], float)

    v1 = idx_mcp - wrist
    v2 = pinky_mcp - wrist
    cross_z = v1[0]*v2[1] - v1[1]*v2[0]   # z component of cross product

    # After image flip: "Right" hand in frame = user's right hand
    # Palm facing: cross_z > 0 for Right, < 0 for Left
    if handedness_label == "Right":
        return cross_z > 0
    else:
        return cross_z < 0


def _finger_states(lm_list, palm_facing, handedness_label):
    """
    Return [thumb, index, middle, ring, pinky] as 0/1 (extended=1).
    Handles both palm-facing and back-of-hand orientations, and both hands.
    """
    fingers = []

    # ── THUMB ───────────────────────────────────────────────────────────
    # Thumb tip = lm[4], thumb IP = lm[3], thumb MCP = lm[2]
    # For back-of-hand: for right hand thumb tip is to the right of IP when extended
    # For palm-facing:  it flips (thumb goes left for right hand when palm faces cam)
    # Use x-axis comparison but flip based on orientation+handedness
    tip_x = lm_list[4][1]
    ip_x = lm_list[3][1]

    if handedness_label == "Right":
        # Back of right hand: thumb extends LEFT (tip_x < ip_x in mirrored frame)
        # Palm of right hand: thumb extends RIGHT (tip_x > ip_x)
        thumb_up = (tip_x > ip_x) if palm_facing else (tip_x < ip_x)
    else:
        # Left hand: opposite
        thumb_up = (tip_x < ip_x) if palm_facing else (tip_x > ip_x)
    fingers.append(1 if thumb_up else 0)

    # ── FINGERS (index=8, middle=12, ring=16, pinky=20) ─────────────────
    # For both orientations: tip y < pip y means finger is extended upward.
    # This is consistent regardless of palm/back — the y-axis doesn't flip.
    for tip in TIP_IDS[1:]:
        fingers.append(1 if lm_list[tip][2] < lm_list[tip-2][2] else 0)

    return fingers


def process_hands(frame):
    """
    Returns (lmList, fingers, palm_facing, handedness_label).
    lmList: list of (idx, x, y)
    fingers: [thumb, index, middle, ring, pinky] 0/1
    palm_facing: bool
    handedness_label: "Left" or "Right"
    """
    res = _hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    lmList = []
    fingers = None
    palm_facing = False
    hand_label = "Right"

    if res.multi_hand_landmarks and res.multi_handedness:
        hl = res.multi_hand_landmarks[0]
        hness = res.multi_handedness[0]
        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        h, w = frame.shape[:2]
        for idx, lm in enumerate(hl.landmark):
            lmList.append((idx, int(lm.x*w), int(lm.y*h)))

        # MediaPipe labels are from the model's perspective (before our flip).
        # We flip the frame horizontally, so Left↔Right are swapped.
        # "Left" or "Right" pre-flip
        raw_label = hness.classification[0].label
        hand_label = "Left" if raw_label == "Right" else "Right"

        palm_facing = _is_palm_facing(lmList, hand_label)
        fingers = _finger_states(lmList, palm_facing, hand_label)

    return lmList, fingers, palm_facing, hand_label


def hand_dist(lm, a, b):
    return math.hypot(lm[a][1]-lm[b][1], lm[a][2]-lm[b][2])


# ============================================================================
# 8.  GESTURE DETECTOR — palm-side aware
# ============================================================================
GESTURE_HOLD = 0.40


class GestureDetector:
    LABELS = {
        "LEFT_CLICK":   "✌  Left Click",
        "RIGHT_CLICK":  "☝  Right Click",
        "DOUBLE_CLICK": "👌  Double Click",
        "SCROLL_UP":    "✋  Scroll Up",
        "SCROLL_DOWN":  "✊  Scroll Down",
        "PALM_PUSH":    "🖐  Scroll Up (palm)",
    }

    def __init__(self, hold_sec=GESTURE_HOLD):
        self.hold = hold_sec
        self.cur = None
        self.t0 = None
        self.fired = False
        self.prog = 0.0

    def classify(self, fingers, lmList, palm_facing):
        if not lmList or fingers is None:
            return None

        pinch = hand_dist(lmList, 4, 8)
        n_up = sum(fingers)

        # Pinch — works regardless of orientation
        if pinch < 38:
            return "DOUBLE_CLICK"

        if palm_facing:
            # ── PALM-SIDE GESTURES ───────────────────────────────────────
            # Open palm (4+ fingers up) → PALM_PUSH (scroll up)
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                return "PALM_PUSH"
            # Fist toward camera
            if n_up == 0:
                return "SCROLL_DOWN"
            # Index only (pointing) → RIGHT_CLICK  (same meaning, palm or back)
            if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
                return "RIGHT_CLICK"
            # Peace sign → LEFT_CLICK
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                return "LEFT_CLICK"
        else:
            # ── BACK-OF-HAND GESTURES ────────────────────────────────────
            if fingers == [0, 1, 1, 0, 0]:
                return "LEFT_CLICK"
            if fingers == [0, 1, 0, 0, 0]:
                return "RIGHT_CLICK"
            if n_up == 0:
                return "SCROLL_DOWN"
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                return "SCROLL_UP"

        return None

    def update(self, name):
        now = time.time()
        if name != self.cur:
            self.cur = name
            self.t0 = now if name else None
            self.fired = False
            self.prog = 0.0
        elif name and self.t0:
            held = now - self.t0
            self.prog = min(1.0, held / self.hold)
            if held >= self.hold:
                if not self.fired:
                    self.fired = True
                    return name
                if name in ("SCROLL_UP", "SCROLL_DOWN", "PALM_PUSH"):
                    return name
        return None

    def draw_hud(self, frame, x, y, palm_facing, hand_label, orient_override):
        orient_str = (f"[{orient_override}]" if orient_override
                      else f"{'PALM' if palm_facing else 'BACK'}")
        cv2.putText(frame, f"Hand:{hand_label} {orient_str}", (x, y-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 0), 1)
        if not self.cur:
            cv2.putText(frame, "no gesture", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
            return
        lbl = self.LABELS.get(self.cur, self.cur)
        col = (0, 230, 100) if self.fired else (0, 180, 255)
        cv2.putText(frame, lbl, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        bw, filled = 180, int(180*self.prog)
        cv2.rectangle(frame, (x, y+6), (x+bw, y+20), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y+6), (x+filled, y+20), col, -1)
        cv2.rectangle(frame, (x, y+6), (x+bw, y+20), (120, 120, 120), 1)


gesture_det = GestureDetector()
_SCROLL_AMT = 5
_last_scroll_t = 0.0
_SCROLL_REPEAT = 0.12


def execute_gesture(act, repeat=False):
    global _last_scroll_t
    now = time.time()
    scroll_acts = ("SCROLL_UP", "SCROLL_DOWN", "PALM_PUSH")
    if act in scroll_acts:
        if now - _last_scroll_t < _SCROLL_REPEAT:
            return
        _last_scroll_t = now
        amt = -_SCROLL_AMT if act == "SCROLL_DOWN" else _SCROLL_AMT
        pyautogui.scroll(amt)
        print(f"[Hand] {act}")
    elif not repeat:
        if act == "LEFT_CLICK":
            pyautogui.click(button='left')
        elif act == "RIGHT_CLICK":
            pyautogui.click(button='right')
        elif act == "DOUBLE_CLICK":
            pyautogui.doubleClick()
        print(f"[Hand] {act}")


# ============================================================================
# 9.  EYE GESTURES
# ============================================================================
class EyeGestures:
    COOLDOWN = 0.6

    def __init__(self):
        self.both_t = None
        self.last = 0

    def update(self, le, re, thresh):
        now = time.time()
        if now - self.last < self.COOLDOWN:
            return None
        lb, rb = le < thresh, re < thresh
        act = None
        if lb and rb:
            if self.both_t is None:
                self.both_t = now
            elif now - self.both_t >= 0.6:
                act = "LEFT_CLICK"
                self.both_t = None
        else:
            self.both_t = None
            if lb and not rb:
                act = "RIGHT_CLICK"
            elif rb and not lb:
                act = "DOUBLE_CLICK"
        if act:
            self.last = now
            print(f"[Eye] {act}")
        return act


eye_gest = EyeGestures()

# ============================================================================
# 10.  SMOOTH MOUSE THREAD
# ============================================================================
mouse_on = False
_m_lock = threading.Lock()
_m_target = [MONITOR_W//2, MONITOR_H//2]
_m_cur = [MONITOR_W//2, MONITOR_H//2]
last_toggle = 0


def _mouse_thread():
    while True:
        if mouse_on:
            with _m_lock:
                tx, ty = _m_target
            cx, cy = _m_cur
            dx, dy = tx-cx, ty-cy
            dist = math.hypot(dx, dy)
            if dist > DEADZONE_PX:
                step = min(dist, max(18, dist*0.55))
                _m_cur[0] = int(cx + dx/dist*step)
                _m_cur[1] = int(cy + dy/dist*step)
                pyautogui.moveTo(_m_cur[0], _m_cur[1])
        time.sleep(0.008)


threading.Thread(target=_mouse_thread, daemon=True).start()


def set_target(sx, sy):
    with _m_lock:
        _m_target[0] = sx
        _m_target[1] = sy


# ============================================================================
# 11.  MEDIAPIPE FACE MESH
# ============================================================================
mp_fm = mp.solutions.face_mesh
face_mesh = mp_fm.FaceMesh(static_image_mode=False, max_num_faces=1,
                           refine_landmarks=True,
                           min_detection_confidence=0.6,
                           min_tracking_confidence=0.5)

# ============================================================================
# 12.  DEBUG 3-D VIEW
# ============================================================================
orbit_yaw, orbit_pitch, orbit_radius = -2.6, 0.0, 600.0


def render_debug(h, w, ctr, sl, sr, gdir, mon):
    dbg = np.zeros((h, w, 3), np.uint8)
    if ctr is None:
        cv2.imshow("3D Debug", dbg)
        return
    f = _focal_px(w)
    cp = ctr + _rot_y(orbit_yaw) @ (_rot_x(orbit_pitch)
                                    @ np.array([0, 0, orbit_radius]))
    fwd = _normalize(ctr-cp)
    rt = _normalize(np.cross(fwd, np.array([0, -1, 0])))
    up = _normalize(np.cross(rt, fwd))
    V = np.stack([rt, up, fwd])

    def proj(P):
        Pc = V@(P-cp)
        if Pc[2] <= 1:
            return None
        return (int(f*Pc[0]/Pc[2]+w/2), int(-f*Pc[1]/Pc[2]+h/2))
    hc = proj(ctr)
    if hc:
        cv2.circle(dbg, hc, 5, (255, 0, 255), -1)
    for sp in [sl, sr]:
        if sp is not None:
            ep = proj(sp)
            if ep:
                cv2.circle(dbg, ep, 10, (0, 220, 220), 2)
    if gdir is not None and sl is not None and sr is not None:
        s, e = (sl+sr)/2, (sl+sr)/2+gdir*500
        p1, p2 = proj(s), proj(e)
        if p1 and p2:
            cv2.line(dbg, p1, p2, (0, 80, 255), 2)
    if mon["ready"]:
        c = mon["center"]
        r = mon["right"]*(mon["width"]/2)
        u = mon["up"]*(mon["height"]/2)
        corners = [c-r+u, c+r+u, c+r-u, c-r-u]
        pts2 = [proj(p) for p in corners]
        if all(pts2):
            for i in range(4):
                cv2.line(dbg, pts2[i], pts2[(i+1) % 4], (0, 255, 0), 2)
    cv2.putText(dbg, f"Mode:{control_mode.upper()}  ES:{edge_stretch:.2f}",
                (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.imshow("3D Debug", dbg)

# ============================================================================
# 13.  KEY HANDLER
# ============================================================================


def handle_key(key):
    global orbit_yaw, orbit_pitch, orbit_radius
    global bias_x, bias_y, sensitivity_x, sensitivity_y, invert_x, invert_y
    global control_mode, mouse_on, last_toggle, open_ear_avg, blink_thresh
    global edge_stretch, _hand_orient_override

    if key == ord('a'):
        orbit_yaw -= 0.05
    elif key == ord('d'):
        orbit_yaw += 0.05
    elif key == ord('w'):
        orbit_pitch += 0.05
    elif key == ord('s') and not calibrator.active:
        orbit_pitch -= 0.05
    elif key == ord('z'):
        orbit_radius += 20
    elif key == ord('x'):
        orbit_radius -= 20
    elif key == ord('i'):
        bias_y += 0.02
    elif key == ord('k'):
        bias_y -= 0.02
    elif key == ord('j'):
        bias_x -= 0.02
    elif key == ord('l'):
        bias_x += 0.02
    elif key == ord('='):
        sensitivity_x += 0.1
        sensitivity_y += 0.1
    elif key == ord('-'):
        sensitivity_x = max(0.1, sensitivity_x-0.1)
        sensitivity_y = max(0.1, sensitivity_y-0.1)
    elif key == ord('['):
        invert_x = not invert_x
    elif key == ord(']'):
        invert_y = not invert_y
    elif key == ord('f'):
        gaze_filter.set_alpha(gaze_filter.base_alpha - 0.02)
        print(f"[Filter] alpha={gaze_filter.base_alpha:.2f}")
    elif key == ord('g'):
        gaze_filter.set_alpha(gaze_filter.base_alpha + 0.02)
        print(f"[Filter] alpha={gaze_filter.base_alpha:.2f}")
    elif key == ord('e'):
        edge_stretch = round(min(1.5, edge_stretch + 0.02), 3)
        print(f"[Edge] stretch={edge_stretch:.2f}")
    elif key == ord('E'):
        edge_stretch = round(max(0.8, edge_stretch - 0.02), 3)
        print(f"[Edge] stretch={edge_stretch:.2f}")
    elif key == ord('t'):
        # Cycle: auto → palm → back → auto
        if _hand_orient_override is None:
            _hand_orient_override = "palm"
        elif _hand_orient_override == "palm":
            _hand_orient_override = "back"
        else:
            _hand_orient_override = None
        print(f"[Hand] Orient override: {_hand_orient_override or 'auto'}")
    elif key == ord('1'):
        control_mode = "eye"
        print("[Mode] Eye-only")
    elif key == ord('2'):
        control_mode = "hand"
        print("[Mode] Hand-only")
    elif key == ord('3'):
        control_mode = "dual"
        print("[Mode] Dual")
    elif key == ord('m'):
        if time.time()-last_toggle > 0.5:
            mouse_on = not mouse_on
            last_toggle = time.time()
            print(f"[Mouse] {'ON' if mouse_on else 'OFF'}")


# ============================================================================
# 14.  CAMERA
# ============================================================================
cap = None
for idx in [1, 2, 3, 0]:
    tmp = cv2.VideoCapture(idx)
    if tmp.isOpened():
        ret, _ = tmp.read()
        if ret:
            cap = tmp
            print(f"[Camera] Index {idx}")
            break
        tmp.release()
if cap is None:
    print("[Error] No camera found!")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FPS, 60)
FW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[Camera] {FW}x{FH} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")

# ============================================================================
# 15.  STATE
# ============================================================================
R_ref = [None]
sph_locked = False
sph_l_off = sph_r_off = None
cal_sc_l = cal_sc_r = None
_cal_iris_r_l = _cal_iris_r_r = None

_lm = _R = _iris_l = _iris_r = _nose_pts = None
_avg_gaze = _gaze_orig = _head_ctr = _sph_l = _sph_r = None
l_ear = r_ear = 0.28

# hand state carried across loop iterations for HUD
_palm_facing_last = False
_hand_label_last = "Right"

# ============================================================================
# 16.  MAIN LOOP
# ============================================================================
print("\n" + "="*62)
print("  TRACKER v4 — S → C → M   |   R=reanchor  E/Shift+E=edges")
print("  T=toggle palm/back override for hand   Q=quit")
print("="*62 + "\n")

CAL_WIN = "Calibration"

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.01)
        continue
    frame = cv2.flip(frame, 1)
    now = time.time()

    _lm = _R = _iris_l = _iris_r = _nose_pts = None
    _head_ctr = _sph_l = _sph_r = _avg_gaze = _gaze_orig = None
    l_ear = r_ear = 0.28

    # =========================================================================
    # A.  FACE MESH + GAZE
    # =========================================================================
    fm_res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if fm_res.multi_face_landmarks:
        _lm = fm_res.multi_face_landmarks[0].landmark
        w, h = FW, FH

        _head_ctr, _R, _nose_pts = compute_head_pose(
            frame, _lm, NOSE_IDX, R_ref)
        _iris_l = iris_centre_3d(_lm, IRIS_L_IDX, w, h)
        _iris_r = iris_centre_3d(_lm, IRIS_R_IDX, w, h)
        ir_rad_l = iris_radius_px(_lm, IRIS_L_IDX, w, h)
        ir_rad_r = iris_radius_px(_lm, IRIS_R_IDX, w, h)

        def _pt(i): return (_lm[i].x*w, _lm[i].y*h)
        l_ear = ear([_pt(i) for i in LEFT_EYE_IDX])
        r_ear = ear([_pt(i) for i in RIGHT_EYE_IDX])

        if sph_locked:
            sc = compute_scale(_nose_pts)
            rl = sc / cal_sc_l
            rr = sc / cal_sc_r
            _sph_l = _head_ctr + _R @ (sph_l_off * rl)
            _sph_r = _head_ctr + _R @ (sph_r_off * rr)
            cv2.circle(frame, (int(_sph_l[0]), int(
                _sph_l[1])), int(20*rl), (255, 80, 0), 1)
            cv2.circle(frame, (int(_sph_r[0]), int(
                _sph_r[1])), int(20*rr), (255, 80, 0), 1)

            dir_l = _normalize(_iris_l - _sph_l)
            dir_r = _normalize(_iris_r - _sph_r)

            w_l = max(0.01, l_ear)
            w_r = max(0.01, r_ear)
            if _cal_iris_r_l and _cal_iris_r_r:
                w_l *= (_cal_iris_r_l / max(1.0, ir_rad_l))
                w_r *= (_cal_iris_r_r / max(1.0, ir_rad_r))
            w_l /= (w_l + w_r)
            w_r = 1.0 - w_l
            raw_g = _normalize(dir_l * w_l + dir_r * w_r)
            raw_g_sm = _normalize(gaze_filter.update(raw_g))
            _avg_gaze = raw_g_sm
            _gaze_orig = (_sph_l + _sph_r) / 2.0

            res = gaze_to_raw_norm(_gaze_orig, _avg_gaze, virtual_monitor)
            if res:
                raw_nx, raw_ny = res

                if calibrator.active:
                    calibrator.feed(raw_nx, raw_ny)

                if calibrator.ready:
                    cx, cy = calibrator.correct(raw_nx, raw_ny)
                else:
                    cx = np.clip(
                        (raw_nx * sensitivity_x + bias_x + 1) / 2.0, 0, 1)
                    cy = np.clip(
                        (1-(raw_ny * sensitivity_y + bias_y)) / 1.0, 0, 1)

                if invert_x:
                    cx = 1 - cx
                if invert_y:
                    cy = 1 - cy

                sx = int(np.clip(cx * MONITOR_W, 0, MONITOR_W-1))
                sy = int(np.clip(cy * MONITOR_H, 0, MONITOR_H-1))

                if control_mode in ("eye", "dual") and mouse_on:
                    set_target(sx, sy)
                write_pos(sx, sy)

                gx = int(cx * FW)
                gy = int(cy * FH)
                cv2.circle(frame, (gx, gy), 12, (0, 60, 255), -1)
                cv2.circle(frame, (gx, gy), 14, (255, 255, 255), 1)
                cv2.putText(frame, f"Gaze ({sx},{sy})", (20, 108),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
                cv2.putText(frame, f"raw:({raw_nx:.2f},{raw_ny:.2f})"
                            f"  ES:{edge_stretch:.2f}",
                            (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 0), 1)
        else:
            cv2.circle(frame, (int(_iris_l[0]), int(
                _iris_l[1])), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(_iris_r[0]), int(
                _iris_r[1])), 5, (0, 0, 255), -1)

        if control_mode == "eye" and sph_locked:
            act = eye_gest.update(l_ear, r_ear, blink_thresh)
            if act:
                execute_gesture(act)

        cv2.putText(frame,
                    f"L-EAR:{l_ear:.2f}  R-EAR:{r_ear:.2f}  thr:{blink_thresh:.2f}",
                    (20, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 0), 1)

    # =========================================================================
    # B.  HAND TRACKING
    # =========================================================================
    if control_mode in ("hand", "dual"):
        lmList, fingers, palm_facing, hand_label = process_hands(frame)
        _palm_facing_last = palm_facing
        _hand_label_last = hand_label

        if lmList and fingers is not None:
            tip_x, tip_y = lmList[8][1], lmList[8][2]
            hand_drives = (control_mode == "hand" or
                           (control_mode == "dual" and not sph_locked))
            if hand_drives and mouse_on:
                set_target(int(np.interp(tip_x, (0, FW), (0, MONITOR_W))),
                           int(np.interp(tip_y, (0, FH), (0, MONITOR_H))))

            g_name = gesture_det.classify(fingers, lmList, palm_facing)
            g_fire = gesture_det.update(g_name)
            if g_fire:
                scroll_acts = ("SCROLL_UP", "SCROLL_DOWN", "PALM_PUSH")
                execute_gesture(g_fire,
                                repeat=(g_fire in scroll_acts and gesture_det.fired))
            gesture_det.draw_hud(frame, 20, 210, palm_facing, hand_label,
                                 _hand_orient_override)
            # Show finger states for debugging
            fstr = "".join(str(f) for f in fingers) if fingers else "-----"
            cv2.putText(frame, f"fingers:{fstr}", (20, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 200), 1)
        else:
            gesture_det.update(None)
            gesture_det.draw_hud(frame, 20, 210, False,
                                 "?", _hand_orient_override)

    # =========================================================================
    # C.  CALIBRATION FULLSCREEN
    # =========================================================================
    if calibrator.active:
        cal_scr = np.zeros((MONITOR_H, MONITOR_W, 3), np.uint8)
        cv2.putText(cal_scr, "CALIBRATION — keep head still, follow the dot",
                    (MONITOR_W//2-310, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
        cv2.putText(cal_scr, "Pulse = settling. Green arc = collecting.",
                    (MONITOR_W//2-230, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
        calibrator.draw_dot(cal_scr)
        cv2.namedWindow(CAL_WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            CAL_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(CAL_WIN, cal_scr)
    else:
        try:
            cv2.destroyWindow(CAL_WIN)
        except:
            pass

    # =========================================================================
    # D.  HUD
    # =========================================================================
    ms_col = (0, 255, 80) if mouse_on else (80, 80, 80)
    cal_col = (0, 255, 120) if calibrator.ready else (60, 60, 200)
    cv2.putText(frame,
                f"Mouse:{'ON' if mouse_on else 'OFF'}  Mode:{control_mode.upper()}(1/2/3) [M]",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ms_col, 2)
    cv2.putText(frame,
                f"Bias X:{bias_x:.2f} Y:{bias_y:.2f}  Sens:{sensitivity_x:.1f}"
                f"  a:{gaze_filter.base_alpha:.2f}  ES:{edge_stretch:.2f}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 0), 1)
    eye_st = "EYE:OK" if sph_locked else "EYE:-- (S→C)"
    mon_st = "MON:OK" if virtual_monitor["ready"] else "MON:-- (S)"
    gc_st = "GCAL:OK" if calibrator.ready else "GCAL:-- (C)"
    cv2.putText(frame, f"{eye_st}  {mon_st}  {gc_st}",
                (20, FH-12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cal_col, 1)
    tips = ["S=Screen(FIRST)", "C=Cal16pts", "R=ReAnchor",
            "E/shiftE=EdgeStretch", "T=PalmOverride",
            "M=Mouse  1/2/3=mode",
            "F/G=filter  =/- sens", "I/K/J/L=bias"]
    for ti, t in enumerate(tips):
        cv2.putText(frame, t, (FW-200, 20+ti*17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1)

    render_debug(FH, FW, _head_ctr, _sph_l, _sph_r, _avg_gaze, virtual_monitor)
    cv2.imshow("Tracker", frame)

    # =========================================================================
    # E.  KEY HANDLING
    # =========================================================================
    key = cv2.waitKey(1) & 0xFF
    handle_key(key)

    if key == ord('q'):
        break

    elif key == ord('c'):
        if _lm is not None and _nose_pts is not None:
            sc = compute_scale(_nose_pts)
            cd = _R.T @ np.array([0, 0, 1])
            sph_l_off = _R.T @ (_iris_l - _head_ctr) + 20.0*cd
            sph_r_off = _R.T @ (_iris_r - _head_ctr) + 20.0*cd
            cal_sc_l = cal_sc_r = sc
            _cal_iris_r_l = iris_radius_px(_lm, IRIS_L_IDX, FW, FH)
            _cal_iris_r_r = iris_radius_px(_lm, IRIS_R_IDX, FW, FH)
            sph_locked = True
            gaze_filter.reset()
            print("[Cal] Eye spheres locked.")
            if virtual_monitor["ready"]:
                calibrator.start()
            else:
                print("[Cal] Press S first (look at screen centre), then C again.")
        else:
            print("[Cal] No face detected.")

    elif key == ord('s') and sph_locked and _avg_gaze is not None:
        dist = estimate_monitor_dist(_lm, FW, FH) if _lm else 250.0
        center = _gaze_orig + _avg_gaze * dist
        normal = -_avg_gaze
        up_w = np.array([0, -1, 0], float)
        if abs(np.dot(normal, up_w)) > 0.99:
            up_w = np.array([1, 0, 0])
        rt = _normalize(np.cross(up_w, normal))
        up = _normalize(np.cross(normal, rt))
        virtual_monitor.update(center=center, right=rt, up=up,
                               width=300.0, height=200.0, ready=True)
        open_ear_avg = (l_ear + r_ear) / 2.0
        blink_thresh = open_ear_avg * BLINK_RATIO
        print(
            f"[Cal] Screen plane set (dist≈{dist:.0f}). Blink thr={blink_thresh:.3f}")
        print("[Cal] Now press C for 16-point calibration.")

    elif key == ord('r'):
        if calibrator.ready and _avg_gaze is not None:
            res = gaze_to_raw_norm(_gaze_orig, _avg_gaze, virtual_monitor)
            if res:
                calibrator.recenter(res[0], res[1])
        else:
            print("[Cal] Calibrate first, then use R to re-anchor.")

# ============================================================================
# CLEANUP
# ============================================================================
cap.release()
cv2.destroyAllWindows()
print("[Done]")
