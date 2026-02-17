import cv2
import numpy as np
import os
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import threading

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
pyautogui.FAILSAFE = False 

# Tuning Variables (Adjust these with keys)
filter_length = 15       # Higher = Smoother
sensitivity_x = 1.0 
sensitivity_y = 1.0
bias_x = 0.0             # Horizontal correction
bias_y = 0.0             # Vertical correction
invert_x = False
invert_y = False

# File Path for Mac
home_dir = os.path.expanduser("~")
screen_position_file = os.path.join(home_dir, "Documents", "screen_position.txt")

# ==========================================
# 2. HELPER FUNCTIONS (Defined Top-Level)
# ==========================================
def write_screen_position(x, y):
    """Writes the current gaze coordinates to a file."""
    try:
        with open(screen_position_file, 'w') as f:
            f.write(f"{x},{y}\n")
    except Exception:
        pass

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _focal_px(width, fov_deg):
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)

def _rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)

def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

def compute_scale(points_3d):
    n = len(points_3d)
    total = 0; count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(points_3d[i] - points_3d[j])
            count += 1
    return total / count if count > 0 else 1.0

# ==========================================
# 3. CORE LOGIC (Head & Gaze)
# ==========================================
def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, color=(0, 255, 0), size=80):
    w, h = frame.shape[1], frame.shape[0]
    points_3d = np.array([[face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w] for i in indices])
    center = np.mean(points_3d, axis=0)
    
    # Draw landmarks
    for i in indices: 
        cv2.circle(frame, (int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)), 2, color, -1)
    
    # PCA for Orientation
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]
    if np.linalg.det(eigvecs) < 0: eigvecs[:, 2] *= -1
    
    # Stabilize
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    if ref_matrix_container[0] is None: ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0: R_final[:, i] *= -1

    return center, R_final, points_3d

def get_gaze_screen_pos(gaze_origin, gaze_dir, mon):
    if not mon["ready"]: return None
    
    normal = np.cross(mon["right"], mon["up"])
    denom = np.dot(gaze_dir, normal)
    if abs(denom) < 1e-6: return None
    
    t = np.dot(mon["center"] - gaze_origin, normal) / denom
    if t < 0: return None # intersection behind eye
        
    intersect_pt = gaze_origin + t * gaze_dir
    v_local = intersect_pt - mon["center"]
    
    x_local = np.dot(v_local, mon["right"])
    y_local = np.dot(v_local, mon["up"])
    
    # Apply Manual Tuning
    x_norm = x_local / (mon["width"] / 2.0 * sensitivity_x)
    y_norm = y_local / (mon["height"] / 2.0 * sensitivity_y)
    
    # Apply Bias
    x_norm += bias_x
    y_norm += bias_y
    
    # Inversions
    if invert_x: x_norm = -x_norm
    if invert_y: y_norm = -y_norm
    
    # Map to Screen
    sx = int((x_norm + 1) * 0.5 * MONITOR_WIDTH)
    sy = int((1 - y_norm) * 0.5 * MONITOR_HEIGHT) # Y is down on screen
    
    sx = max(0, min(sx, MONITOR_WIDTH))
    sy = max(0, min(sy, MONITOR_HEIGHT))
    
    return sx, sy, x_norm, y_norm

def update_orbit_from_keys(key_code):
    global orbit_yaw, orbit_pitch, orbit_radius, bias_x, bias_y, filter_length, sensitivity_x, sensitivity_y, invert_x, invert_y
    
    # Camera Orbit
    if key_code == ord('a'): orbit_yaw -= 0.05
    elif key_code == ord('d'): orbit_yaw += 0.05
    elif key_code == ord('w'): orbit_pitch += 0.05
    elif key_code == ord('s'): orbit_pitch -= 0.05 # Only works if not calibrating
    elif key_code == ord('z'): orbit_radius += 20
    elif key_code == ord('x'): orbit_radius -= 20
    
    # Bias Correction (Arrow keys mapped to I/J/K/L)
    elif key_code == ord('i'): bias_y += 0.02
    elif key_code == ord('k'): bias_y -= 0.02
    elif key_code == ord('j'): bias_x -= 0.02
    elif key_code == ord('l'): bias_x += 0.02
    
    # Sensitivity
    elif key_code == ord('='): sensitivity_x += 0.1; sensitivity_y += 0.1
    elif key_code == ord('-'): sensitivity_x = max(0.1, sensitivity_x - 0.1); sensitivity_y = max(0.1, sensitivity_y - 0.1)
    
    # Inversion
    elif key_code == ord('['): invert_x = not invert_x
    elif key_code == ord(']'): invert_y = not invert_y

def render_debug_view(h, w, head_center, eyes_l, eyes_r, gaze_dir, mon):
    debug = np.zeros((h, w, 3), dtype=np.uint8)
    if head_center is None: return debug
    
    # Simple projection for debug view
    f_px = _focal_px(w, 50.0)
    
    # Camera transform
    cam_pos = head_center + _rot_y(orbit_yaw) @ (_rot_x(orbit_pitch) @ np.array([0, 0, orbit_radius]))
    fwd = _normalize(head_center - cam_pos)
    right = _normalize(np.cross(fwd, np.array([0, -1, 0])))
    up = _normalize(np.cross(right, fwd))
    V = np.stack([right, up, fwd])
    
    def proj(P):
        Pc = V @ (P - cam_pos)
        if Pc[2] <= 1: return None
        x = f_px * (Pc[0]/Pc[2]) + w/2
        y = -f_px * (Pc[1]/Pc[2]) + h/2
        return (int(x), int(y))

    # Draw Head
    hc = proj(head_center)
    if hc: cv2.circle(debug, hc, 5, (255, 0, 255), -1)
    
    # Draw Eyes
    if eyes_l is not None:
        el = proj(eyes_l)
        if el: cv2.circle(debug, el, 8, (255, 255, 0), 1)
    if eyes_r is not None:
        er = proj(eyes_r)
        if er: cv2.circle(debug, er, 8, (255, 255, 0), 1)
        
    # Draw Gaze
    if gaze_dir is not None and eyes_l is not None:
        start = (eyes_l + eyes_r) / 2
        end = start + gaze_dir * 400
        p1, p2 = proj(start), proj(end)
        if p1 and p2: cv2.line(debug, p1, p2, (0, 0, 255), 2)
        
    # Draw Monitor
    if mon["ready"]:
        c = mon["center"]
        r = mon["right"] * (mon["width"]/2)
        u = mon["up"] * (mon["height"]/2)
        corners = [c - r + u, c + r + u, c + r - u, c - r - u]
        pts = [proj(p) for p in corners]
        if all(pts):
            for i in range(4):
                cv2.line(debug, pts[i], pts[(i+1)%4], (0, 255, 0), 2)

    cv2.imshow("3D Debug", debug)

# ==========================================
# 4. INITIALIZATION
# ==========================================
mouse_control_enabled = False
mouse_lock = threading.Lock()
mouse_target = [CENTER_X, CENTER_Y]
last_mouse_pos = [CENTER_X, CENTER_Y]
combined_gaze_directions = deque(maxlen=filter_length)

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# State
orbit_yaw = -2.6; orbit_pitch = 0.0; orbit_radius = 600.0
virtual_monitor = {"center": None, "right": None, "up": None, "width": 0, "height": 0, "ready": False}
R_ref_nose = [None]; calibration_nose_scale = None
left_sphere_locked = False; right_sphere_locked = False
left_sphere_local_offset = None; right_sphere_local_offset = None
left_calibration_nose_scale = None; right_calibration_nose_scale = None
last_toggle_time = 0 

# Camera Detection Loop
cap = None
# Try indices 1, 2, 3 (External) then 0 (Internal)
for idx in [1, 2, 3, 0]:
    temp_cap = cv2.VideoCapture(idx)
    if temp_cap.isOpened():
        # Check if it actually returns a frame
        ret, frame = temp_cap.read()
        if ret:
            print(f"[Camera] Found working camera at Index {idx}")
            cap = temp_cap
            break
        else:
            temp_cap.release()

if cap is None:
    print("[Error] No working cameras found!")
    exit()

# Set Resolution
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 461, 125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248]

def mouse_mover():
    while True:
        if mouse_control_enabled:
            with mouse_lock:
                tx, ty = mouse_target
                # Deadzone
                dx = tx - last_mouse_pos[0]
                dy = ty - last_mouse_pos[1]
                if math.sqrt(dx*dx + dy*dy) > 15: # 15px deadzone
                    pyautogui.moveTo(tx, ty)
                    last_mouse_pos[0] = tx
                    last_mouse_pos[1] = ty
        time.sleep(0.01)

threading.Thread(target=mouse_mover, daemon=True).start()

# ==========================================
# 5. MAIN LOOP
# ==========================================
print("--- CONTROLS ---")
print("C: Calibrate Eyes (Look at Camera)")
print("S: Calibrate Screen (Look at Center of Monitor)")
print("M: Toggle Mouse Control")
print("Arrows (I/J/K/L): Fix Bias (Up/Left/Down/Right)")
print("- / =: Adjust Sensitivity")
print("[ / ]: Invert X / Y")
print("Q: Quit")

while True:
    ret, frame = cap.read()
    if not ret: time.sleep(0.01); continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    head_center = None
    sphere_l = None
    sphere_r = None
    iris_l_3d = None
    iris_r_3d = None
    avg_gaze = None

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        
        # Draw Landmarks
        head_center, R_final, nose_points = compute_and_draw_coordinate_box(frame, lm, nose_indices, R_ref_nose)
        
        # Iris Points
        l_iris_lm = lm[468]
        r_iris_lm = lm[473]
        iris_l_3d = np.array([l_iris_lm.x * w, l_iris_lm.y * h, l_iris_lm.z * w])
        iris_r_3d = np.array([r_iris_lm.x * w, r_iris_lm.y * h, r_iris_lm.z * w])
        
        # Calculate Eye Spheres
        if not left_sphere_locked:
            cv2.circle(frame, (int(iris_l_3d[0]), int(iris_l_3d[1])), 5, (0,0,255), -1)
        else:
            curr_scale = compute_scale(nose_points)
            # Left
            ratio_l = curr_scale / left_calibration_nose_scale
            sphere_l = head_center + R_final @ (left_sphere_local_offset * ratio_l)
            cv2.circle(frame, (int(sphere_l[0]), int(sphere_l[1])), int(20*ratio_l), (255,0,0), 1)
            # Right
            ratio_r = curr_scale / right_calibration_nose_scale
            sphere_r = head_center + R_final @ (right_sphere_local_offset * ratio_r)
            cv2.circle(frame, (int(sphere_r[0]), int(sphere_r[1])), int(20*ratio_r), (255,0,0), 1)
            
            # Gaze
            dir_l = _normalize(iris_l_3d - sphere_l)
            dir_r = _normalize(iris_r_3d - sphere_r)
            raw_gaze = _normalize(dir_l + dir_r)
            
            # Filter
            if combined_gaze_directions.maxlen != filter_length:
                combined_gaze_directions = deque(combined_gaze_directions, maxlen=filter_length)
            combined_gaze_directions.append(raw_gaze)
            avg_gaze = _normalize(np.mean(combined_gaze_directions, axis=0))
            
            # Intersection
            gaze_origin = (sphere_l + sphere_r) / 2.0
            screen_res = get_gaze_screen_pos(gaze_origin, avg_gaze, virtual_monitor)
            
            if screen_res:
                sx, sy, nx, ny = screen_res
                if mouse_control_enabled:
                    with mouse_lock: mouse_target = [sx, sy]
                
                write_screen_position(sx, sy)
                
                # UI Overlay
                color = (0, 0, 255) if abs(nx)>1 or abs(ny)>1 else (0, 255, 0)
                cv2.putText(frame, f"Gaze: {nx:.2f}, {ny:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # UI Text
    status = "ON" if mouse_control_enabled else "OFF"
    cv2.putText(frame, f"Mouse: {status} (M)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Bias X:{bias_x:.2f} Y:{bias_y:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    render_debug_view(h, w, head_center, sphere_l, sphere_r, avg_gaze, virtual_monitor)
    cv2.imshow("Tracker", frame)
    
    key = cv2.waitKey(1) & 0xFF
    update_orbit_from_keys(key)
    
    if key == ord('q'): break
    elif key == ord('m'):
        if time.time() - last_toggle_time > 0.5:
            mouse_control_enabled = not mouse_control_enabled
            last_toggle_time = time.time()
            
    elif key == ord('c') and results.multi_face_landmarks:
        # Calibrate Eyes
        curr_scale = compute_scale(nose_points)
        cam_dir = R_final.T @ np.array([0,0,1])
        
        left_sphere_local_offset = R_final.T @ (iris_l_3d - head_center) + 20.0 * cam_dir
        left_calibration_nose_scale = curr_scale
        left_sphere_locked = True
        
        right_sphere_local_offset = R_final.T @ (iris_r_3d - head_center) + 20.0 * cam_dir
        right_calibration_nose_scale = curr_scale
        right_sphere_locked = True
        print("[System] Eyes Calibrated.")
        
    elif key == ord('s') and left_sphere_locked:
        # Calibrate Monitor
        if avg_gaze is not None:
            dist = 50.0 * 5.0 # Approx 50cm
            center = gaze_origin + avg_gaze * dist
            normal = -avg_gaze
            up_w = np.array([0, -1, 0], dtype=float)
            if abs(np.dot(normal, up_w)) > 0.99: up_w = np.array([1, 0, 0])
            
            right = _normalize(np.cross(up_w, normal))
            up = _normalize(np.cross(normal, right))
            
            virtual_monitor = {
                "center": center, "right": right, "up": up,
                "width": 60.0 * 5.0, "height": 40.0 * 5.0, "ready": True
            }
            print("[System] Monitor Calibrated.")

cap.release()
cv2.destroyAllWindows()