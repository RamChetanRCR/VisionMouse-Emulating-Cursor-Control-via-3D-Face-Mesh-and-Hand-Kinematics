# VisionMouse 🖱️👁️🤚

A hands-free cursor control system that emulates full mouse functionality 
using real-time 3D eye gaze tracking and hand gesture recognition — 
no physical mouse required.

## Demo
Control your cursor by looking at the screen. Click, scroll, and 
right-click using hand gestures or eye blinks.

## Tech Stack
- **Computer Vision:** MediaPipe Face Mesh (468 landmarks), MediaPipe Hands
- **Eye Tracking:** Iris centre detection, Eye Aspect Ratio (EAR) for blink detection
- **Gaze Filtering:** Adaptive Double-EMA filter for smooth cursor movement
- **Calibration:** 16-point polynomial regression (Ridge, scikit-learn) with auto range normalisation
- **Hand Gestures:** Palm-aware finger state detection (back-of-hand + palm-facing)
- **Mouse Control:** Multi-threaded smooth cursor via PyAutoGUI
- **Libraries:** OpenCV, NumPy, SciPy, scikit-learn, PyAutoGUI

## Features

### 👁️ Eye Tracking
- 3D head pose estimation using PCA on facial landmarks
- Iris tracking with EAR-weighted gaze direction fusion
- 16-point calibration with polynomial correction for edge accuracy
- Adaptive EMA gaze filter (speeds up during fast movement, smooths during fixation)
- Re-anchor drift correction (press R)

### 🤚 Hand Gestures
| Gesture | Action |
|---------|--------|
| ✌️ Index + Middle up | Left Click |
| ☝️ Index only | Right Click |
| 👌 Pinch (thumb + index) | Double Click |
| ✋ Open hand (back) | Scroll Up |
| ✊ Fist | Scroll Down |
| 🖐 Open palm (facing cam) | Scroll Up |

### 👁️ Eye Gestures
| Gesture | Action |
|---------|--------|
| Both eyes closed 0.6s | Left Click |
| Left wink | Right Click |
| Right wink | Double Click |

### Modes
- `1` — Eye only
- `2` — Hand only  
- `3` — Dual (default)

## Setup & Run

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirement.txt
python MonitorTracking.py
```

## Calibration (every session)
1. Press `S` — look at monitor centre (sets screen plane)
2. Press `C` — follow the 16-point dot
3. Press `M` — enable mouse control
4. Press `R` anytime to re-anchor drift

## Controls
| Key | Action |
|-----|--------|
| S | Set screen plane |
| C | Start 16-point calibration |
| R | Re-anchor gaze drift |
| M | Toggle mouse on/off |
| E / Shift+E | Nudge edge stretch up/down |
| T | Toggle palm/back-of-hand override |
| 1 / 2 / 3 | Switch mode (Eye/Hand/Dual) |
| F / G | Slower / faster filter |
| = / - | Increase / decrease sensitivity |
| Q | Quit |

## Architecture
```
Camera Frame
    ├── Face Mesh (MediaPipe)
    │     ├── Head Pose (PCA on nose landmarks)
    │     ├── Iris Centre 3D (weighted L+R)
    │     ├── EAR Blink Detection
    │     └── Gaze → Screen via Polynomial Calibration
    │
    └── Hand Tracking (MediaPipe)
          ├── Palm orientation detection (cross product)
          ├── Finger state classification
          ├── Gesture hold timer (0.4s debounce)
          └── PyAutoGUI action execution
```
