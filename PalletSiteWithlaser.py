"""
Industrial Pallet Inspection Workflow Dashboard
================================================
Simulates a real-time multi-stage pallet inspection system.
Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import random
import os
import glob
from datetime import datetime
import os, sys

print("WORKING DIR:", os.getcwd())
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IDEAL_IMAGE_PATH = os.path.join(BASE_DIR, "ideal.jpeg")
MODEL_PATH = os.path.join(BASE_DIR, "pallet_crack_model.keras")
SAMPLE_FOLDER    = os.path.join(BASE_DIR, "sample_images")

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="PalletVision QC",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

/* Global */
:root {
    --bg-primary:    #0b0d12;
    --bg-panel:      #10131a;
    --bg-card:       #161b26;
    --border-dim:    #1e2535;
    --border-glow:   #2a3a5c;
    --accent-cyan:   #00d4ff;
    --accent-green:  #00ff99;
    --accent-amber:  #ffb300;
    --accent-red:    #ff3b5c;
    --text-primary:  #e8edf5;
    --text-dim:      #6b7a99;
    --text-mono:     'Share Tech Mono', monospace;
    --text-ui:       'Rajdhani', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--text-ui) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-panel) !important;
    border-right: 1px solid var(--border-dim) !important;
}
[data-testid="stSidebar"] * { font-family: var(--text-ui) !important; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: var(--text-mono) !important;
    font-size: 13px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 8px 22px !important;
    border-radius: 2px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.08) !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.25) !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-family: var(--text-mono) !important;
    color: var(--accent-cyan) !important;
    font-size: 2rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--text-ui) !important;
    color: var(--text-dim) !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green)) !important;
    border-radius: 1px !important;
}
[data-testid="stProgress"] > div {
    background: var(--border-dim) !important;
    border-radius: 1px !important;
}

/* Expanders / text */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: var(--text-ui) !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* Code / monospaced blocks */
code, pre {
    font-family: var(--text-mono) !important;
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
}

/* Divider */
hr { border-color: var(--border-dim) !important; }

/* Selectbox / slider */
[data-baseweb="select"] { background: var(--bg-card) !important; }
.stSlider [data-testid="stSlider"] { color: var(--accent-cyan) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
STAGES = ["Camera Check", "Conductivity Check", "AI Check", "Load / Deflection Check", "Laser Check"]
STAGE_ICONS = ["📷", "⚡", "🧠", "⚖️", "🔴"]

COND_OK_RANGE    = (0.85, 1.20)   # siemens/m — acceptable
COND_WARN_RANGE  = (0.60, 0.85)
COND_FAIL_BELOW  = 0.60

DEFL_OK_MAX      = 8.0            # mm — acceptable deflection
DEFL_WARN_MAX    = 14.0
DEFL_FAIL_ABOVE  = 14.0

# ─── Session State Init ───────────────────────────────────────────────────────
def init_state():
    defaults = {
        "pallet_id":       None,
        "pallet_count":    0,
        "stage":           -1,          # -1 = idle
        "running":         False,
        "results":         {},
        "log":             [],
        "image_folder": SAMPLE_FOLDER,
        "cond_ok_hi":      COND_OK_RANGE[1],
        "cond_warn_lo":    COND_WARN_RANGE[0],
        "defl_ok_max":     DEFL_OK_MAX,
        "defl_warn_max":   DEFL_WARN_MAX,
        "auto_loop":       False,
        "cv_mode":         "Canny Edge",
        "border_margin":   30,
        "ai_conf_thresh":  0.08,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Utility ──────────────────────────────────────────────────────────────────
def new_pallet_id():
    ts = datetime.now().strftime("%H%M%S")
    rnd = random.randint(100, 999)
    return f"PLT-{ts}-{rnd}"

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state["log"].append(f"[{ts}]  {msg}")
    # Keep last 80 entries
    if len(st.session_state["log"]) > 80:
        st.session_state["log"] = st.session_state["log"][-80:]

def status_html(label: str, color: str) -> str:
    """Returns a compact colored badge."""
    colors = {
        "green":  ("#00ff99", "rgba(0,255,153,0.08)"),
        "yellow": ("#ffb300", "rgba(255,179,0,0.08)"),
        "red":    ("#ff3b5c", "rgba(255,59,92,0.08)"),
        "cyan":   ("#00d4ff", "rgba(0,212,255,0.08)"),
    }
    fg, bg = colors.get(color, ("#e8edf5", "rgba(255,255,255,0.05)"))
    return (
        f'<span style="display:inline-block;padding:3px 14px;border-radius:2px;'
        f'background:{bg};border:1px solid {fg};color:{fg};'
        f'font-family:\'Share Tech Mono\',monospace;font-size:13px;letter-spacing:1px;">'
        f'{label}</span>'
    )

# ─── Image Processing ─────────────────────────────────────────────────────────
def load_random_image(folder: str):
    """Load a random image from the specified folder; generate fallback if empty."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    if not files:
        # Generate a synthetic pallet image as fallback
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (18, 22, 30)
        for y in range(80, 400, 55):
            c = tuple(np.random.randint(60, 120, 3).tolist())
            cv2.rectangle(img, (50, y), (590, y+38), c, -1)
            cv2.rectangle(img, (50, y), (590, y+38), (35, 35, 35), 2)
        for x in [50, 210, 375, 560]:
            cv2.rectangle(img, (x, 80), (x+28, 400), (45,38,28), -1)
        noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        return img, "synthetic"
    path = random.choice(files)
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
    return img, os.path.basename(path)

def process_image_canny(img: np.ndarray) -> np.ndarray:
    """Apply Canny edge detection with a colorized overlay."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Colorize edges: cyan on dark background
    out = img.copy()
    out[edges > 0] = [255, 210, 0]          # BGR → amber edges
    # Dim the non-edge pixels slightly for contrast
    mask = edges == 0
    out[mask] = (out[mask] * 0.45).astype(np.uint8)
    # Overlay scanlines for industrial feel
    for y in range(0, out.shape[0], 4):
        out[y] = (out[y] * 0.7).astype(np.uint8)
    return out

def process_image_contours(img: np.ndarray) -> np.ndarray:
    """Find and draw contours with color-coded overlays."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    out = (out * 0.35).astype(np.uint8)     # darken base
    # Draw contours in cyan gradient by area
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    for i, cnt in enumerate(contours_sorted[:20]):
        ratio = 1.0 - (i / max(len(contours_sorted), 1))
        color = (int(255 * ratio), int(200 * ratio), 0)   # BGR warm to cool
        cv2.drawContours(out, [cnt], -1, color, 1)
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 220, 255), 1)
    return out

def process_image_heatmap(img: np.ndarray) -> np.ndarray:
    """Apply a false-color heatmap using gradient magnitude."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.35, heatmap, 0.65, 0)
    return out

def apply_cv_overlay(img: np.ndarray, mode: str) -> np.ndarray:
    """Dispatcher for CV processing modes."""
    if mode == "Canny Edge":
        return process_image_canny(img)
    elif mode == "Contour Map":
        return process_image_contours(img)
    elif mode == "Gradient Heatmap":
        return process_image_heatmap(img)
    return img

def add_hud(img: np.ndarray, pallet_id: str) -> np.ndarray:
    """Add HUD-style overlay: grid, corner brackets, ID stamp."""
    h, w = img.shape[:2]
    out = img.copy()
    # Corner brackets
    blen, bthk = 24, 2
    bc = (0, 220, 255)
    for (x, y, dx, dy) in [(5,5,1,1),(w-5,5,-1,1),(5,h-5,1,-1),(w-5,h-5,-1,-1)]:
        cv2.line(out, (x, y), (x+dx*blen, y), bc, bthk)
        cv2.line(out, (x, y), (x, y+dy*blen), bc, bthk)
    # Pallet ID stamp
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"ID: {pallet_id}"
    cv2.putText(out, label, (12, h-12), font, 0.42, (0, 200, 220), 1, cv2.LINE_AA)
    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(out, ts, (w-200, h-12), font, 0.38, (80, 100, 140), 1, cv2.LINE_AA)
    # Thin center crosshair
    mid_x, mid_y = w//2, h//2
    cv2.line(out, (mid_x-20, mid_y), (mid_x+20, mid_y), (0,180,200), 1)
    cv2.line(out, (mid_x, mid_y-20), (mid_x, mid_y+20), (0,180,200), 1)
    return out

# ─── AI Heatmap (placeholder) ────────────────────────────────────────────────
def generate_ai_heatmap(img: np.ndarray) -> np.ndarray:
    """
    Placeholder AI visualization — generates a fake attention heatmap.
    Replace this function body with real model inference later.
    """
    h, w = img.shape[:2]
    # Create fake gaussian attention blobs
    heatmap = np.zeros((h, w), dtype=np.float32)
    for _ in range(random.randint(2, 5)):
        cx = random.randint(w//6, 5*w//6)
        cy = random.randint(h//6, 5*h//6)
        sigma_x = random.randint(40, 100)
        sigma_y = random.randint(30, 80)
        for y in range(h):
            for x in range(w):
                heatmap[y, x] += np.exp(
                    -((x-cx)**2/(2*sigma_x**2) + (y-cy)**2/(2*sigma_y**2))
                )
    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    blended = cv2.addWeighted(img, 0.50, colored, 0.50, 0)
    # Add bounding boxes for detected "regions"
    h_norm = heatmap > 160
    h_u8 = h_norm.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(h_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 800:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(blended, (x, y), (x+bw, y+bh), (0, 255, 120), 2)
            cv2.putText(blended, "ROI", (x+4, y+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)
    return blended

# ─── Inspection Stage Functions ───────────────────────────────────────────────

def run_camera_check(image_folder: str, cv_mode: str, pallet_id: str) -> dict:
    """
    Stage 1 — Camera Check (multi-contour presence check, robust matching)
    """


    log("▶ Camera check: loading image…")
    raw_bgr, fname = load_random_image(image_folder)
    log(f"  Image loaded: {fname}  ({raw_bgr.shape[1]}×{raw_bgr.shape[0]})")

    processed_bgr = apply_cv_overlay(raw_bgr, cv_mode)

    # --- Helper: extract contours ---
    def get_contours(img, max_cnt):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 60)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        min_length = 500
        contours = [c for c in contours if cv2.arcLength(c, True) > min_length]

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours[:max_cnt]

    # --- Load ideal image safely ---
    ideal_bgr = cv2.imread(IDEAL_IMAGE_PATH)
    if ideal_bgr is None:
        raise ValueError(f"Failed to load ideal image: {IDEAL_IMAGE_PATH}")

    # Resize ideal to match test image
    h, w = raw_bgr.shape[:2]
    ideal_resized = cv2.resize(ideal_bgr, (w, h))

    # Extract contours
    ideal_contours = get_contours(ideal_resized, 25)
    test_contours  = get_contours(raw_bgr, 75)

    matched = 0
    matches = []
    used_indices = set()  # <-- prevents reusing same contour

    threshold = 0.1

    # --- Robust matching ---
    for ideal_cnt in ideal_contours:
        best_score = float("inf")
        best_idx = -1

        for i, test_cnt in enumerate(test_contours):
            if i in used_indices:
                continue

            score = cv2.matchShapes(ideal_cnt, test_cnt, 1, 0.0)

            if score < best_score:
                best_score = score
                best_idx = i

        if best_score < threshold and best_idx != -1:
            matched += 1
            matches.append(True)
            used_indices.add(best_idx)
        else:
            matches.append(False)

    # --- Compute result ---
    total = len(ideal_contours)
    ratio = matched / total if total > 0 else 0

    if ratio > 0.8:
        status = "OK"
    elif ratio > 0.5:
        status = "WARN"
    else:
        status = "FAIL"

    log(f"  Contours matched: {matched}/{total} ({ratio:.2f}) | Status: {status}")

    # --- Visualization ---
    for i, ideal_cnt in enumerate(ideal_contours):
        color = (0, 255, 0) if matches[i] else (0, 0, 255)
        cv2.drawContours(processed_bgr, [ideal_cnt], -1, color, 2)

    if status == "FAIL":
        cv2.putText(
            processed_bgr,
            "MISSING STRUCTURE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    # --- HUD ---
    raw_hud = add_hud(raw_bgr.copy(), pallet_id)
    proc_hud = add_hud(processed_bgr, pallet_id)

    # --- Convert for Streamlit ---
    raw_rgb  = cv2.cvtColor(raw_hud,  cv2.COLOR_BGR2RGB)
    proc_rgb = cv2.cvtColor(proc_hud, cv2.COLOR_BGR2RGB)

    return {
        "raw_rgb":    raw_rgb,
        "proc_rgb":   proc_rgb,
        "filename":   fname,
        "brightness": ratio,   # used as match ratio
        "status":     status,
        "cv_mode":    cv_mode,
    }


def run_conductivity_check(warn_lo: float, ok_hi: float) -> dict:
    """
    Stage 2 — Conductivity Check
    Simulates a resistivity sensor reading.
    Replace with real hardware SDK call later.
    """
    log("▶ Conductivity check: sampling sensor…")
    # Simulate with slight clustering around 'good' values most of the time
    base = random.gauss(0.95, 0.25)
    value = round(max(0.1, min(2.0, base)), 3)

    if value >= warn_lo and value <= ok_hi:
        status, color = "OK", "green"
    elif value >= COND_FAIL_BELOW and value < warn_lo:
        status, color = "WARNING", "yellow"
    else:
        status, color = "FAIL", "red"

    log(f"  Conductivity: {value:.3f} S/m  |  Status: {status}")
    return {"value": value, "status": status, "color": color, "unit": "S/m"}



def _build_defect_heatmap(raw_bgr: np.ndarray, mask: np.ndarray, border_margin: int = 25) -> np.ndarray:
    """
    Gaussian defect heatmap with border exclusion law applied.
    """

    import cv2
    import numpy as np

    h, w = raw_bgr.shape[:2]

    # ─────────────────────────────────────────────
    # BORDER LAW (zero out unreliable edge zone)
    # ─────────────────────────────────────────────
    if border_margin > 0:
        mask = mask.copy()
        mask[:border_margin, :] = 0
        mask[-border_margin:, :] = 0
        mask[:, :border_margin] = 0
        mask[:, -border_margin:] = 0

    # ─────────────────────────────────────────────
    # COMPONENT ANALYSIS
    # ─────────────────────────────────────────────
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(mask)

    heat = np.zeros((h, w), dtype=np.float32)
    ys, xs = np.ogrid[:h, :w]

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 20:
            continue

        comp = (label_map == lbl).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        arc = cv2.arcLength(cnts[0], False) if cnts else 1.0

        cx, cy = centroids[lbl]
        sigma = np.clip(arc * 0.25, 10, 120)

        blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        heat += blob

    if heat.max() > 0:
        heat /= heat.max()

    heat_u8 = (heat * 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)

    alpha = heat[:, :, None]
    base = (raw_bgr * 0.5).astype(np.float32)

    return (base * (1 - alpha) + color * alpha).astype(np.uint8)


def run_ai_check(raw_bgr: np.ndarray, camera_status: str = "OK") -> dict:

        import cv2
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        global keras_model

        BORDER_MARGIN = 25

        camera_status = camera_status.upper()

        # ─────────────────────────────────────────────
        # 0. CAMERA GATE
        # ─────────────────────────────────────────────
        if camera_status == "OK":
            log("▶ AI skipped (camera OK)")

            return {
                "label": "PASS",
                "status": "OK",
                "confidence": 99.0,
                "heatmap": cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB),
                "ai_skipped": True
            }

        # ─────────────────────────────────────────────
        # 1. LOAD MODEL
        # ─────────────────────────────────────────────
        import streamlit as st

        @st.cache_resource
        def load_model_cached():
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False
            )
            log("▶ AI model loaded")
            return model

        keras_model = load_model_cached()

        # ─────────────────────────────────────────────
        # 2. ADAPTIVE THRESHOLD
        # ─────────────────────────────────────────────
        conf_thresh = 0.06 if camera_status == "FAIL" else 0.10

        # ─────────────────────────────────────────────
        # 3. PREPROCESS
        # ─────────────────────────────────────────────
        raw_bgr = cv2.resize(raw_bgr, (640, 480))

        gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        inp = gray.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, (0, -1))

        pred = keras_model.predict(inp, verbose=0)[0]

        # ─────────────────────────────────────────────
        # 4. SEGMENTATION
        # ─────────────────────────────────────────────
        if len(pred.shape) >= 2:

            pred_2d = pred[:, :, 0] if len(pred.shape) == 3 else pred
            pred_2d = cv2.resize(pred_2d, (640, 480))

            # ── BORDER LAW (critical fix) ──
            pred_2d[:BORDER_MARGIN, :] = 0
            pred_2d[-BORDER_MARGIN:, :] = 0
            pred_2d[:, :BORDER_MARGIN] = 0
            pred_2d[:, -BORDER_MARGIN:] = 0

            mask = (pred_2d > conf_thresh).astype(np.uint8)

            num_labels, labels = cv2.connectedComponents(mask)

            regions = 0
            ratio = float(np.sum(mask)) / (640 * 480)

            for lbl in range(1, num_labels):
                comp = labels == lbl
                if np.sum(comp) < 20:
                    continue

                score = float(np.mean(pred_2d[comp]))

                if score < conf_thresh:
                    continue

                regions += 1

            overlay = _build_defect_heatmap(raw_bgr, mask, BORDER_MARGIN)

        # ─────────────────────────────────────────────
        # 5. CLASSIFICATION FALLBACK
        # ─────────────────────────────────────────────
        else:
            score = float(pred.flatten()[0])
            regions = int(score > conf_thresh)
            ratio = score
            overlay = raw_bgr.copy()

        # ─────────────────────────────────────────────
        # 6. DECISION LOGIC
        # ─────────────────────────────────────────────
        if camera_status == "FAIL":
            status, label = ("FAIL", "CRACK DETECTED") if regions else ("WARN", "SUSPICIOUS")
        else:
            is_defect = (regions >= 1)

            if is_defect:
                status = "FAIL"
                label = "CRACK DETECTED"
            else:
                status = "OK"
                label = "PASS"

        confidence = round((1.0 - ratio) * 100, 1)

        log(f"▶ FINAL → {label} | conf={confidence}% | regions={regions}")

        return {
            "label": label,
            "status": status,
            "confidence": confidence,
            "heatmap": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            "camera_status": camera_status,
            "regions": regions,
            "ai_ratio": ratio,
            "ai_skipped": False
        }

def run_load_check(ok_max: float, warn_max: float) -> dict:
    """
    Stage 4 — Load / Deflection Check
    Simulates strain-gauge deflection reading.
    """
    log("▶ Load check: measuring deflection…")
    value = round(abs(random.gauss(5.0, 4.5)), 2)

    if value <= ok_max:
        status, color = "OK", "green"
    elif value <= warn_max:
        status, color = "WARNING", "yellow"
    else:
        status, color = "FAIL", "red"

    log(f"  Deflection: {value:.2f} mm  |  Status: {status}")
    return {"value": value, "status": status, "color": color,
            "ok_max": ok_max, "warn_max": warn_max}


def run_laser_check() -> dict:
    """
    Stage 5 — Laser Corner Block Check
    Simulates 4 laser distance sensors positioned at pallet corners.
    Each sensor checks whether a structural corner block is present
    and within tolerance. Returns per-corner results + overall status.
    """
    log("▶ Laser check: scanning 4 corner blocks…")

    NOMINAL_MM  = 145.0    # expected block height
    TOL_OK_MM   = 4.0      # ±4 mm  → OK
    TOL_WARN_MM = 10.0     # ±10 mm → WARNING, beyond → FAIL

    corners = ["TL", "TR", "BL", "BR"]   # Top-Left, Top-Right, Bottom-Left, Bottom-Right
    corner_labels = {
        "TL": "Top-Left",
        "TR": "Top-Right",
        "BL": "Bottom-Left",
        "BR": "Bottom-Right",
    }
    results_per_corner = {}

    for corner in corners:
        # Simulate measurement: normally distributed around nominal, occasional outlier
        if random.random() < 0.08:    # ~8% chance of a bad block
            measured = round(NOMINAL_MM + random.uniform(12, 35) * random.choice([-1, 1]), 2)
        else:
            measured = round(NOMINAL_MM + random.gauss(0, 2.5), 2)

        deviation = abs(measured - NOMINAL_MM)

        if deviation <= TOL_OK_MM:
            cstatus, ccolor = "OK", "green"
        elif deviation <= TOL_WARN_MM:
            cstatus, ccolor = "WARN", "yellow"
        else:
            cstatus, ccolor = "FAIL", "red"

        results_per_corner[corner] = {
            "label":     corner_labels[corner],
            "measured":  measured,
            "deviation": round(deviation, 2),
            "status":    cstatus,
            "color":     ccolor,
        }
        log(f"  {corner}: {measured:.1f} mm (Δ{deviation:.1f}) → {cstatus}")

    # Overall: worst corner wins
    all_statuses = [v["status"] for v in results_per_corner.values()]
    if "FAIL" in all_statuses:
        overall_status = "FAIL"
    elif "WARN" in all_statuses:
        overall_status = "WARNING"
    else:
        overall_status = "OK"

    log(f"  Laser overall: {overall_status}")
    return {
        "corners":  results_per_corner,
        "status":   overall_status,
        "nominal":  NOMINAL_MM,
        "tol_ok":   TOL_OK_MM,
        "tol_warn": TOL_WARN_MM,
    }

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙ Configuration")
    st.divider()

    st.markdown("**Image Source**")
    folder = st.text_input("Image folder", value=st.session_state["image_folder"], key="folder_input")
    st.session_state["image_folder"] = folder

    st.markdown("**CV Processing Mode**")
    cv_mode = st.selectbox(
        "Overlay type",
        ["Canny Edge", "Contour Map", "Gradient Heatmap"],
        index=["Canny Edge", "Contour Map", "Gradient Heatmap"].index(st.session_state["cv_mode"]),
    )
    st.session_state["cv_mode"] = cv_mode

    st.divider()
    st.markdown("**Conductivity Thresholds (S/m)**")
    cond_warn_lo = st.slider("Warn below", 0.1, 1.5, st.session_state["cond_warn_lo"], 0.05)
    cond_ok_hi   = st.slider("Fail above", 0.5, 2.0, st.session_state["cond_ok_hi"],   0.05)
    st.session_state["cond_warn_lo"] = cond_warn_lo
    st.session_state["cond_ok_hi"]   = cond_ok_hi

    st.divider()
    st.markdown("**Deflection Thresholds (mm)**")
    defl_ok   = st.slider("OK max",   1.0, 20.0, st.session_state["defl_ok_max"],   0.5)
    defl_warn = st.slider("Warn max", 5.0, 30.0, st.session_state["defl_warn_max"], 0.5)
    st.session_state["defl_ok_max"]   = defl_ok
    st.session_state["defl_warn_max"] = defl_warn

    st.divider()
    st.markdown("**AI Check Settings**")
    ai_conf = st.slider(
        "Detection threshold", 0.01, 0.50,
        float(st.session_state["ai_conf_thresh"]), 0.01,
        help="Model confidence cutoff. Raise to reduce false positives."
    )
    st.session_state["ai_conf_thresh"] = ai_conf

    border_m = st.slider(
        "Border ignore margin (px)", 0, 80,
        int(st.session_state["border_margin"]), 5,
        help="Pixels stripped from all 4 edges before AI analysis."
    )
    st.session_state["border_margin"] = border_m

    st.divider()
    auto_loop = st.toggle("Auto-loop pallets", value=st.session_state["auto_loop"])
    st.session_state["auto_loop"] = auto_loop

    st.divider()
    st.markdown(
        '<p style="font-family:\'Share Tech Mono\',monospace;font-size:11px;'
        'color:#3a4560;letter-spacing:1px;">PalletVision QC v1.0.0<br>'
        'Nexus United · 2026</p>',
        unsafe_allow_html=True,
    )

# ─── Header ───────────────────────────────────────────────────────────────────
col_logo, col_title, col_kpi = st.columns([1, 4, 2])
with col_logo:
    st.markdown(
        '<div style="font-size:48px;margin-top:8px;">🔍</div>',
        unsafe_allow_html=True,
    )
with col_title:
    st.markdown(
        '<h1 style="margin:0;font-family:Rajdhani,sans-serif;font-weight:700;'
        'font-size:2.2rem;letter-spacing:4px;color:#e8edf5;text-transform:uppercase;">'
        'PalletVision QC</h1>'
        '<p style="margin:0;font-family:\'Share Tech Mono\',monospace;font-size:12px;'
        'color:#3a4560;letter-spacing:2px;">INDUSTRIAL INSPECTION WORKFLOW SYSTEM</p>',
        unsafe_allow_html=True,
    )
with col_kpi:
    st.metric("Pallets Inspected", st.session_state["pallet_count"])

st.divider()

# ─── Stage Progress Bar ────────────────────────────────────────────────────────
def render_stage_bar(current_stage: int):
    cols = st.columns(len(STAGES))
    for i, (icon, name) in enumerate(zip(STAGE_ICONS, STAGES)):
        with cols[i]:
            is_done    = current_stage > i
            is_active  = current_stage == i
            is_pending = current_stage < i

            if is_active:
                bg   = "rgba(0,212,255,0.10)"
                bord = "#00d4ff"
                fg   = "#00d4ff"
                tag  = "● ACTIVE"
            elif is_done:
                bg   = "rgba(0,255,153,0.06)"
                bord = "#00ff99"
                fg   = "#00ff99"
                tag  = "✓ DONE"
            else:
                bg   = "rgba(255,255,255,0.02)"
                bord = "#1e2535"
                fg   = "#3a4560"
                tag  = "○ QUEUE"

            st.markdown(
                f'<div style="background:{bg};border:1px solid {bord};border-radius:3px;'
                f'padding:12px 10px;text-align:center;">'
                f'<div style="font-size:22px;margin-bottom:4px;">{icon}</div>'
                f'<div style="font-family:Rajdhani,sans-serif;font-weight:700;'
                f'font-size:13px;color:{fg};letter-spacing:1px;text-transform:uppercase;">'
                f'{name}</div>'
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
                f'color:{fg};opacity:0.7;margin-top:4px;">{tag}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

render_stage_bar(st.session_state["stage"])

st.markdown("<br>", unsafe_allow_html=True)

# ─── Control Buttons ──────────────────────────────────────────────────────────
btn_col1, btn_col2, btn_col3, spacer = st.columns([1.5, 1.5, 1.5, 4])

with btn_col1:
    start_clicked = st.button("▶  Start Inspection", key="btn_start",
                               disabled=st.session_state["running"])
with btn_col2:
    next_clicked  = st.button("⏭  Next Pallet", key="btn_next",
                               disabled=st.session_state["running"] or
                               st.session_state["stage"] not in (-1, len(STAGES)))
with btn_col3:
    reset_clicked = st.button("↺  Reset", key="btn_reset")

if reset_clicked:
    st.session_state["stage"]   = -1
    st.session_state["running"] = False
    st.session_state["results"] = {}
    st.session_state["pallet_id"] = None
    log("── System reset ──")
    st.rerun()

# Trigger new inspection
trigger = start_clicked or next_clicked
if trigger and not st.session_state["running"]:
    st.session_state["pallet_id"]  = new_pallet_id()
    st.session_state["stage"]      = 0
    st.session_state["running"]    = True
    st.session_state["results"]    = {}
    log(f"═══ New pallet: {st.session_state['pallet_id']} ═══")
    st.rerun()

st.divider()

# ─── Active Pallet Banner ─────────────────────────────────────────────────────
if st.session_state["pallet_id"]:
    pid = st.session_state["pallet_id"]
    st.markdown(
        f'<div style="background:rgba(0,212,255,0.04);border-left:3px solid #00d4ff;'
        f'padding:8px 16px;font-family:\'Share Tech Mono\',monospace;font-size:13px;'
        f'letter-spacing:1px;color:#00d4ff;">PALLET  ›  {pid}</div>',
        unsafe_allow_html=True,
    )

# ─── Main Inspection Loop ─────────────────────────────────────────────────────
if st.session_state["running"] and st.session_state["stage"] < len(STAGES):
    stage = st.session_state["stage"]
    stage_name = STAGES[stage]

    with st.spinner(f"Running {stage_name}…"):

        # ── Stage 0: Camera Check ────────────────────────────────────────────
        if stage == 0:
            res = run_camera_check(
                st.session_state["image_folder"],
                st.session_state["cv_mode"],
                st.session_state["pallet_id"],
            )
            st.session_state["results"]["camera"] = res
            time.sleep(0.8)

        # ── Stage 1: Conductivity Check ───────────────────────────────────────
        elif stage == 1:
            res = run_conductivity_check(
                st.session_state["cond_warn_lo"],
                st.session_state["cond_ok_hi"],
            )
            st.session_state["results"]["conductivity"] = res
            time.sleep(0.6)

        # ── Stage 2: AI Check ─────────────────────────────────────────────────
        elif stage == 2:
            cam_res = st.session_state["results"].get("camera", {})
            cam_status = cam_res.get("status", "OK")

            # -----------------------------------------
            # SKIP AI if camera says pallet is OK
            # -----------------------------------------
            if cam_status == "OK":
                log("▶ AI skipped (camera OK)")

                res = {
                    "label": "SKIPPED",
                    "status": "OK",
                    "confidence": 100,
                    "ai_ratio": 0.0,
                    "regions": 0,
                    "note": "Skipped due to clean camera result"
                }

                st.session_state["results"]["ai"] = res
                time.sleep(0.3)

            else:
                # -----------------------------------------
                # RUN AI ONLY WHEN NEEDED
                # -----------------------------------------
                raw_rgb = cam_res.get("raw_rgb")

                if raw_rgb is not None:
                    raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
                else:
                    raw_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

                res = run_ai_check(
                    raw_bgr,
                    camera_status=cam_status
                )

                st.session_state["results"]["ai"] = res
                time.sleep(0.5)

        # ── Stage 3: Load / Deflection Check ─────────────────────────────────
        elif stage == 3:
            res = run_load_check(
                st.session_state["defl_ok_max"],
                st.session_state["defl_warn_max"],
            )
            st.session_state["results"]["load"] = res
            time.sleep(0.6)

        # ── Stage 4: Laser Corner Check ───────────────────────────────────────
        elif stage == 4:
            res = run_laser_check()
            st.session_state["results"]["laser"] = res
            time.sleep(0.7)

    # Advance stage
    st.session_state["stage"] += 1

    # If all stages complete
    if st.session_state["stage"] >= len(STAGES):
        st.session_state["pallet_count"] += 1
        st.session_state["running"] = False
        log(f"═══ Inspection complete: {st.session_state['pallet_id']} ═══")
        if st.session_state["auto_loop"]:
            time.sleep(1.5)
            st.session_state["pallet_id"]  = new_pallet_id()
            st.session_state["stage"]      = 0
            st.session_state["running"]    = True
            st.session_state["results"]    = {}
            log(f"═══ Auto-loop → new pallet: {st.session_state['pallet_id']} ═══")

    st.rerun()

def normalize_status(s):
    s = (s or "").upper()

    if s in ("OK", "PASS", "GOOD"):
        return "OK"
    if s in ("WARN", "WARNING", "SUSPICIOUS"):
        return "WARN"
    if s in ("FAIL", "ERROR", "BAD"):
        return "FAIL"

    return "WARN"

# ─── Results Display ──────────────────────────────────────────────────────────
results = st.session_state["results"]

if "camera" in results:
    cam = results["camera"]
    st.markdown("### 📷 Camera Check")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p style="font-family:\'Share Tech Mono\',monospace;font-size:11px;'
                    'color:#3a4560;letter-spacing:1px;">ORIGINAL</p>', unsafe_allow_html=True)
        st.image(cam["raw_rgb"], use_container_width=True)
    with c2:
        st.markdown(f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:11px;'
                    f'color:#3a4560;letter-spacing:1px;">{cam["cv_mode"].upper()}</p>',
                    unsafe_allow_html=True)
        st.image(cam["proc_rgb"], use_container_width=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Brightness", cam["brightness"])
    with m2:
        st.metric("Resolution", f'{cam["raw_rgb"].shape[1]}×{cam["raw_rgb"].shape[0]}')
    with m3:
        color_map = {"OK": "green", "WARN": "yellow", "FAIL": "red"}
        st.markdown(
            f'**Status:** {status_html(cam["status"], color_map.get(cam["status"],"cyan"))}',
            unsafe_allow_html=True,
        )
    st.divider()

if "conductivity" in results:
    cnd = results["conductivity"]
    st.markdown("### ⚡ Conductivity Check")
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.metric("Conductivity", f'{cnd["value"]:.3f} S/m')
    with c2:
        color_map = {"OK": "green", "WARNING": "yellow", "FAIL": "red"}
        st.markdown(
            f'<br>**Status:** {status_html(cnd["status"], color_map[cnd["status"]])}',
            unsafe_allow_html=True,
        )
    with c3:
        # Visual gauge
        norm = min(cnd["value"] / 2.0, 1.0)
        bar_color = {"OK": "#00ff99", "WARNING": "#ffb300", "FAIL": "#ff3b5c"}[cnd["status"]]
        st.markdown(
            f'<br><div style="background:#1e2535;border-radius:2px;height:14px;width:100%;">'
            f'<div style="background:{bar_color};border-radius:2px;height:14px;'
            f'width:{norm*100:.1f}%;transition:width 0.5s;"></div></div>'
            f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
            f'color:#3a4560;margin-top:4px;">'
            f'0.0 S/m ─────────────────────── 2.0 S/m</p>',
            unsafe_allow_html=True,
        )
    st.divider()

if "ai" in results:
    ai = results["ai"]
    st.markdown("### 🧠 AI Check")
    c1, c2 = st.columns(2)
    with c1:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("AI Status", ai["label"])
        with m2:
            st.metric("Confidence", f'{ai["confidence"]:.1f}%')
        st.progress(ai["confidence"] / 100)
        # Status badge reflects the ACTUAL model output, not a hardcoded "PASS"
        ai_color_map = {"OK": "green", "WARNING": "yellow", "FAIL": "red"}
        ai_badge_color = ai_color_map.get(ai["status"], "cyan")
        result_text = f'{ai["label"]}' + (
            f'  —  {ai.get("defect_pixels", 0)} px / {ai.get("defect_regions", 0)} regions'
            if ai["status"] != "OK" else "  —  No defects detected"
        )
        #st.markdown(
         #   f'**Result:** {status_html(result_text, ai_badge_color)}',
          #  unsafe_allow_html=True,
        #)
        details = []
        if ai.get("defect_ratio") is not None:
            details.append(f'Coverage: {ai["defect_ratio"]:.3f}%')
        if ai.get("conf_thresh") is not None:
            details.append(f'Threshold: {ai["conf_thresh"]:.3f}')
        if ai.get("border_margin") is not None:
            details.append(f'Border: {ai["border_margin"]}px ignored')
        if details:
            st.markdown(
                f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:11px;'
                f'color:#6b7a99;margin-top:6px;">'
                + "  ·  ".join(details) + '</p>',
                unsafe_allow_html=True,
            )
    with c2:
        if "heatmap" in ai:
            heatmap_label = (
                "DEFECT HEATMAP — cracks highlighted, blob size ∝ crack length"
                if ai["status"] != "OK"
                else "ATTENTION MAP — no significant defects"
            )
            st.markdown(
                f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:11px;'
                f'color:#3a4560;letter-spacing:1px;">{heatmap_label}</p>',
                unsafe_allow_html=True,
            )
            st.image(ai["heatmap"], use_container_width=True)
    st.divider()

if "load" in results:
    ld = results["load"]
    st.markdown("### ⚖️ Load / Deflection Check")
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.metric("Deflection", f'{ld["value"]:.2f} mm')
    with c2:
        color_map = {"OK": "green", "WARNING": "yellow", "FAIL": "red"}
        st.markdown(
            f'<br>**Status:** {status_html(ld["status"], color_map[ld["status"]])}',
            unsafe_allow_html=True,
        )
    with c3:
        norm = min(ld["value"] / (ld["warn_max"] * 1.4), 1.0)
        bar_color = {"OK": "#00ff99", "WARNING": "#ffb300", "FAIL": "#ff3b5c"}[ld["status"]]
        max_label = ld["warn_max"] * 1.4
        st.markdown(
            f'<br><div style="background:#1e2535;border-radius:2px;height:14px;width:100%;">'
            f'<div style="background:{bar_color};border-radius:2px;height:14px;'
            f'width:{norm*100:.1f}%;transition:width 0.5s;"></div></div>'
            f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
            f'color:#3a4560;margin-top:4px;">'
            f'0 mm ──────────────────────── {max_label:.0f} mm</p>',
            unsafe_allow_html=True,
        )
    st.divider()

if "laser" in results:
    lr = results["laser"]
    st.markdown("### 🔴 Laser Check — Corner Block Scan")

    # ─────────────────────────────────────────────
    CORNER_POSITIONS = {
        "TL": ("Top-Left",     "╔"),
        "TR": ("Top-Right",    "╗"),
        "BL": ("Bottom-Left",  "╚"),
        "BR": ("Bottom-Right", "╝"),
    }

    STATUS_COLORS = {
        "OK":   ("#00ff99", "rgba(0,255,153,0.06)", "rgba(0,255,153,0.15)"),
        "WARN": ("#ffb300", "rgba(255,179,0,0.06)", "rgba(255,179,0,0.15)"),
        "FAIL": ("#ff3b5c", "rgba(255,59,92,0.06)",  "rgba(255,59,92,0.18)"),
    }

    STATUS_ICON = {
        "OK": "▲",
        "WARN": "◆",
        "FAIL": "✕"
    }

    
    def norm_status(s):
        s = (s or "").upper()
        if s in ("OK", "PASS", "GOOD"):
            return "OK"
        if s in ("WARN", "WARNING", "SUSPICIOUS"):
            return "WARN"
        if s in ("FAIL", "ERROR", "BAD"):
            return "FAIL"
        return "WARN"

    # ─────────────────────────────────────────────
    # SAFE DATA ACCESS
    # ─────────────────────────────────────────────
    corners = lr.get("corners", {})
    nom     = lr.get("nominal", 0)
    tok     = lr.get("tol_ok", 0)
    twarn   = lr.get("tol_warn", 0)

    row1_cols = st.columns([1, 3, 1])
    row2_cols = st.columns([1, 3, 1])

    # ─────────────────────────────────────────────
    # CORNER CARD
    # ─────────────────────────────────────────────
    def corner_card(key, col):
        c = corners.get(key, {})

        st_key = norm_status(c.get("status"))

        fg, bg_dim, bg_hi = STATUS_COLORS[st_key]
        icon = STATUS_ICON[st_key]
        sym  = CORNER_POSITIONS[key][1]

        col.markdown(
            f'<div style="background:{bg_hi};border:1.5px solid {fg};border-radius:4px;'
            f'padding:14px 10px;text-align:center;">'

            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:22px;'
            f'color:{fg};line-height:1;">{sym}</div>'

            f'<div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:12px;'
            f'color:{fg};letter-spacing:1.5px;margin-top:4px;text-transform:uppercase;">'
            f'{c.get("label", key)}</div>'

            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:20px;'
            f'color:{fg};margin:6px 0;">{c.get("measured", 0):.1f} mm</div>'

            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:11px;'
            f'color:{fg};opacity:0.75;">Δ {c.get("deviation", 0):.1f} mm</div>'

            f'<div style="margin-top:8px;font-family:Rajdhani,sans-serif;font-weight:700;'
            f'font-size:13px;letter-spacing:2px;color:{fg};">{icon} {st_key}</div>'

            f'</div>',
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────
    # ROW 1
    # ─────────────────────────────────────────────
    corner_card("TL", row1_cols[0])

    with row1_cols[1]:

        tl_s = norm_status(corners.get("TL", {}).get("status"))
        tr_s = norm_status(corners.get("TR", {}).get("status"))
        bl_s = norm_status(corners.get("BL", {}).get("status"))
        br_s = norm_status(corners.get("BR", {}).get("status"))

        tl_fg = STATUS_COLORS[tl_s][0]
        tr_fg = STATUS_COLORS[tr_s][0]
        bl_fg = STATUS_COLORS[bl_s][0]
        br_fg = STATUS_COLORS[br_s][0]

        overall = norm_status(lr.get("status"))
        overall_color = STATUS_COLORS[overall][0]
        overall_icon = STATUS_ICON[overall]

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.02);border:1px solid #1e2535;'
            f'border-radius:4px;padding:18px 20px;text-align:center;min-height:140px;">'

            f'<div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:11px;'
            f'letter-spacing:2px;color:#3a4560;text-transform:uppercase;margin-bottom:8px;">'
            f'PALLET SCHEMATIC</div>'

            f'<div style="display:flex;justify-content:space-between;font-family:\'Share Tech Mono\',monospace;'
            f'font-size:10px;color:#3a4560;">'
            f'<span style="color:{tl_fg};">● TL</span><span style="color:{tr_fg};">TR ●</span></div>'

            f'<div style="border:1px solid #2a3a5c;border-radius:2px;margin:6px 0;'
            f'height:48px;background:rgba(42,58,92,0.15);display:flex;'
            f'align-items:center;justify-content:center;">'
            f'<span style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:13px;'
            f'letter-spacing:3px;color:#2a3a5c;">PALLET SURFACE</span></div>'

            f'<div style="display:flex;justify-content:space-between;font-family:\'Share Tech Mono\',monospace;'
            f'font-size:10px;color:#3a4560;">'
            f'<span style="color:{bl_fg};">● BL</span><span style="color:{br_fg};">BR ●</span></div>'

            f'<div style="margin-top:12px;font-family:\'Share Tech Mono\',monospace;font-size:11px;'
            f'color:#3a4560;">Nominal: {nom:.0f} mm  ·  OK: ±{tok:.0f}  ·  Warn: ±{twarn:.0f}</div>'

            f'<div style="margin-top:6px;font-family:Rajdhani,sans-serif;font-weight:700;font-size:1.1rem;'
            f'letter-spacing:3px;color:{overall_color};">{overall_icon} {overall}</div>'

            f'</div>',
            unsafe_allow_html=True,
        )

    corner_card("TR", row1_cols[2])

    # ─────────────────────────────────────────────
    # ROW 2
    # ─────────────────────────────────────────────
    corner_card("BL", row2_cols[0])
    row2_cols[1].markdown("")
    corner_card("BR", row2_cols[2])

    # ─────────────────────────────────────────────
    # GAUGES
    # ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    gauge_cols = st.columns(4)

    for col, k in zip(gauge_cols, ["TL", "TR", "BL", "BR"]):
        c = corners.get(k, {})
        st_key = norm_status(c.get("status"))
        fg = STATUS_COLORS[st_key][0]

        dev = c.get("deviation", 0)
        norm = min(dev / (twarn * 1.3 if twarn else 1), 1.0)

        with col:
            st.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
                f'color:#3a4560;margin-bottom:3px;">{k} DEVIATION</div>'

                f'<div style="background:#1e2535;border-radius:2px;height:8px;">'
                f'<div style="background:{fg};height:8px;width:{norm*100:.1f}%;border-radius:2px;"></div>'
                f'</div>'

                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:10px;'
                f'color:{fg};margin-top:3px;">{dev:.1f} mm</div>',
                unsafe_allow_html=True,
            )

    st.divider()

# Final summary card
if st.session_state["stage"] >= len(STAGES) and results:
    statuses = []
    for key in ["camera", "conductivity", "ai", "load", "laser"]:
        if key in results:
            statuses.append(results[key].get("status", "OK"))

    overall = "FAIL" if "FAIL" in statuses else ("WARNING" if "WARNING" in statuses else "OK")
    color   = {"OK": "#00ff99", "WARNING": "#ffb300", "FAIL": "#ff3b5c"}[overall]
    icon    = {"OK": "✓", "WARNING": "⚠", "FAIL": "✗"}[overall]

    st.markdown(
        f'<div style="background:rgba(0,0,0,0.3);border:2px solid {color};border-radius:4px;'
        f'padding:20px;text-align:center;margin:16px 0;">'
        f'<div style="font-size:36px;margin-bottom:8px;">{icon}</div>'
        f'<div style="font-family:Rajdhani,sans-serif;font-weight:700;font-size:1.6rem;'
        f'color:{color};letter-spacing:3px;text-transform:uppercase;">Overall: {overall}</div>'
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:12px;'
        f'color:#6b7a99;margin-top:6px;">{st.session_state["pallet_id"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─── Idle State Prompt ────────────────────────────────────────────────────────
if st.session_state["stage"] == -1:
    st.markdown(
        '<div style="text-align:center;padding:48px 0;">'
        '<div style="font-size:56px;margin-bottom:16px;">📦</div>'
        '<p style="font-family:Rajdhani,sans-serif;font-weight:600;font-size:1.2rem;'
        'color:#3a4560;letter-spacing:2px;text-transform:uppercase;">'
        'No pallet in queue — press Start Inspection</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# ─── Log Panel ────────────────────────────────────────────────────────────────
if st.session_state["log"]:
    with st.expander("📋 Inspection Log", expanded=False):
        log_lines = "\n".join(reversed(st.session_state["log"]))
        st.markdown(
            f'<div style="background:#0b0d12;border:1px solid #1e2535;border-radius:3px;'
            f'padding:12px;font-family:\'Share Tech Mono\',monospace;font-size:11px;'
            f'color:#6b7a99;line-height:1.8;max-height:280px;overflow-y:auto;">'
            f'{log_lines.replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True,
        )