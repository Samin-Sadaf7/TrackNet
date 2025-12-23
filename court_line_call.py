import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

CSV = "TrackNetV2/Professional/match16/csv/2_08_08_ball.csv"
VIDEO = "TrackNetV2/Professional/match16/video/2_08_08.mp4"
OUTPUT = "hawkeye_ground_hit.png"

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(CSV)
df = df[df["Visibility"] == 1].reset_index(drop=True)

frames = df["Frame"].values.astype(int)
xs = df["X"].values.astype(float)
ys = df["Y"].values.astype(float)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

dt = 1.0 / fps

# ------------------------------------------------------------
# 2. ROBUST PHYSICS-BASED IMPACT DETECTION
# ------------------------------------------------------------
def detect_ground_impact(frames, xs, ys):
    # Smooth Y
    win = min(15, len(ys) // 2 * 2 + 1)
    ys_s = savgol_filter(ys, win, 2)

    vy = np.gradient(ys_s)
    ay = np.gradient(vy)

    # Find falling tail
    tail_start = None
    for i in range(len(vy) - 8):
        if np.all(vy[i:i+5] > 1.0):
            tail_start = i
            break

    if tail_start is None:
        raise RuntimeError("No falling phase detected")

    # Estimate gravity
    g_est = np.median(ay[tail_start:])
    g_est = max(g_est, 1.2)  # stabilize

    # Last visible point
    i_last = len(ys_s) - 1
    y_last = ys_s[i_last]
    v_last = vy[i_last]

    # Ground Y = max Y in last visible window (correct)
    ground_y = int(np.max(ys[max(0, i_last - 5):]))

    # Solve y = y0 + v*t + 0.5*g*t^2
    a = 0.5 * g_est
    b = v_last
    c = y_last - ground_y

    disc = b*b - 4*a*c
    if disc < 0:
        t_hit = dt
    else:
        t_hit = (-b + np.sqrt(disc)) / (2*a)

    frames_after = max(1, int(round(t_hit / dt)))
    predicted_frame = frames[i_last] + frames_after

    # 🔒 CLAMP TO VIDEO RANGE (CRITICAL FIX)
    impact_frame = min(predicted_frame, TOTAL_FRAMES - 1)

    return {
        "frame": impact_frame,
        "x": int(xs[i_last]),
        "y_ground": ground_y,
        "last_visible_frame": frames[i_last],
        "predicted_frame": predicted_frame
    }

impact = detect_ground_impact(frames, xs, ys)

# ------------------------------------------------------------
# 3. LOAD IMPACT FRAME (SAFE)
# ------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, impact["frame"])
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError(
        f"Cannot read frame {impact['frame']} / {TOTAL_FRAMES}"
    )

# ------------------------------------------------------------
# 4. TRUE HAWK-EYE REPLAY
# ------------------------------------------------------------
def hawkeye_replay(frame, x, y):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(result, 0.75, np.zeros_like(result), 0.25, 0)

    shadow = np.zeros_like(result, dtype=np.float32)
    center = (x, y)

    cv2.circle(shadow, center, 20, (230, 230, 230), -1)
    cv2.circle(shadow, center, 50, (170, 170, 170), -1)
    cv2.circle(shadow, center, 90, (120, 120, 120), -1)

    shadow = cv2.GaussianBlur(shadow, (101, 101), 60)
    result = cv2.addWeighted(result, 1.0, shadow.astype(np.uint8), 0.9, 0)

    cv2.circle(result, center, 6, (0, 255, 0), -1)
    cv2.circle(result, center, 18, (0, 255, 0), 2)

    cv2.line(result, (x - 40, y), (x + 40, y), (0, 255, 0), 2)
    cv2.line(result, (x, y - 40), (x, y + 40), (0, 255, 0), 2)

    cv2.putText(result, "HAWK-EYE REPLAY",
                (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4,
                (255, 255, 255), 2)

    cv2.putText(result,
                f"Impact Frame: {impact['frame']}",
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2)

    return result

final = hawkeye_replay(frame, impact["x"], impact["y_ground"])
cv2.imwrite(OUTPUT, final)

# ------------------------------------------------------------
# 5. DEBUG OUTPUT
# ------------------------------------------------------------
print("\n==============================")
print("LAST VISIBLE FRAME:", impact["last_visible_frame"])
print("PREDICTED IMPACT FRAME:", impact["predicted_frame"])
print("USED IMPACT FRAME:", impact["frame"])
print("GROUND Y:", impact["y_ground"])
print("TOTAL VIDEO FRAMES:", TOTAL_FRAMES)
print("OUTPUT:", OUTPUT)
print("==============================\n")