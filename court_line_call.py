import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

CSV = "TrackNetV2/Professional/match16/csv/1_13_20_ball.csv"
VIDEO = "TrackNetV2/Professional/match16/video/1_13_20.mp4"
OUTPUT = "hawkeye_full_court_shadow.png"

# ----------------- 1. LOAD DATA -----------------
df = pd.read_csv(CSV)
df = df[df["Visibility"] == 1].reset_index(drop=True)
frames = df["Frame"].values.astype(int)
xs = df["X"].values.astype(float)
ys = df["Y"].values.astype(float)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
dt = 1.0 / fps

# ----------------- 2. DETECT IMPACT FRAME -----------------
def detect_ground_impact(frames, xs, ys):
    win = min(15, len(ys)//2*2 + 1)
    ys_s = savgol_filter(ys, win, 2)
    vy = np.gradient(ys_s)
    ay = np.gradient(vy)
    tail_start = None
    for i in range(len(vy)-8):
        if np.all(vy[i:i+5] > 1.0):
            tail_start = i
            break
    if tail_start is None:
        raise RuntimeError("No falling phase detected")
    g_est = max(np.median(ay[tail_start:]), 1.2)
    i_last = len(ys_s)-1
    y_last = ys_s[i_last]
    v_last = vy[i_last]
    ground_y = int(np.max(ys[max(0,i_last-5):]))
    a = 0.5 * g_est
    b = v_last
    c = y_last - ground_y
    disc = b*b - 4*a*c
    t_hit = (-b + np.sqrt(disc))/(2*a) if disc>0 else dt
    frames_after = max(1,int(round(t_hit/dt)))
    impact_frame = min(frames[i_last]+frames_after,TOTAL_FRAMES-1)
    return {"frame":impact_frame,"x":int(xs[i_last]),"y_ground":ground_y}

impact = detect_ground_impact(frames, xs, ys)

# ----------------- 3. COURT SCALED DIMENSIONS -----------------
COURT_LENGTH = 13.41   # 44 ft
COURT_WIDTH = 6.10     # 20 ft doubles
SINGLES_WIDTH = 5.18   # 17 ft singles
SHORT_SERVICE_LINE_DIST = 1.98
PIXELS_PER_METER = 80
TOP_H = int(COURT_LENGTH * PIXELS_PER_METER) + 100
TOP_W = int(COURT_WIDTH * PIXELS_PER_METER)*2 + 150
OFFSET_X = 50
OFFSET_Y = 50

result = np.zeros((TOP_H, TOP_W, 3), dtype=np.uint8)

# ----------------- 4. CONVERSION -----------------
def to_pixels(x_m, y_m, offset_x=OFFSET_X, offset_y=OFFSET_Y):
    return (int(x_m*PIXELS_PER_METER)+offset_x, int(y_m*PIXELS_PER_METER)+offset_y)

# ----------------- 5. DRAW COURTS -----------------
def draw_court(start_x, start_y, doubles=True):
    court_w = COURT_WIDTH if doubles else SINGLES_WIDTH
    color_outer = (255,255,255) if doubles else (0,255,255)
    thickness_outer = 3 if doubles else 2
    color_inner = (180,180,180)
    
    outer_pts = np.array([
        to_pixels(0,0,start_x,start_y),
        to_pixels(court_w,0,start_x,start_y),
        to_pixels(court_w,COURT_LENGTH,start_x,start_y),
        to_pixels(0,COURT_LENGTH,start_x,start_y)
    ], np.int32)
    cv2.polylines(result,[outer_pts],True,color_outer,thickness_outer)
    
    # Net
    cv2.line(result, to_pixels(0,COURT_LENGTH/2,start_x,start_y),
             to_pixels(court_w,COURT_LENGTH/2,start_x,start_y), color_inner,2)
    
    # Short service lines
    cv2.line(result, to_pixels(0,SHORT_SERVICE_LINE_DIST,start_x,start_y),
             to_pixels(court_w,SHORT_SERVICE_LINE_DIST,start_x,start_y), color_inner,2)
    cv2.line(result, to_pixels(0,COURT_LENGTH-SHORT_SERVICE_LINE_DIST,start_x,start_y),
             to_pixels(court_w,COURT_LENGTH-SHORT_SERVICE_LINE_DIST,start_x,start_y), color_inner,2)
    
    # Center line
    cv2.line(result, to_pixels(court_w/2,SHORT_SERVICE_LINE_DIST,start_x,start_y),
             to_pixels(court_w/2,COURT_LENGTH-SHORT_SERVICE_LINE_DIST,start_x,start_y), color_inner,2)
    
    return outer_pts

doubles_polygon = draw_court(OFFSET_X, OFFSET_Y, doubles=True)
singles_polygon = draw_court(TOP_W//2 + 25, OFFSET_Y, doubles=False)

# ----------------- 6. LOAD VIDEO IMPACT FRAME -----------------
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, impact["frame"])
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError(f"Cannot read impact frame {impact['frame']}")

# Map shuttle drop point from video to court (doubles)
impact_x = np.interp(impact["x"], [0, W], [0, COURT_WIDTH])
impact_y = np.interp(impact["y_ground"], [0, H], [0, COURT_LENGTH])
impact_point = to_pixels(impact_x, impact_y, OFFSET_X, OFFSET_Y)

# ----------------- 7. DRAW SHADOW EFFECT -----------------
shadow = np.zeros_like(result, dtype=np.uint8)
cv2.circle(shadow, impact_point, 25, (60,60,60), -1)  # shadow larger
cv2.circle(shadow, impact_point, 15, (120,120,120), -1) # inner darker
shadow = cv2.GaussianBlur(shadow, (101,101), 40)
result = cv2.addWeighted(result,1.0,shadow,0.7,0)

# ----------------- 8. DRAW IMPACT POINT -----------------
inside = cv2.pointPolygonTest(doubles_polygon, impact_point, False) >= 0
color = (0,255,0) if inside else (0,0,255)
cv2.circle(result, impact_point, 12, color, -1)
cv2.circle(result, impact_point, 20, color, 2)

# ----------------- 9. VERDICT -----------------
verdict = "IN" if inside else "OUT"
cv2.putText(result, f"VERDICT: {verdict}", (20,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, color,3)
cv2.putText(result, "HAWK-EYE TOP VIEW", (20,90), cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)

# ----------------- 10. SAVE -----------------
cv2.imwrite(OUTPUT, result)
print("OUTPUT saved:", OUTPUT)
print("Impact Point:", impact_point)
print("VERDICT:", verdict)