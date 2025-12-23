import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

CSV = "TrackNetV2/Professional/match16/csv/2_08_08_ball.csv"
VIDEO = "TrackNetV2/Professional/match16/video/2_08_08.mp4"
OUTPUT = f"hawkeye_ground_{VIDEO[38:]}.png"

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

# ----------------- 3. LOAD IMPACT FRAME -----------------
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, impact["frame"])
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError(f"Cannot read impact frame {impact['frame']}")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ----------------- 4. DETECT COURT LINES (ROTATED) -----------------
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
v_channel = hsv[:,:,2]
blur = cv2.GaussianBlur(v_channel,(5,5),0)
edges = cv2.Canny(blur,30,120,apertureSize=3)
lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=50,minLineLength=50,maxLineGap=20)

# store candidate lines with their angle
candidate_lines = []
if lines is not None:
    for x1,y1,x2,y2 in lines[:,0]:
        angle = np.arctan2((y2-y1),(x2-x1))  # in radians
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length > 50:  # ignore tiny lines
            candidate_lines.append((x1,y1,x2,y2,angle))

if not candidate_lines:
    raise RuntimeError("No candidate lines detected")

# ----------------- 4b. CLUSTER LINES INTO HORIZONTAL/VERTICAL -----------------
# dominant horizontal: around 0 or pi
h_lines = [l[:4] for l in candidate_lines if abs(np.sin(l[4])) < 0.2]
# dominant vertical: around pi/2
v_lines = [l[:4] for l in candidate_lines if abs(np.cos(l[4])) < 0.2]

def select_extreme_lines(lines, axis=1):
    """Pick top/bottom for horizontal (axis=1), left/right for vertical (axis=0)"""
    if not lines:
        return None, None
    coords = np.array([ (l[axis]+l[axis+2])/2 for l in lines ])
    min_line = lines[np.argmin(coords)]
    max_line = lines[np.argmax(coords)]
    return min_line, max_line

top_line, bottom_line = select_extreme_lines(h_lines, axis=1)
left_line, right_line = select_extreme_lines(v_lines, axis=0)

if None in [top_line,bottom_line,left_line,right_line]:
    raise RuntimeError("Court horizontal/vertical lines not detected")

court_polygon = np.array([
    [left_line[0],top_line[1]],
    [right_line[0],top_line[1]],
    [right_line[0],bottom_line[1]],
    [left_line[0],bottom_line[1]]
])

# ----------------- 5. IN/OUT VERDICT -----------------
impact_point = (impact["x"],impact["y_ground"])
inside = cv2.pointPolygonTest(court_polygon,impact_point,False)>=0
verdict = "IN" if inside else "OUT"
color = (0,255,0) if inside else (0,0,255)

# ----------------- 6. DRAW COURT AND IMPACT -----------------
result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# draw court lines
for l in [top_line,bottom_line,left_line,right_line]:
    x1,y1,x2,y2 = l
    cv2.line(result,(x1,y1),(x2,y2),(255,255,255),2)

# ----------------- 6b. PROJECTION LINES OUTSIDE COURT -----------------
def project_line_outside(polygon, impact_point, direction='vertical'):
    """Return a point outside the court polygon along given direction."""
    x, y = impact_point
    xs = polygon[:,0]
    ys = polygon[:,1]

    if direction == 'vertical':
        top_y = min(ys)
        bottom_y = max(ys)
        if abs(y - top_y) < abs(y - bottom_y):
            return (x, top_y - 20)  # 20 pixels outside
        else:
            return (x, bottom_y + 20)
    else:  # horizontal
        left_x = min(xs)
        right_x = max(xs)
        if abs(x - left_x) < abs(x - right_x):
            return (left_x - 20, y)
        else:
            return (right_x + 20, y)

# draw projection lines from outside the court
vert_start = project_line_outside(court_polygon, impact_point, 'vertical')
hor_start = project_line_outside(court_polygon, impact_point, 'horizontal')
cv2.line(result, vert_start, impact_point, color, 2)
cv2.line(result, hor_start, impact_point, color, 2)

# impact shadow
shadow = np.zeros_like(result,dtype=np.float32)
cv2.circle(shadow,impact_point,85,(120,120,120),-1)
cv2.circle(shadow,impact_point,40,(180,180,180),-1)
shadow = cv2.GaussianBlur(shadow,(101,101),60)
result = cv2.addWeighted(result,1.0,shadow.astype(np.uint8),0.9,0)

# impact marker
cv2.circle(result,impact_point,7,color,-1)
cv2.circle(result,impact_point,18,color,2)

# verdict text
cv2.putText(result,f"VERDICT: {verdict}",(40,60),cv2.FONT_HERSHEY_DUPLEX,1.6,color,3)
cv2.putText(result,"HAWK-EYE LINE CALL",(40,110),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2)

cv2.imwrite(OUTPUT,result)

print("\n==============================")
print("IMPACT FRAME:",impact["frame"])
print("IMPACT POINT:",impact_point)
print("VERDICT:",verdict)
print("OUTPUT:",OUTPUT)
print("==============================\n")