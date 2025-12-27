import cv2
import numpy as np

VIDEO = "TrackNetV2/Test/match2/video/2_08_12.mp4"

OUT_CAMERA = "output_camera_view.png"
OUT_ZOOM = "output_ground_zoom.png"
OUT_COURT = "output_top_court.png"

# =========================================================
# 1. PLAYER COUNT → GAME TYPE
# =========================================================
def detect_game_type(video_path, sample_frames=20):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    indices = np.linspace(int(total*0.2), int(total*0.8), sample_frames, dtype=int)
    counts = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        _, weights = hog.detectMultiScale(frame)
        counts.append(len([w for w in weights if w>0.5]))
    cap.release()
    return np.median(counts) >= 3.5 if counts else True

# =========================================================
# 2. ENHANCED SHUTTLE TRACKING WITH KALMAN FILTER
# =========================================================
class ShuttleTracker:
    def __init__(self):
        # Kalman filter setup
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],
                                                  [0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],
                                                 [0,1,0,1],
                                                 [0,0,1,0],
                                                 [0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32)*0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32)*0.5
        self.last_pos = None

    def detect_in_frame(self, frame, fg):
        H, W = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value = hsv[:,:,2]
        mask_white = cv2.inRange(value, 190, 255)  # higher threshold for shuttle
        mask = cv2.bitwise_and(fg, mask_white)
        mask[int(H*0.8):,:] = 0  # ignore lower area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel)
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best, score_best = None, 0
        for c in contours:
            area = cv2.contourArea(c)
            if not 5<area<500:
                continue
            x,y,w,h = cv2.boundingRect(c)
            cx,cy = x+w//2, y+h//2
            score = 0
            # area scoring
            if 10<area<200: score+=3
            elif 200<=area<400: score+=2
            # aspect ratio
            ratio = w/h if h>0 else 0
            if 0.5<ratio<1.8: score+=2
            # proximity to last position
            if self.last_pos and np.linalg.norm([cx-self.last_pos[0], cy-self.last_pos[1]])<120:
                score+=3
            # position preference (middle court)
            if H*0.3<cy<H*0.8: score+=1
            if score>score_best:
                best, score_best = (cx,cy), score

        # Kalman prediction
        prediction = self.kalman.predict()
        if best:
            measurement = np.array([[np.float32(best[0])],
                                    [np.float32(best[1])]])
            self.kalman.correct(measurement)
            self.last_pos = best
        else:
            # fallback to prediction
            best = (int(prediction[0]), int(prediction[1]))
        return best if score_best>=4 else None

def detect_trajectory(video):
    cap = cv2.VideoCapture(video)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bg = cv2.createBackgroundSubtractorMOG2(500,16,False)
    tracker = ShuttleTracker()
    traj, f = [], 0
    while True:
        ret,frame=cap.read()
        if not ret: break
        fg=bg.apply(frame)
        pos = tracker.detect_in_frame(frame, fg)
        if pos:
            traj.append({"frame":f,"x":pos[0],"y":pos[1]})
        f+=1
    cap.release()
    return traj,H,W

def smooth_trajectory(traj, window=3):
    if not traj: return traj
    xs = [p['x'] for p in traj]
    ys = [p['y'] for p in traj]
    xs_smooth = np.convolve(xs, np.ones(window)/window, mode='same')
    ys_smooth = np.convolve(ys, np.ones(window)/window, mode='same')
    for i,p in enumerate(traj):
        p['x']=int(xs_smooth[i])
        p['y']=int(ys_smooth[i])
    return traj

# =========================================================
# 3. IMPACT
# =========================================================
def detect_impact(traj):
    return traj[-1]

# =========================================================
# 4. CAMERA TRAJECTORY
# =========================================================
def draw_camera_trajectory(frame,traj,impact):
    out = frame.copy()
    pts = [(p['x'],p['y']) for p in traj]
    if len(pts)>1:
        cv2.polylines(out,[np.array(pts)],False,(0,255,255),2)
    for x,y in pts:
        cv2.circle(out,(x,y),2,(0,200,255),-1)
    cv2.circle(out,(impact['x'],impact['y']),12,(0,0,255),-1)
    return out

# =========================================================
# 5. SIDE LINE DETECTION
# =========================================================
def detect_side_lines(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,50,150,apertureSize=3)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,200,minLineLength=300,maxLineGap=20)
    verticals=[]
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if abs(x1-x2)<10 and abs(y1-y2)>200:
                verticals.append((x1,y1,x2,y2))
    if len(verticals)<2:
        H,W=frame.shape[:2]
        return (int(W*0.1),0,int(W*0.1),H),(int(W*0.9),0,int(W*0.9),H)
    verticals=sorted(verticals,key=lambda v:v[0])
    return verticals[0],verticals[-1]

# =========================================================
# 6. FULL BADMINTON COURT DRAWING
# =========================================================
COURT_L = 13.41
COURT_W = 6.10
PPM = 90
MARGIN = 80
TOP_H = int(COURT_L*PPM)+MARGIN*2
TOP_W = int(COURT_W*PPM)+MARGIN*2
NET_Y = COURT_L/2

def px(x,y):
    return int(x*PPM+MARGIN), int(y*PPM+MARGIN)

def draw_full_badminton_court():
    court = np.zeros((TOP_H,TOP_W,3),np.uint8)
    cv2.rectangle(court,px(0,0),px(COURT_W,COURT_L),(255,255,255),3)
    cv2.line(court,px(0,NET_Y),px(COURT_W,NET_Y),(150,150,150),3)
    cv2.line(court, px(COURT_W/2,0), px(COURT_W/2,COURT_L), (150,150,150),2)
    SSL = 1.98
    cv2.line(court, px(0,SSL), px(COURT_W,SSL),(0,255,0),2)
    cv2.line(court, px(0,COURT_L-SSL), px(COURT_W,COURT_L-SSL),(0,255,0),2)
    LSL = 0.76
    cv2.line(court, px(0,LSL), px(COURT_W,LSL),(0,255,0),1)
    cv2.line(court, px(0,COURT_L-LSL), px(COURT_W,COURT_L-LSL),(0,255,0),1)
    SSW = 5.18
    cv2.line(court, px(0,0), px(0,COURT_L),(0,200,0),2)
    cv2.line(court, px(COURT_W,0), px(COURT_W,COURT_L),(0,200,0),2)
    for m in range(1,11):
        x1,y1=px(0,m*COURT_L/10)
        x2,y2=px(COURT_W,m*COURT_L/10)
        cv2.line(court,(x1,y1),(x2,y2),(0,200,0),1)
        cv2.putText(court,str(m),(3,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,0),1)
        x1,y1=px(m*COURT_W/10,0)
        x2,y2=px(m*COURT_W/10,COURT_L)
        cv2.line(court,(x1,y1),(x2,y2),(0,200,0),1)
        cv2.putText(court,str(m),(x1+3,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,0),1)
    poly=np.array([px(0,0),px(COURT_W,0),px(COURT_W,COURT_L),px(0,COURT_L)])
    return court,poly

# =========================================================
# 7. MAIN PIPELINE
# =========================================================
print("Detecting game type...")
is_doubles=detect_game_type(VIDEO)

traj,Hc,Wc=detect_trajectory(VIDEO)
traj=smooth_trajectory(traj)
impact=detect_impact(traj)

cap=cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES,impact['frame'])
_,frame=cap.read()
cap.release()

left,right=detect_side_lines(frame)
px_width=abs(right[0]-left[0])
meters_per_pixel=COURT_W/px_width

x_m=(impact['x']-left[0])*meters_per_pixel
y_m=impact['y']/Hc*COURT_L
x_m=np.clip(x_m,0,COURT_W)
y_m=np.clip(y_m,0,COURT_L)
impact_px=px(x_m,y_m)

camera_view=draw_camera_trajectory(frame,traj,impact)
cv2.imwrite(OUT_CAMERA,camera_view)

court,poly=draw_full_badminton_court()
inside=cv2.pointPolygonTest(poly,impact_px,False)>=0
color=(0,255,0) if inside else (0,0,255)
cv2.circle(court,(impact_px[0]+5,impact_px[1]+5),14,(40,40,40),-1)
cv2.circle(court,impact_px,12,color,-1)
cv2.putText(court,"IN" if inside else "OUT",(30,50),cv2.FONT_HERSHEY_DUPLEX,1.6,color,3)
cv2.imwrite(OUT_COURT,court)

z=120
x1,y1=max(0,impact_px[0]-z),max(0,impact_px[1]-z)
x2,y2=min(TOP_W,impact_px[0]+z),min(TOP_H,impact_px[1]+z)
zoom=cv2.resize(court[y1:y2,x1:x2],(400,400))
cv2.imwrite(OUT_ZOOM,zoom)

print("✓ Fully scaled badminton court Hawk-Eye outputs saved")