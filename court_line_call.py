import cv2
import numpy as np
import pandas as pd

CSV_PATH = "TrackNetV2/Professional/match6/csv/1_05_03_ball.csv"
VIDEO_PATH = "TrackNetV2/Professional/match6/video/1_05_03.mp4"
OUTPUT_IMAGE = "hawk_eye_decision.png"

df = pd.read_csv(CSV_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Cannot open video")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, sample_frame = cap.read()
if not ret:
    raise IOError("Cannot read video frame")
frame_height = sample_frame.shape[0]
frame_width = sample_frame.shape[1]

print(f"Video: {frame_width}x{frame_height}, {total_frames} frames")
print(f"CSV has {len(df)} entries, {len(df[df['Visibility']==1])} visible")

# Get visible shuttle positions from CSV
visible_df = df[df["Visibility"] == 1].reset_index(drop=True)

if len(visible_df) < 5:
    raise ValueError("Not enough visible frames")

# ===== STEP 1: DETECT START AND END OF PLAY =====
def detect_play_boundaries(visible_df, frame_height):
    """
    Detect when play starts and ends:
    - Play starts: first time shuttle appears (after frame 0)
    - Play ends: last time shuttle is visible
    """
    frames = visible_df["Frame"].values
    y_positions = visible_df["Y"].values
    
    # Skip frame 0 - find first frame after 0
    play_start_idx = 0
    for i, frame_num in enumerate(frames):
        if frame_num > 0:
            play_start_idx = i
            break
    
    # Play ends at last visible frame
    play_end_idx = len(visible_df) - 1
    
    start_frame = int(frames[play_start_idx])
    end_frame = int(frames[play_end_idx])
    
    print(f"\nPlay boundaries detected:")
    print(f"  Start: Frame {start_frame} (CSV index {play_start_idx})")
    print(f"  End: Frame {end_frame} (CSV index {play_end_idx})")
    
    return play_start_idx, play_end_idx

play_start_idx, play_end_idx = detect_play_boundaries(visible_df, frame_height)

# ===== STEP 2: FIND LAST GROUND IMPACT =====
def find_last_ground_impact(visible_df, start_idx, end_idx, frame_height):
    """
    Find the LAST ground impact in the rally by analyzing trajectory.
    Ground impact indicators:
    1. Shuttle is in lower portion of frame (ground region)
    2. Y-velocity decreases (shuttle stops descending)
    3. After ground impact, shuttle either stops or bounces up
    """
    
    # Work backwards from end to find last ground impact
    ground_candidates = []
    
    for i in range(start_idx, end_idx + 1):
        frame_num = int(visible_df.loc[i, "Frame"])
        x = float(visible_df.loc[i, "X"])
        y = float(visible_df.loc[i, "Y"])
        y_ratio = y / frame_height
        
        # Only consider frames in ground region (lower 30% of frame)
        if y_ratio < 0.70:
            continue
        
        # Calculate velocities
        if i > start_idx and i < end_idx:
            prev_y = float(visible_df.loc[i-1, "Y"])
            next_y = float(visible_df.loc[i+1, "Y"])
            
            y_velocity = y - prev_y  # Positive = moving down
            next_velocity = next_y - y
            velocity_change = y_velocity - next_velocity
            
            ground_candidates.append({
                'index': i,
                'frame': frame_num,
                'x': x,
                'y': y,
                'y_ratio': y_ratio,
                'y_velocity': y_velocity,
                'next_velocity': next_velocity,
                'velocity_change': abs(velocity_change)
            })
    
    if not ground_candidates:
        # Fallback: use frame with highest Y (lowest point)
        max_y_idx = visible_df.loc[start_idx:end_idx, "Y"].idxmax()
        frame_num = int(visible_df.loc[max_y_idx, "Frame"])
        x = float(visible_df.loc[max_y_idx, "X"])
        y = float(visible_df.loc[max_y_idx, "Y"])
        print(f"\nNo ground candidates found, using lowest point:")
        print(f"  Frame {frame_num}: ({x:.1f}, {y:.1f})")
        return max_y_idx
    
    print(f"\nFound {len(ground_candidates)} ground contact candidates:")
    for candidate in ground_candidates:
        print(f"  Frame {candidate['frame']}: "
              f"pos=({candidate['x']:.0f}, {candidate['y']:.0f}), "
              f"y={candidate['y_ratio']*100:.0f}%, "
              f"vel={candidate['y_velocity']:.1f}→{candidate['next_velocity']:.1f}")
    
    # Find LAST ground impact - work backwards
    # Look for last frame where shuttle was descending and then stopped/bounced
    for candidate in reversed(ground_candidates):
        # Descending (vel > 0) and then stops/reverses (next_vel <= 0 or much smaller)
        if candidate['y_velocity'] > 1.5 and candidate['next_velocity'] < 1.0:
            print(f"\n*** LAST GROUND IMPACT: Frame {candidate['frame']} ***")
            print(f"    Velocity changed from {candidate['y_velocity']:.1f} to {candidate['next_velocity']:.1f}")
            return candidate['index']
        
        # Or reversed direction (bounce)
        if candidate['y_velocity'] > 1.5 and candidate['next_velocity'] < -1.5:
            print(f"\n*** LAST GROUND IMPACT (bounce): Frame {candidate['frame']} ***")
            print(f"    Velocity reversed from {candidate['y_velocity']:.1f} to {candidate['next_velocity']:.1f}")
            return candidate['index']
    
    # Fallback: use last candidate (closest to end of rally)
    last_candidate = ground_candidates[-1]
    print(f"\n*** LAST GROUND IMPACT (fallback): Frame {last_candidate['frame']} ***")
    return last_candidate['index']

ground_idx = find_last_ground_impact(visible_df, play_start_idx, play_end_idx, frame_height)
ground_frame = int(visible_df.loc[ground_idx, "Frame"])
ground_x = float(visible_df.loc[ground_idx, "X"])
ground_y = float(visible_df.loc[ground_idx, "Y"])

print(f"\nFinal ground impact selection:")
print(f"  Frame: {ground_frame}")
print(f"  Position: ({ground_x:.1f}, {ground_y:.1f})")
print(f"  Y-ratio: {ground_y/frame_height*100:.1f}%")

# Show trajectory around ground impact
print(f"\nTrajectory around last ground impact:")
for i in range(max(play_start_idx, ground_idx-3), min(play_end_idx+1, ground_idx+4)):
    frame_num = int(visible_df.loc[i, "Frame"])
    x = float(visible_df.loc[i, "X"])
    y = float(visible_df.loc[i, "Y"])
    marker = " <== LAST GROUND IMPACT" if i == ground_idx else ""
    print(f"  Frame {frame_num}: ({x:.1f}, {y:.1f}) - {y/frame_height*100:.0f}%{marker}")

# ===== STEP 3: DETECT COURT LINES =====
def white_line_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 160])
    upper = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

def accumulate_court_evidence(cap, center_frame, window=20):
    acc = None
    for f in range(center_frame-window, center_frame+window):
        if f < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue
        mask = white_line_mask(frame).astype(np.float32)
        acc = mask if acc is None else acc + mask
    if acc is not None:
        acc = (acc / acc.max() * 255).astype(np.uint8)
    return acc

print("\nDetecting court lines...")
court_map = accumulate_court_evidence(cap, ground_frame)

edges = cv2.Canny(court_map, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=120, maxLineGap=15)
if lines is None:
    raise RuntimeError("No court lines detected")

def extend_line(line, scale=3000):
    x1, y1, x2, y2 = line
    dx, dy = x2-x1, y2-y1
    return int(x1-dx*scale), int(y1-dy*scale), int(x2+dx*scale), int(y2+dy*scale)

def get_line_angle(line):
    x1, y1, x2, y2 = line
    angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
    return min(angle, 180-angle)

def get_line_center(line):
    x1, y1, x2, y2 = line
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def get_line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

extended_lines = [extend_line(l[0]) for l in lines]

# Separate into horizontal and vertical lines
all_horizontal = []
all_vertical = []

for line in extended_lines:
    angle = get_line_angle(line)
    if angle < 30:
        all_horizontal.append(line)
    elif angle > 60:
        all_vertical.append(line)

print(f"\nDetected {len(all_horizontal)} horizontal lines and {len(all_vertical)} vertical lines")

# ===== INTELLIGENT COURT RECONSTRUCTION =====
# Camera is from behind, so we might see:
# - Near baseline (bottom of image) - always visible
# - Far baseline (top of image) - might be occluded
# - Both sidelines - usually visible but might be partial

def reconstruct_court_rectangle(all_horizontal, all_vertical, frame_width, frame_height, shuttle_pos):
    """
    Reconstruct court rectangle handling occlusion from behind-view camera.
    Strategy:
    1. Bottom baseline: lowest horizontal line (closest to camera)
    2. Top baseline: highest horizontal line OR extrapolate if missing
    3. Sidelines: leftmost and rightmost vertical lines
    4. Use shuttle position to validate court makes sense
    """
    shuttle_x, shuttle_y = shuttle_pos
    
    # Sort horizontal lines by Y position
    horizontal_by_y = sorted(all_horizontal, key=lambda l: get_line_center(l)[1])
    
    # Bottom baseline: line closest to bottom of frame (near camera)
    if len(horizontal_by_y) > 0:
        bottom_baseline = horizontal_by_y[-1]
        bottom_y = get_line_center(bottom_baseline)[1]
        print(f"  Bottom baseline found at y={bottom_y:.0f}")
    else:
        # Fallback: create baseline at bottom of frame
        bottom_y = frame_height - 50
        bottom_baseline = (0, int(bottom_y), frame_width, int(bottom_y))
        print(f"  Bottom baseline estimated at y={bottom_y:.0f}")
    
    # Top baseline: line closest to top OR extrapolate
    if len(horizontal_by_y) >= 2:
        # Use the topmost detected line
        top_baseline = horizontal_by_y[0]
        top_y = get_line_center(top_baseline)[1]
        print(f"  Top baseline found at y={top_y:.0f}")
    elif len(horizontal_by_y) == 1:
        # Only one baseline detected - estimate the other
        # Typical badminton court from behind: far baseline is ~60-70% up from bottom
        bottom_y = get_line_center(horizontal_by_y[0])[1]
        court_depth = frame_height * 0.6  # Estimate court takes 60% of frame height
        top_y = max(50, bottom_y - court_depth)
        top_baseline = (0, int(top_y), frame_width, int(top_y))
        print(f"  Top baseline extrapolated at y={top_y:.0f}")
    else:
        # No horizontal lines - use frame proportions
        top_y = frame_height * 0.3
        top_baseline = (0, int(top_y), frame_width, int(top_y))
        print(f"  Top baseline estimated at y={top_y:.0f}")
    
    # Sort vertical lines by X position
    vertical_by_x = sorted(all_vertical, key=lambda l: get_line_center(l)[0])
    
    # Sidelines: leftmost and rightmost
    if len(vertical_by_x) >= 2:
        left_sideline = vertical_by_x[0]
        right_sideline = vertical_by_x[-1]
        left_x = get_line_center(left_sideline)[0]
        right_x = get_line_center(right_sideline)[0]
        print(f"  Left sideline found at x={left_x:.0f}")
        print(f"  Right sideline found at x={right_x:.0f}")
    elif len(vertical_by_x) == 1:
        # Only one sideline detected - extrapolate the other
        detected_x = get_line_center(vertical_by_x[0])[0]
        
        # Determine if this is left or right based on shuttle position
        if shuttle_x < detected_x:
            # Detected line is on right, shuttle is on left
            right_sideline = vertical_by_x[0]
            right_x = detected_x
            # Estimate left sideline
            court_width = frame_width * 0.6
            left_x = max(50, right_x - court_width)
            left_sideline = (int(left_x), 0, int(left_x), frame_height)
            print(f"  Right sideline found at x={right_x:.0f}")
            print(f"  Left sideline extrapolated at x={left_x:.0f}")
        else:
            # Detected line is on left, shuttle is on right
            left_sideline = vertical_by_x[0]
            left_x = detected_x
            # Estimate right sideline
            court_width = frame_width * 0.6
            right_x = min(frame_width - 50, left_x + court_width)
            right_sideline = (int(right_x), 0, int(right_x), frame_height)
            print(f"  Left sideline found at x={left_x:.0f}")
            print(f"  Right sideline extrapolated at x={right_x:.0f}")
    else:
        # No sidelines detected - use frame proportions centered on shuttle
        court_width = frame_width * 0.6
        center_x = shuttle_x
        left_x = max(50, center_x - court_width/2)
        right_x = min(frame_width - 50, center_x + court_width/2)
        left_sideline = (int(left_x), 0, int(left_x), frame_height)
        right_sideline = (int(right_x), 0, int(right_x), frame_height)
        print(f"  Sidelines estimated at x={left_x:.0f} and x={right_x:.0f}")
    
    # Validate court makes sense
    court_width = abs(right_x - left_x)
    court_height = abs(bottom_y - top_y)
    
    print(f"\n  Reconstructed court:")
    print(f"    Width: {court_width:.0f}px")
    print(f"    Height: {court_height:.0f}px")
    print(f"    Aspect ratio: {court_width/court_height:.2f}")
    
    return [top_baseline, bottom_baseline], [left_sideline, right_sideline]

horizontal_lines, vertical_lines = reconstruct_court_rectangle(
    all_horizontal, all_vertical, frame_width, frame_height, (ground_x, ground_y)
)

# ===== STEP 4: CHECK IF SHUTTLE IS INSIDE COURT RECTANGLE =====
def point_to_line_dist(pt, line):
    x0, y0 = pt
    x1, y1, x2, y2 = line
    return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.hypot(y2-y1, x2-x1)

def point_side_of_line(pt, line):
    """
    Determine which side of line the point is on.
    Returns: positive if on one side, negative if on other side
    """
    x0, y0 = pt
    x1, y1, x2, y2 = line
    return (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1

def is_point_inside_court(pt, horizontal_lines, vertical_lines, line_thickness_px=5.0):
    """
    Check if point is inside the court rectangle using badminton rules:
    - Shuttle touching or inside the line = IN
    - Shuttle completely outside all lines = OUT
    
    Court boundary = outer edges of the 4 lines (top, bottom, left, right)
    """
    x, y = pt
    
    if len(horizontal_lines) < 2:
        return "UNKNOWN", 0.0, None, "need 2 baselines"
    
    if len(vertical_lines) < 2:
        return "UNKNOWN", 0.0, None, "need 2 sidelines"
    
    # Get boundaries
    top_baseline = horizontal_lines[0]
    bottom_baseline = horizontal_lines[1]
    left_sideline = vertical_lines[0]
    right_sideline = vertical_lines[1]
    
    # Get Y coordinates of horizontal lines
    top_y = get_line_center(top_baseline)[1]
    bottom_y = get_line_center(bottom_baseline)[1]
    
    # Get X coordinates of vertical lines  
    left_x = get_line_center(left_sideline)[0]
    right_x = get_line_center(right_sideline)[0]
    
    # Calculate distances to each boundary
    dist_top = abs(y - top_y)
    dist_bottom = abs(y - bottom_y)
    dist_left = abs(x - left_x)
    dist_right = abs(x - right_x)
    
    # Find closest boundary
    min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
    if min_dist == dist_top:
        closest_line = top_baseline
        closest_name = "top baseline"
    elif min_dist == dist_bottom:
        closest_line = bottom_baseline
        closest_name = "bottom baseline"
    elif min_dist == dist_left:
        closest_line = left_sideline
        closest_name = "left sideline"
    else:
        closest_line = right_sideline
        closest_name = "right sideline"
    
    # Check if point is inside the court rectangle
    # Add line_thickness_px to account for line width (shuttle on line = IN)
    tolerance = line_thickness_px
    
    inside_horizontal = (y >= top_y - tolerance) and (y <= bottom_y + tolerance)
    inside_vertical = (x >= left_x - tolerance) and (x <= right_x + tolerance)
    
    print(f"\n  Court boundary check:")
    print(f"    Shuttle Y: {y:.1f}, Court Y range: [{top_y:.1f}, {bottom_y:.1f}]")
    print(f"    Shuttle X: {x:.1f}, Court X range: [{left_x:.1f}, {right_x:.1f}]")
    print(f"    Inside horizontal bounds: {inside_horizontal}")
    print(f"    Inside vertical bounds: {inside_vertical}")
    print(f"    Closest boundary: {closest_name} ({min_dist:.1f}px)")
    
    # Badminton rule: ON the line = IN
    # Very close to line (within line thickness) = touching line = IN
    if inside_horizontal and inside_vertical:
        if min_dist <= line_thickness_px * 2:
            decision = "IN"
            confidence = 0.92
            reason = f"on/touching {closest_name}"
        else:
            decision = "IN"
            confidence = 0.97
            reason = f"clearly inside court ({min_dist:.1f}px from {closest_name})"
    else:
        # Outside the court rectangle
        decision = "OUT"
        confidence = 0.95
        
        out_directions = []
        if y < top_y - tolerance:
            out_directions.append(f"above top baseline by {top_y - y:.1f}px")
        if y > bottom_y + tolerance:
            out_directions.append(f"below bottom baseline by {y - bottom_y:.1f}px")
        if x < left_x - tolerance:
            out_directions.append(f"left of left sideline by {left_x - x:.1f}px")
        if x > right_x + tolerance:
            out_directions.append(f"right of right sideline by {x - right_x:.1f}px")
        
        reason = ", ".join(out_directions)
    
    boundaries = {
        'top': top_baseline,
        'bottom': bottom_baseline,
        'left': left_sideline,
        'right': right_sideline,
        'closest': closest_line,
        'closest_name': closest_name,
        'closest_dist': min_dist
    }
    
    return decision, confidence, boundaries, reason

final_decision, final_confidence, boundaries, reason_text = is_point_inside_court(
    (ground_x, ground_y), horizontal_lines, vertical_lines
)

print(f"\nFinal Decision: {final_decision}")
print(f"Confidence: {final_confidence*100:.1f}%")
print(f"Reason: {reason_text}")

# ===== STEP 5: GENERATE OUTPUT IMAGE =====
cap.set(cv2.CAP_PROP_POS_FRAMES, ground_frame)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read ground frame")

# Draw the court rectangle (all 4 boundary lines)
if boundaries:
    line_color = (0, 255, 255)  # Cyan for court lines
    line_thickness = 3
    
    # Draw all 4 boundary lines
    for boundary_name in ['top', 'bottom', 'left', 'right']:
        x1, y1, x2, y2 = boundaries[boundary_name]
        cv2.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)
    
    # Draw the complete court rectangle to make it clear
    top_y = int(get_line_center(boundaries['top'])[1])
    bottom_y = int(get_line_center(boundaries['bottom'])[1])
    left_x = int(get_line_center(boundaries['left'])[0])
    right_x = int(get_line_center(boundaries['right'])[0])
    
    # Draw corner markers
    corner_size = 15
    corner_color = (0, 255, 255)
    corners = [
        (left_x, top_y),
        (right_x, top_y),
        (right_x, bottom_y),
        (left_x, bottom_y)
    ]
    for corner in corners:
        cv2.circle(frame, corner, corner_size, corner_color, 2)
    
    # Highlight violated boundary if OUT
    if final_decision == "OUT" and 'closest' in boundaries:
        x1, y1, x2, y2 = boundaries['closest']
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Red for violated boundary

# Draw shuttle position
cx, cy = int(ground_x), int(ground_y)
color = (0, 255, 0) if final_decision == "IN" else (0, 0, 255)

# Large circle for shuttle
cv2.circle(frame, (cx, cy), 12, color, -1)
cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 3)

# Draw crosshair
cv2.line(frame, (cx-20, cy), (cx+20, cy), color, 2)
cv2.line(frame, (cx, cy-20), (cx, cy+20), color, 2)

# Draw line to closest boundary
if boundaries and 'closest' in boundaries:
    closest_dist = boundaries.get('closest_dist', 0)
    x1, y1, x2, y2 = boundaries['closest']
    
    # Find perpendicular point on line
    dx, dy = x2 - x1, y2 - y1
    if dx*dx + dy*dy > 0:
        t = ((cx - x1) * dx + (cy - y1) * dy) / (dx*dx + dy*dy)
        nearest_x = int(x1 + t * dx)
        nearest_y = int(y1 + t * dy)
        
        # Draw measurement line
        cv2.line(frame, (cx, cy), (nearest_x, nearest_y), (255, 0, 255), 2)
        cv2.circle(frame, (nearest_x, nearest_y), 6, (255, 0, 255), -1)
        
        # Draw distance text near the line
        mid_x = (cx + nearest_x) // 2
        mid_y = (cy + nearest_y) // 2
        cv2.putText(frame, f"{closest_dist:.1f}px", (mid_x + 10, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Decision text
label = f"{final_decision}"
text_color = (0, 255, 0) if final_decision == "IN" else (0, 0, 255)
cv2.putText(frame, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, text_color, 6)

# Additional info
cv2.putText(frame, f"Confidence: {final_confidence*100:.1f}%", (50, 140), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
cv2.putText(frame, f"Frame: {ground_frame}", (50, 180), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(frame, f"{reason_text}", (50, 220), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

cv2.imwrite(OUTPUT_IMAGE, frame)

print("\n" + "="*60)
print("HAWK-EYE DECISION SYSTEM - LAST GROUND IMPACT")
print("="*60)
print(f"Play Start: Frame {int(visible_df.loc[play_start_idx, 'Frame'])}")
print(f"Play End: Frame {int(visible_df.loc[play_end_idx, 'Frame'])}")
print(f"Last Ground Impact: Frame {ground_frame}")
print(f"Shuttle Position: ({ground_x:.2f}, {ground_y:.2f})")
print(f"\nCourt Rectangle Check:")
print(f"  Point inside court rectangle: {final_decision != 'OUT'}")
print(f"\nFINAL DECISION: {final_decision}")
print(f"Confidence: {final_confidence*100:.1f}%")
print(f"Reason: {reason_text}")
print(f"\nOutput: {OUTPUT_IMAGE}")
print("="*60 + "\n")

cap.release()