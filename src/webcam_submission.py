import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- 1. CONFIGURATION ---
OUTPUT_FILE = os.path.join("outputs", "task1_submission.mp4")
RECORD_SECONDS = 8

# --- 2. GEOMETRY HELPERS ---
def calculate_angle(a, b, c):
    """Calculates the angle at point b given points a, b, c."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# --- 3. POSTURE RULES (REQUIRED: 3 RULES) ---

def check_lateral_raise(landmarks):
    """Rule 1: Lateral Raise - Wrists should not go above shoulders."""
    l_shldr = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    
    # Logic: Smaller Y means higher on screen
    if l_wrist.y < (l_shldr.y - 0.1):
        return "ARMS TOO HIGH!", (0, 0, 255) # Red
    return "Good Form", (0, 255, 0) # Green

def check_bicep_curl(landmarks):
    """Rule 2: Bicep Curl - Elbows should remain tucked (stationary)."""
    # Get coordinates
    shldr = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    hip   = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    
    # Calculate angle of shoulder-elbow-hip. Should be ~180 (straight line)
    angle = calculate_angle(shldr, elbow, hip)
    
    if angle < 160: # Elbow swinging forward
        return "KEEP ELBOWS TUCKED", (0, 0, 255)
    return "Good Curl Form", (0, 255, 0)

def check_back_symmetry(landmarks):
    """Rule 3: Back Symmetry - Shoulders should be level."""
    l_shldr_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    r_shldr_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    
    diff = abs(l_shldr_y - r_shldr_y)
    
    if diff > 0.05: # Significant tilt
        return "UNBALANCED POSTURE", (0, 0, 255)
    return "Good Symmetry", (0, 255, 0)

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"--- SMARTAN AI TASK 1 SUBMISSION ---")
    print("1. Initializing AI...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0
    
    os.makedirs("outputs", exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"2. Camera Active! Recording for {RECORD_SECONDS} seconds...")
    start_time = time.time()

    while (time.time() - start_time) < RECORD_SECONDS:
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # We use Rule 1 for this video demo
            msg, color = check_lateral_raise(results.pose_landmarks.landmark)
            
            cv2.putText(image, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(image)
        remaining = int(RECORD_SECONDS - (time.time() - start_time))
        print(f"   Recording... {remaining}s left", end='\r')

    cap.release()
    out.release()
    print(f"\n\n✅ DONE! Video saved to: {OUTPUT_FILE}")