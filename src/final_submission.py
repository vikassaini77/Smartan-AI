import os
# --- FIX: FORCE DISABLE WINDOWS MEDIA FOUNDATION ---
# This prevents the "Hanging/Freezing" on Windows
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import mediapipe as mp
import sys

# --- SETTINGS ---
MAX_FRAMES = 150 

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "lateral_raise.mp4")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "task1_submission.mp4")

print(f"--- FINAL FIX ATTEMPT ---")
print(f"1. Target Video: {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    print("❌ ERROR: Video file not found!")
    sys.exit()

print("2. Initializing AI Models...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Use the default driver (0) now that we disabled MSMF
cap = cv2.VideoCapture(INPUT_FILE)

if not cap.isOpened():
    print("❌ ERROR: Video failed to open. The file might be corrupt.")
    sys.exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
fps = 25

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

print("3. Processing started... Numbers MUST move below:")

count = 0
while True:
    ret, frame = cap.read()
    if not ret or count > MAX_FRAMES:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Simple Logic: Check Wrist Height
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
        left_shldr = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        
        msg = "Good Form"
        color = (0, 255, 0)
        if left_wrist < (left_shldr - 0.1): 
            msg = "TOO HIGH!"
            color = (0, 0, 255)

        cv2.putText(image, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    out.write(image)
    count += 1
    
    # Print status
    print(f"   >>> Processed Frame {count} ...", end='\r')

cap.release()
out.release()
print(f"\n\n✅ SUCCESS! Video saved to: {OUTPUT_FILE}")
