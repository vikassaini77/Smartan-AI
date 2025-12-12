import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# Get the absolute path to the project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "lateral_raise.mp4")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "task1_submission.mp4")

print(f"--- SMARTAN AI TASK ---")
print(f"Working Directory: {BASE_DIR}")
print(f"Reading Video: {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    print("❌ ERROR: Video file missing. Please make sure data/lateral_raise.mp4 exists.")
    sys.exit()

print("Loading AI Models... (Please wait)")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(INPUT_FILE)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

print(f"Processing {total} frames...")

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Simple Logic: Check Wrist Height
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
        left_shldr = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        
        msg = "Good Form"
        color = (0, 255, 0)
        
        # Remember: Smaller Y is HIGHER on screen
        if left_wrist < (left_shldr - 0.1): 
            msg = "TOO HIGH!"
            color = (0, 0, 255)

        cv2.putText(image, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    out.write(image)
    count += 1
    if count % 10 == 0: print(f" -> Frame {count}/{total}")

cap.release()
out.release()
print(f"✅ DONE. Video saved to: {OUTPUT_FILE}")
