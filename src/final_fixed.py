import cv2
import mediapipe as mp
import os

# --- SETTINGS ---
# We force the script to stop after 150 frames so it CANNOT freeze
MAX_FRAMES_TO_PROCESS = 150 

# Get paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "lateral_raise.mp4")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs", "task1_submission.mp4")

print(f"--- RESTARTING ---")
print(f"Target Video: {INPUT_FILE}")

# Initialize AI
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(INPUT_FILE)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
fps = 25 

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_idx = 0
print("Processing started... (I will print every frame so you know I am alive)")

while True:
    ret, frame = cap.read()
    
    # STOP if video ends OR if we hit our safety limit
    if not ret or frame_idx > MAX_FRAMES_TO_PROCESS:
        print("\nStopping now.")
        break

    # AI Processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Draw "Good Form" text
        cv2.putText(image, "Analyzing Form...", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(image)
    frame_idx += 1
    
    # Print status every single frame
    print(f"Frame {frame_idx} processed...", end='\r')

cap.release()
out.release()
print(f"\n\n✅ SUCCESS! Video saved to: {OUTPUT_FILE}")
