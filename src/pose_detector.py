

import cv2
import numpy as np
import math

# Try to import MediaPipe, otherwise set a flag and the demo will use fake pose generator.
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_pose = mp.solutions.pose
except Exception as e:
    MP_AVAILABLE = False

class MediaPipeWrapper:
    def __init__(self, model_complexity=1, smooth=True):
        if not MP_AVAILABLE:
            raise RuntimeError("MediaPipe not available")
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):
        # frame: BGR image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)
        if not res.pose_landmarks:
            return None
        h, w = frame.shape[:2]
        landmarks = {}
        for idx, lm in enumerate(res.pose_landmarks.landmark):
            landmarks[idx] = (lm.x * w, lm.y * h, lm.z * w)
        return landmarks

    def close(self):
        self.pose.close()

# Fake pose generator for demo-mode (so the repo runs even without MediaPipe)
# It simulates a bicep-curl like motion for one person.
def fake_pose_generator(frame_idx, frame_count, frame_w, frame_h):
    # Generate landmarks dict with approximate locations for required indices
    # We will map to MediaPipe-like indices used in metrics.py
    # Provide nose, shoulders, elbows, wrists, hips
    # Animate elbow y to simulate curl
    t = frame_idx / max(1, frame_count-1)
    # elbow y oscillates: start down (y~0.65), go up to 0.45, back to 0.65
    import math
    cycle = 0.5 - 0.5*math.cos(2*math.pi*t)  # smooth transition 0->1->0 across frames
    # positions
    center_x = frame_w // 2
    nose = (center_x, int(frame_h*0.2), 0)
    left_shoulder = (center_x - 80, int(frame_h*0.3), 0)
    right_shoulder = (center_x + 80, int(frame_h*0.3), 0)
    left_elbow_y = int(frame_h*(0.65 - 0.2*cycle))
    right_elbow_y = left_elbow_y
    left_elbow = (left_shoulder[0], left_elbow_y, 0)
    right_elbow = (right_shoulder[0], right_elbow_y, 0)
    left_wrist = (left_elbow[0], int(left_elbow_y + 60), 0)
    right_wrist = (right_elbow[0], int(right_elbow_y + 60), 0)
    left_hip = (center_x - 50, int(frame_h*0.6), 0)
    right_hip = (center_x + 50, int(frame_h*0.6), 0)
    # Using subset of indices; metrics.py expects certain MP indices - we will use same keys there
    lm = {
        0: nose,
        11: left_shoulder, 12: right_shoulder,
        13: left_elbow, 14: right_elbow,
        15: left_wrist, 16: right_wrist,
        23: left_hip, 24: right_hip
    }
    return lm

# Wrapper that chooses real MediaPipe if available else fake generator
class PoseDetector:
    def __init__(self, use_mediapipe=True):
        self.use_mediapipe = use_mediapipe and MP_AVAILABLE
        if self.use_mediapipe:
            self.detector = MediaPipeWrapper()
        else:
            self.detector = None

    def detect(self, frame, frame_idx=None, frame_count=None):
        if self.use_mediapipe:
            return self.detector.detect(frame)
        else:
            h, w = frame.shape[:2]
            return fake_pose_generator(frame_idx or 0, frame_count or 1, w, h)

    def close(self):
        if self.detector:
            self.detector.close()
