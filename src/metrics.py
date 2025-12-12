
import numpy as np
from math import acos, degrees

def angle_between(a, b, c):
    # a, b, c are (x,y) tuples. returns angle at b formed by a-b-c in degrees
    a = np.array(a[:2]); b = np.array(b[:2]); c = np.array(c[:2])
    ab = a - b
    cb = c - b
    dot = np.dot(ab, cb)
    norm = np.linalg.norm(ab) * np.linalg.norm(cb)
    if norm == 0:
        return 0.0
    cosang = np.clip(dot / norm, -1.0, 1.0)
    return degrees(acos(cosang))

# MediaPipe indices used
MP = {
 'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
 'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
 'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
 'LEFT_HIP': 23, 'RIGHT_HIP': 24,
 'NOSE': 0
}

def elbow_angle(landmarks, side='LEFT'):
    if side == 'LEFT':
        s = MP['LEFT_SHOULDER']; e = MP['LEFT_ELBOW']; w = MP['LEFT_WRIST']
    else:
        s = MP['RIGHT_SHOULDER']; e = MP['RIGHT_ELBOW']; w = MP['RIGHT_WRIST']
    if e not in landmarks or s not in landmarks or w not in landmarks:
        return None
    return angle_between(landmarks[s], landmarks[e], landmarks[w])

def shoulder_hip_tilt(landmarks):
    # tilt of shoulders: angle of line between shoulders relative to horizontal (degrees)
    if MP['LEFT_SHOULDER'] not in landmarks or MP['RIGHT_SHOULDER'] not in landmarks:
        return None
    ls = np.array(landmarks[MP['LEFT_SHOULDER']][:2])
    rs = np.array(landmarks[MP['RIGHT_SHOULDER']][:2])
    dx, dy = rs - ls
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    return angle

def arm_elevation(landmarks, side='LEFT'):
    # angle between vector (shoulder->wrist) and vertical axis; 0 = arm down, 90 = horizontal
    if side == 'LEFT':
        s = MP['LEFT_SHOULDER']; w = MP['LEFT_WRIST']
    else:
        s = MP['RIGHT_SHOULDER']; w = MP['RIGHT_WRIST']
    if s not in landmarks or w not in landmarks:
        return None
    sx, sy = landmarks[s][:2]; wx, wy = landmarks[w][:2]
    # vector from shoulder to wrist
    vx, vy = wx - sx, sy - wy  # invert y to make vertical upwards positive
    # angle from vertical (0 deg) to vector
    import math
    vmag = math.hypot(vx, vy)
    if vmag == 0:
        return 0.0
    # vertical vector is (0,1)
    dot = vy / vmag
    dot = max(min(dot,1.0), -1.0)
    angle = math.degrees(math.acos(dot))
    return angle
