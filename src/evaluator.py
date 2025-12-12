
from .metrics import elbow_angle, shoulder_hip_tilt, arm_elevation
def evaluate_frame(landmarks):
    """
    Returns a list of feedback strings for the frame
    """
    fb = []
    # Elbow rules
    le = elbow_angle(landmarks, 'LEFT')
    re = elbow_angle(landmarks, 'RIGHT')
    if le is not None:
        if le < 60:
            fb.append(f"Left elbow: full flex ({le:.1f}°) OK")
        elif le < 90:
            fb.append(f"Left elbow: partial flex ({le:.1f}°)")
        else:
            fb.append(f"Left elbow: not flexing ({le:.1f}°)")
    if re is not None:
        if re < 60:
            fb.append(f"Right elbow: full flex ({re:.1f}°) OK")
        elif re < 90:
            fb.append(f"Right elbow: partial flex ({re:.1f}°)")
        else:
            fb.append(f"Right elbow: not flexing ({re:.1f}°)")

    # Lateral raise / arm elevation
    la = arm_elevation(landmarks, 'LEFT')
    ra = arm_elevation(landmarks, 'RIGHT')
    if la is not None:
        if la > 70:
            fb.append(f"Left arm: good elevation ({la:.1f}°)")
        else:
            fb.append(f"Left arm: low elevation ({la:.1f}°)")
    if ra is not None:
        if ra > 70:
            fb.append(f"Right arm: good elevation ({ra:.1f}°)")
        else:
            fb.append(f"Right arm: low elevation ({ra:.1f}°)")

    # Back posture tilt
    tilt = shoulder_hip_tilt(landmarks)
    if tilt is not None:
        if tilt > 12:
            fb.append(f"Shoulder tilt high: {tilt:.1f}° (asymmetric)")
        else:
            fb.append(f"Shoulder tilt: {tilt:.1f}° (OK)")

    return fb
