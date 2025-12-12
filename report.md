# Report - Form Correctness Detection using Pose Estimation

## Authors
- Candidate Name 1
- Candidate Name 2

## Overview
This project creates a rule-based pipeline that uses human pose estimation to evaluate exercise form for:
- Bicep Curl
- Lateral Raise
- Back posture alignment

## Posture Rules and Logic
1. **Bicep Curl - Elbow Angle**
   - Angle at elbow (shoulder-elbow-wrist).
   - Full flex: elbow angle < 60°. Full extension: elbow angle > 160°.
   - Rule: If elbow never goes below 60° during rep → incomplete flex. If top extension < 160° → incomplete extension.

2. **Lateral Raise - Arm Elevation & Alignment**
   - Elevation angle compared to torso; wrist roughly level with shoulder at peak.
   - Rule: Peak elevation < 70° → "not high enough". Excess forward/back offset → wrong plane.

3. **Back Posture - Torso Tilt & Symmetry**
   - Tilt between shoulders and hips; left/right shoulder heights.
   - Rule: Shoulder-hip tilt > 12° → "asymmetric/twisted back".

## Implementation Notes
- Pose detection: MediaPipe (preferred). The code falls back to a simulated pose generator when MediaPipe isn't available.
- Smoothing: Exponential Moving Average (simple) and optional Savitzky-Golay smoothing for derivatives.
- Evaluation: Frame-wise checks and window-based aggregation.
- Multi-person handling: prioritize largest bounding box or nearest pose to frame center; track using centroid distance across frames.

## Challenges
- Occlusion and incorrect landmark detection.
- Multi-person scenarios: ambiguous identity and overlaps. Strategies: bbox-area priority, temporal matching, or require single-person demos.
- Camera angle variability: thresholds may need calibration per user and camera.

## How to run
See README.md. The `run_demo.py` script processes `data/sample_videos/sample_bicep.mp4` by default and writes `outputs/overlay_demo.mp4`.

## Potential Improvements
- Per-user calibration of limb lengths and thresholds.
- Train a small classifier on computed features (angles, velocities) to improve robustness.
- Use OpenPose for stronger multi-person support.

