# Smartan.AI - Form Correctness Detection (Internship Task)

This repository contains a runnable demo pipeline for form correctness detection using pose estimation.
It includes:
- MediaPipe-based pose detector (if MediaPipe is installed)
- A fallback **fake** pose generator so the demo runs even without MediaPipe
- Rule-based evaluation for **Bicep Curl**, **Lateral Raise**, and **Back Posture**
- Smoothing utilities, overlay renderer, and MLflow logger (optional)
- Sample video `data/sample_videos/sample_bicep.mp4`
- A generated demo overlay at `outputs/overlay_demo.mp4` (after running)

## Quick start (recommended)
1. Extract the ZIP.
2. Create and activate a Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   *Note:* If `mediapipe` installation fails on your platform, the demo will automatically use a fake pose mode that still demonstrates the overlay and evaluation output.

3. Run the demo (uses the included sample video):
   ```bash
   python src/run_demo.py --input data/sample_videos/sample_bicep.mp4 --output outputs/overlay_demo.mp4
   ```

4. Check `outputs/overlay_demo.mp4` for the overlay video and `outputs/logs.txt` for per-frame feedback.

## Repo structure
```
smartan-internship-cv/
├─ data/sample_videos/      # sample video(s)
├─ src/
│  ├─ pose_detector.py
│  ├─ metrics.py
│  ├─ smoothing.py
│  ├─ evaluator.py
│  ├─ overlay.py
│  ├─ mlflow_logger.py
│  └─ run_demo.py
├─ outputs/
├─ requirements.txt
├─ README.md
└─ report.pdf (or report.md)
```

## Notes
- The included sample video is synthetic and created so you can run the pipeline immediately.
- For real usage, replace the sample video with your recorded clips (3-5 seconds each) in `data/sample_videos/`.
- The evaluator includes thresholds and comments; tune them using your real videos.

Good luck — go smash that internship assignment! ✨
# Smartan-AI
