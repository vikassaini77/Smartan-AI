import cv2, os, argparse, time
from pathlib import Path
from src.pose_detector import PoseDetector
from src.smoothing import ema_smooth
from src.evaluator import evaluate_frame
from src.overlay import draw_landmarks, draw_feedback_text

def process_video(input_path, output_path, use_mediapipe=True):
    # REVERT: Removed cv2.CAP_FFMPEG to let OpenCV choose the best backend automatically
    cap = cv2.VideoCapture(str(input_path))
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w,h))
    
    detector = PoseDetector(use_mediapipe=use_mediapipe)
    history = []
    max_hist = 6
    
    os.makedirs("outputs", exist_ok=True)
    logs_path = Path("outputs") / "logs.txt"

    idx = 0
    # Note: frame_count might show as -1 for WebM files. This is normal.
    print(f"Processing video: {input_path} -> {output_path} (frames approx: {frame_count})")

    with open(logs_path, "w", encoding="utf-8") as log_file:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            lm = detector.detect(frame, frame_idx=idx, frame_count=frame_count)
            if lm is None:
                lm = {}
                
            history.append(lm)
            if len(history) > max_hist:
                history.pop(0)
                
            smoothed = ema_smooth(history, alpha=0.5)
            fb = evaluate_frame(smoothed)
            
            color = (0,255,0)
            bad_keywords = ['not','low','incomplete','asymmetric','not flexing']
            if any(any(k in m.lower() for k in bad_keywords) for m in fb):
                color = (0,0,255)
                
            draw_landmarks(frame, smoothed, status_color=color)
            draw_feedback_text(frame, fb)
            out.write(frame)

            line = f"frame {idx}: " + (" | ".join(fb))
            log_file.write(line + "\n")
            
            # IMPROVED LOGGING: Prints every 10 frames regardless of total count
            if (idx % 10) == 0:
                print(f" frame {idx} processed...")
                
            idx += 1

    cap.release()
    out.release()
    detector.close()

    try:
        size = logs_path.stat().st_size
    except:
        size = 0
    print(f"Done. Processed {idx} frames. Output written to {output_path}")
    return logs_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_videos/sample_bicep.mp4")
    parser.add_argument("--output", default="outputs/overlay_demo.mp4")
    parser.add_argument("--mediapipe", action="store_true", help="Force use MediaPipe if installed")
    args = parser.parse_args()
    process_video(args.input, args.output, use_mediapipe=args.mediapipe)