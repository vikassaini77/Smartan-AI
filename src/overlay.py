
import cv2
def draw_landmarks(frame, landmarks, status_color=(0,255,0)):
    # simple circles for provided landmarks
    for idx, (x,y,_) in landmarks.items():
        try:
            cv2.circle(frame, (int(x), int(y)), 4, status_color, -1)
        except:
            pass

def draw_feedback_text(frame, feedback_msgs):
    y0 = 30
    for i, msg in enumerate(feedback_msgs[:6]):
        cv2.putText(frame, msg, (10, y0 + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return frame
