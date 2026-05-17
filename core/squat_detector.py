import cv2
import mediapipe as mp
import numpy as np
from core.utils import (
    calculate_angle, calculate_distance, extract_landmarks,
    get_landmark_array, draw_landmarks, mp_pose
)
from core.session_report import SessionLogger, calculate_rep_score
from core.voice_feedback import VoiceFeedback

SQUAT_DOWN_THRESHOLD = 100
SQUAT_UP_THRESHOLD   = 160
SQUAT_DEPTH_MIN      = 110


class SquatDetector:
    def __init__(self):
        self.stage   = None
        self.counter = 0
        self.errors  = []
        self.last_rep_score = None

    def reset(self):
        self.stage   = None
        self.counter = 0
        self.errors  = []
        self.last_rep_score = None

    def detect(self, landmarks):
        errors = []
        rep_completed = False
        current_score = None

        try:
            r_hip   = get_landmark_array(landmarks, "right_hip")
            r_knee  = get_landmark_array(landmarks, "right_knee")
            r_ankle = get_landmark_array(landmarks, "right_ankle")
            l_hip   = get_landmark_array(landmarks, "left_hip")
            l_knee  = get_landmark_array(landmarks, "left_knee")
            l_ankle = get_landmark_array(landmarks, "left_ankle")

            r_knee_angle  = calculate_angle(r_hip, r_knee, r_ankle)
            l_knee_angle  = calculate_angle(l_hip, l_knee, l_ankle)
            avg_knee_angle = (r_knee_angle + l_knee_angle) / 2

            # Track stage but don't count yet
            if avg_knee_angle > SQUAT_UP_THRESHOLD:
                self.stage = "up"
            reached_down = avg_knee_angle < SQUAT_DOWN_THRESHOLD and self.stage == "up"

            # Error 1: Not deep enough
            if reached_down and avg_knee_angle > SQUAT_DEPTH_MIN:
                errors.append("Not deep enough")

            # Error 2: Knees caving in
            knee_width  = abs(r_knee[0] - l_knee[0])
            ankle_width = abs(r_ankle[0] - l_ankle[0])
            if reached_down and knee_width < ankle_width * 0.85:
                errors.append("Knees caving in")

            # Error 3: Leaning too far forward
            l_shoulder = get_landmark_array(landmarks, "left_shoulder")
            back_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            if back_angle < 45:
                errors.append("Leaning too far forward")

            current_score = calculate_rep_score([
                {"angle": avg_knee_angle, "target": 90, "tolerance": 50, "weight": 2},
                {"angle": back_angle, "target": 80, "tolerance": 35, "weight": 1},
            ])

            # Count completed reps and attach their form score.
            if reached_down:
                self.stage = "down"
                self.counter += 1
                self.last_rep_score = current_score
                rep_completed = True

        except Exception:
            pass

        self.errors = errors
        return {
            "stage":   self.stage,
            "counter": self.counter,
            "errors":  errors,
            "score": current_score,
            "rep_score": self.last_rep_score,
            "completed_rep": rep_completed,
        }


def run_squat_detection(source=0):
    detector     = SquatDetector()
    session      = SessionLogger("Squat")
    voice        = VoiceFeedback(cooldown=2.0)
    prev_counter = 0
    cap          = cv2.VideoCapture(source)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            draw_landmarks(image, results)

            landmarks = extract_landmarks(results)
            if landmarks:
                data = detector.detect(landmarks)

                if data["completed_rep"]:
                    session.log(data["counter"], data["rep_score"], data["errors"])

                if data["counter"] != prev_counter:
                    voice.speak_rep(data["counter"])
                    prev_counter = data["counter"]
                else:
                    voice.speak_errors(data["errors"])

                cv2.rectangle(image, (0, 0), (380, 115), (0, 0, 0), -1)
                cv2.putText(image, f"Stage: {data['stage'] or '-'}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, f"Reps: {data['counter']}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                score_text = "-" if data["rep_score"] is None else f"{data['rep_score']:.1f}%"
                cv2.putText(image, f"Last score: {score_text}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                y_offset = 145
                for err in data["errors"]:
                    cv2.putText(image, f"! {err}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 35

            cv2.imshow("Squat Correction", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    session.show_summary_chart()


if __name__ == "__main__":
    run_squat_detection(0)
