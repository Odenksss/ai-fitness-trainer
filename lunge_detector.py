import cv2
import mediapipe as mp
import numpy as np
from core.utils import (
    calculate_angle, extract_landmarks, get_landmark_array,
    draw_landmarks, mp_pose
)
from core.session_report import SessionLogger, calculate_rep_score
from core.voice_feedback import VoiceFeedback

KNEE_LOWER_THRESHOLD = 60
KNEE_UPPER_THRESHOLD = 135
LUNGE_DOWN_THRESHOLD = 110
LUNGE_UP_THRESHOLD   = 160


class LungeDetector:
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
            r_foot  = get_landmark_array(landmarks, "right_foot_index")
            l_hip   = get_landmark_array(landmarks, "left_hip")
            l_knee  = get_landmark_array(landmarks, "left_knee")
            l_ankle = get_landmark_array(landmarks, "left_ankle")
            l_foot  = get_landmark_array(landmarks, "left_foot_index")

            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)

            if r_knee_angle < l_knee_angle:
                front_knee_angle = r_knee_angle
                front_knee = r_knee
                front_foot = r_foot
            else:
                front_knee_angle = l_knee_angle
                front_knee = l_knee
                front_foot = l_foot

            # Track stage but don't count yet
            if front_knee_angle > LUNGE_UP_THRESHOLD:
                self.stage = "up"
            reached_down = front_knee_angle < LUNGE_DOWN_THRESHOLD and self.stage == "up"

            # Error 1: Knee angle out of range
            if front_knee_angle < KNEE_LOWER_THRESHOLD:
                errors.append("Knee bends too much")
            elif front_knee_angle > KNEE_UPPER_THRESHOLD and reached_down:
                errors.append("Knee not bent enough")

            # Error 2: Knee over toe
            if front_knee[0] > front_foot[0] + 0.02:
                errors.append("Knee over toe")

            # Error 3: Torso lean
            l_shoulder = get_landmark_array(landmarks, "left_shoulder")
            r_shoulder = get_landmark_array(landmarks, "right_shoulder")
            mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2,
                            (l_shoulder[1] + r_shoulder[1]) / 2,
                            (l_shoulder[2] + r_shoulder[2]) / 2]
            mid_hip = [(l_hip[0] + r_hip[0]) / 2,
                       (l_hip[1] + r_hip[1]) / 2,
                       (l_hip[2] + r_hip[2]) / 2]
            torso_angle = calculate_angle(
                [mid_shoulder[0], mid_shoulder[1] - 0.1, mid_shoulder[2]],
                mid_shoulder, mid_hip
            )
            if torso_angle < 155:
                errors.append("Lean forward")

            current_score = calculate_rep_score([
                {"angle": front_knee_angle, "target": 90, "tolerance": 45, "weight": 2},
                {"angle": torso_angle, "target": 180, "tolerance": 25, "weight": 1},
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


def run_lunge_detection(source=0):
    detector     = LungeDetector()
    session      = SessionLogger("Lunge")
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

            cv2.imshow("Lunge Correction", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    session.show_summary_chart()


if __name__ == "__main__":
    run_lunge_detection(0)
