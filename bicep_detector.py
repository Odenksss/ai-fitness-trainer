import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from core.utils import (
    calculate_angle, extract_landmarks, get_landmark_array,
    draw_landmarks, mp_pose
)
from core.session_report import SessionLogger, calculate_rep_score
from core.voice_feedback import VoiceFeedback

# Stage thresholds
CURL_UP_THRESHOLD = 60
CURL_DOWN_THRESHOLD = 160
LOOSE_UPPER_ARM_THRESHOLD = 40


class BicepCurlDetector:
    def __init__(self, model_path=None):
        self.stage = None
        self.counter = 0
        self.errors = []
        self.loose_upper_arm_error = False
        self.weak_contraction_error = False
        self.lean_back_error = False
        self.last_rep_score = None

        self.lean_model = None
        if model_path and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.lean_model = pickle.load(f)

    def reset(self):
        self.stage = None
        self.counter = 0
        self.errors = []
        self.loose_upper_arm_error = False
        self.weak_contraction_error = False
        self.lean_back_error = False
        self.last_rep_score = None

    def detect(self, landmarks):
        errors = []
        rep_completed = False
        current_score = None

        try:
            r_shoulder = get_landmark_array(landmarks, "right_shoulder")
            r_elbow    = get_landmark_array(landmarks, "right_elbow")
            r_wrist    = get_landmark_array(landmarks, "right_wrist")
            r_hip      = get_landmark_array(landmarks, "right_hip")

            curl_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

            # Track stage but don't count yet
            if curl_angle > CURL_DOWN_THRESHOLD:
                self.stage = "down"
            reached_top = curl_angle < CURL_UP_THRESHOLD and self.stage == "down"

            # --- Error 1: Loose upper arm ---
            shoulder_proj = [r_shoulder[0], r_elbow[1], r_shoulder[2]]
            upper_arm_angle = calculate_angle(shoulder_proj, r_shoulder, r_elbow)
            if upper_arm_angle > LOOSE_UPPER_ARM_THRESHOLD:
                errors.append("Loose upper arm")
                self.loose_upper_arm_error = True
            else:
                self.loose_upper_arm_error = False

            # --- Error 2: Weak peak contraction ---
            if reached_top and curl_angle > CURL_UP_THRESHOLD:
                errors.append("Weak peak contraction")
                self.weak_contraction_error = True
            else:
                self.weak_contraction_error = False

            # --- Error 3: Lean back ---
            l_shoulder = get_landmark_array(landmarks, "left_shoulder")
            l_hip      = get_landmark_array(landmarks, "left_hip")
            torso_angle = calculate_angle(
                [r_shoulder[0], r_shoulder[1] - 0.1, r_shoulder[2]],
                r_shoulder,
                r_hip
            )
            if torso_angle < 160:
                errors.append("Lean too far back")
                self.lean_back_error = True
            else:
                self.lean_back_error = False

            current_score = calculate_rep_score([
                {"angle": curl_angle, "target": 45, "tolerance": 35, "weight": 2},
                {"angle": upper_arm_angle, "target": 0, "tolerance": 40, "weight": 1},
                {"angle": torso_angle, "target": 180, "tolerance": 30, "weight": 1},
            ])

            # Count completed reps and attach their form score.
            if reached_top:
                self.stage = "up"
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


def run_bicep_curl_detection(source=0):
    detector     = BicepCurlDetector()
    session      = SessionLogger("Bicep Curl")
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

                # Voice feedback
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

            cv2.imshow("Bicep Curl Correction", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    session.show_summary_chart()


if __name__ == "__main__":
    run_bicep_curl_detection(0)
