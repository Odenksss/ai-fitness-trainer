import cv2
import mediapipe as mp
import numpy as np
import time
from core.utils import (
    calculate_angle, extract_landmarks, get_landmark_array,
    draw_landmarks, mp_pose
)
from core.session_report import SessionLogger, calculate_rep_score
from core.voice_feedback import VoiceFeedback

# Body alignment thresholds (degrees)
BODY_ALIGNMENT_UPPER = 195  # hip above this → high lower back error
BODY_ALIGNMENT_LOWER = 165  # hip below this → low lower back error


class PlankDetector:
    def __init__(self):
        self.errors = []
        self.high_lower_back = False
        self.low_lower_back = False
        self.hold_time = 0
        self.score = None

    def reset(self):
        self.errors = []
        self.high_lower_back = False
        self.low_lower_back = False
        self.hold_time = 0
        self.score = None

    def detect(self, landmarks):
        """Analyze landmarks and return error feedback."""
        errors = []
        body_angle = None
        neck_angle = None
        current_score = None

        try:
            # Use left side for primary check
            l_shoulder = get_landmark_array(landmarks, "left_shoulder")
            l_hip = get_landmark_array(landmarks, "left_hip")
            l_ankle = get_landmark_array(landmarks, "left_ankle")

            # Angle at hip: shoulder -> hip -> ankle
            body_angle = calculate_angle(l_shoulder, l_hip, l_ankle)

            if body_angle > BODY_ALIGNMENT_UPPER:
                errors.append("High lower back")
                self.high_lower_back = True
                self.low_lower_back = False
            elif body_angle < BODY_ALIGNMENT_LOWER:
                errors.append("Low lower back")
                self.low_lower_back = True
                self.high_lower_back = False
            else:
                self.high_lower_back = False
                self.low_lower_back = False

            # Head alignment check (neck angle)
            nose = get_landmark_array(landmarks, "nose")
            neck_angle = calculate_angle(nose, l_shoulder, l_hip)
            if neck_angle < 150:
                errors.append("Head dropped")

            current_score = calculate_rep_score([
                {"angle": body_angle, "target": 180, "tolerance": 25, "weight": 2},
                {"angle": neck_angle, "target": 180, "tolerance": 30, "weight": 1},
            ])

        except Exception as e:
            pass

        self.errors = errors
        self.score = current_score
        return {
            "errors": errors,
            "body_angle": body_angle,
            "score": current_score,
        }


def run_plank_detection(source=0):
    """Run real-time plank detection."""
    detector = PlankDetector()
    session = SessionLogger("Plank", unit_label="Second")
    voice = VoiceFeedback(cooldown=5.0)
    cap = cv2.VideoCapture(source)
    start_time = time.time()
    last_logged_second = 0

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
                elapsed_seconds = int(time.time() - start_time)

                if elapsed_seconds > last_logged_second and data["score"] is not None:
                    session.log(elapsed_seconds, data["score"], data["errors"])
                    last_logged_second = elapsed_seconds

                # Voice feedback
                voice.speak_errors(data["errors"])

                cv2.rectangle(image, (0, 0), (380, 120), (0, 0, 0), -1)
                cv2.putText(image, "Exercise: PLANK", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, f"Hold: {elapsed_seconds}s", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                score_text = "-" if data["score"] is None else f"{data['score']:.1f}%"
                cv2.putText(image, f"Score: {score_text}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                y_offset = 145
                if data["errors"]:
                    for err in data["errors"]:
                        cv2.putText(image, f"! {err}", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        y_offset += 35
                else:
                    cv2.putText(image, "Good form!", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Plank Correction", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    session.show_summary_chart()


if __name__ == "__main__":
    run_plank_detection(0)
