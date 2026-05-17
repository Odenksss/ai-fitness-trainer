import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def calculate_angle(a, b, c):
    """
    Calculate the angle at point b formed by points a, b, c.
    Each point is [x, y] or [x, y, z].
    Returns angle in degrees.
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    point1 = np.array(point1[:2])
    point2 = np.array(point2[:2])
    return np.linalg.norm(point1 - point2)


def extract_landmarks(results):
    """
    Extract landmark coordinates from MediaPipe results.
    Returns a dict of landmark_name -> [x, y, z, visibility]
    """
    if not results.pose_landmarks:
        return None

    landmarks = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        name = mp_pose.PoseLandmark(idx).name.lower()
        landmarks[name] = [lm.x, lm.y, lm.z, lm.visibility]

    return landmarks


def get_landmark_array(landmarks, name):
    """Get [x, y, z] array for a named landmark."""
    return landmarks[name][:3]


def rescale_frame(frame, percent=75):
    """Rescale a video frame to a given percentage."""
    import cv2
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def draw_landmarks(image, results):
    """Draw pose landmarks on image."""
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )