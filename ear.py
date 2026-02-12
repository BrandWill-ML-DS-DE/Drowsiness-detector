# ear.py
from scipy.spatial import distance

def calculate_ear(eye_landmarks):
    """
    Computes Eye Aspect Ratio (EAR)

    eye_landmarks: list of 6 (x,y) tuples
    """

    # Vertical distances
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])

    # Horizontal distance
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])

    ear = (A + B) / (2.0 * C)
    return ear

