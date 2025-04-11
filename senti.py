import cv2
import mediapipe as mp
import math

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Improved set of facial landmarks
LANDMARKS = {
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "left_eyebrow": 70,
    "right_eyebrow": 300,
    "nose_tip": 1,
    "chin": 152
}

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def analyze_mood(landmarks):
    eye_width = distance(landmarks["left_eye_outer"], landmarks["right_eye_outer"])
    mouth_width = distance(landmarks["mouth_left"], landmarks["mouth_right"])
    mouth_height = distance(landmarks["mouth_top"], landmarks["mouth_bottom"])
    eyebrow_distance = distance(landmarks["left_eyebrow"], landmarks["right_eyebrow"])
    nose_to_chin = distance(landmarks["nose_tip"], landmarks["chin"])

    mouth_openness = mouth_height / nose_to_chin
    smile_ratio = mouth_width / eye_width

    # Improved Mood Logic
    if mouth_openness > 0.45:
        return "Surprised"
    elif smile_ratio > 1.8 and mouth_openness > 0.2:
        return "Happy"
    elif smile_ratio < 1.5 and mouth_openness < 0.15:
        return "Neutral or Sad"
    else:
        return "Neutral"

# OpenCV video capture
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark_coords = {}
                for name, idx in LANDMARKS.items():
                    lm = face_landmarks.landmark[idx]
                    landmark_coords[name] = (int(lm.x * w), int(lm.y * h))

                mood = analyze_mood(landmark_coords)

                # Draw mood text
                cv2.putText(frame, f"Mood: {mood}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Optionally, draw landmarks
                for point in landmark_coords.values():
                    cv2.circle(frame, point, 2, (255, 0, 255), -1)

        cv2.imshow("Sentiment Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
