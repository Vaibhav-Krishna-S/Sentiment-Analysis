import cv2
import mediapipe as mp
import math

# Initialize the MediaPipe pose detection utility
mp_pose = mp.solutions.pose

# Create an instance of the pose detection model
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5)

# Function to detect poses and count squats
def detect_poses(image):
    global squat_count, squat_position, prev_squat_position

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with the pose detection model
    results = pose.process(image_rgb)

    # Draw the detected poses on the image
    mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Check if a pose is detected
    if results.pose_landmarks:
        # Get the coordinates of the left and right hip joints
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Get the coordinates of the left and right knee joints
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

        # Calculate the angle between the hip and knee joints
        left_angle = calculate_angle(left_hip, left_knee, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE])
        right_angle = calculate_angle(right_hip, right_knee, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE])

        # Determine the squat position
        if left_angle > 160 and right_angle > 160:
            squat_position = "Squatting"
        else:
            squat_position = "Standing"

        # Count the number of squats
        if squat_position == "Squatting" and squat_position != prev_squat_position:
            squat_count += 1

    # Update the previous squat position
    prev_squat_position = squat_position

    # Display the squat count on the image
    cv2.putText(image, f"Squats: {squat_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    angle_radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees + 360 if angle_degrees < 0 else angle_degrees

# Initialize the squat count and position
squat_count = 0
squat_position = "Standing"
prev_squat_position = "Standing"

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Detect poses and count squats in the frame
    output_frame = detect_poses(frame)

    # Display the output frame
    cv2.imshow('Personalized Gym Trainer', output_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
