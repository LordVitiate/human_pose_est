import cv2
from ultralytics import YOLO

# Define the video stream URL
video_stream_url = 'rtsp://admin:!Qwerty1234@192.168.1.233:554/Streaming/Channels/101'

# Load the pre-trained YOLOv8 pose estimation model
pose_model = YOLO('yolov8n-pose.pt')  # Pose estimation model

# Define keypoint indices for COCO dataset
KEYPOINT_INDEX = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Open the video stream
cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Background subtraction
backSub = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Variables for motion detection
previous_positions = {}
min_contour_area = 500  # Minimum area threshold for contours

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video stream.")
        break

    # Apply background subtraction
    fg_mask = backSub.apply(frame)
    
    # Remove shadows using global thresholding
    _, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to clean up the mask
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    # Find contours in the cleaned mask
    contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # If no moving objects are detected, skip further processing
    if not large_contours:
        cv2.imshow('Human Pose Estimation', frame)  # Display the original frame
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        continue

    # Create a copy of the frame for drawing results
    frame_out = frame.copy()

    # Process each detected moving object
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Crop the frame to the bounding box of the moving object
        cropped_frame = frame[y:y+h, x:x+w]

        # Perform pose estimation only on the cropped frame
        results = pose_model(cropped_frame, conf=0.5)  # Adjust confidence threshold as needed

        # Process each detected person in the cropped frame
        for result in results:
            keypoints = result.keypoints.xy  # Get keypoints (x, y coordinates)
            boxes = result.boxes  # Get bounding boxes

            for kp in keypoints:
                # Extract relevant keypoints
                nose = kp[KEYPOINT_INDEX['nose']] if len(kp) > KEYPOINT_INDEX['nose'] else None
                left_wrist = kp[KEYPOINT_INDEX['left_wrist']] if len(kp) > KEYPOINT_INDEX['left_wrist'] else None
                right_wrist = kp[KEYPOINT_INDEX['right_wrist']] if len(kp) > KEYPOINT_INDEX['right_wrist'] else None
                left_hip = kp[KEYPOINT_INDEX['left_hip']] if len(kp) > KEYPOINT_INDEX['left_hip'] else None
                right_hip = kp[KEYPOINT_INDEX['right_hip']] if len(kp) > KEYPOINT_INDEX['right_hip'] else None
                left_knee = kp[KEYPOINT_INDEX['left_knee']] if len(kp) > KEYPOINT_INDEX['left_knee'] else None
                right_knee = kp[KEYPOINT_INDEX['right_knee']] if len(kp) > KEYPOINT_INDEX['right_knee'] else None

                # Check if keypoints exist
                if all([kp is not None for kp in [nose, left_hip, right_hip]]):
                    # Detect "Hand Above Head"
                    hand_above_head = (
                        (left_wrist is not None and left_wrist[1] < nose[1]) or
                        (right_wrist is not None and right_wrist[1] < nose[1])
                    )

                    # Detect "Lies" (hips are close to knees)
                    lies = (
                        abs(left_hip[1] - left_knee[1]) < 50 and  # Left hip near left knee
                        abs(right_hip[1] - right_knee[1]) < 50     # Right hip near right knee
                    )

                    # Detect "Goes" (motion detection based on previous position)
                    box = boxes[0].xyxy[0]  # Get the first bounding box
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    current_position = (x_center.item(), y_center.item())

                    goes = False
                    if id(result) in previous_positions:
                        prev_x, prev_y = previous_positions[id(result)]
                        distance_moved = ((current_position[0] - prev_x) ** 2 + (current_position[1] - prev_y) ** 2) ** 0.5
                        if distance_moved > 20:  # Threshold for movement
                            goes = True

                    previous_positions[id(result)] = current_position

                    # Default pose is "Stays"
                    label = "Stays"
                    color = (128, 128, 128)

                    # Update label based on detected pose
                    if hand_above_head:
                        label = "Hand Above Head"
                        color = (0, 255, 0)
                    elif lies:
                        label = "Lies"
                        color = (0, 0, 255)
                    elif goes:
                        label = "Goes"
                        color = (255, 0, 0)

                    # Draw keypoints
                    for kpx, kpy in kp:
                        cv2.circle(frame_out, (int(x + kpx), int(y + kpy)), 5, (0, 255, 0), -1)  # Draw keypoints as green circles

                    # Draw bounding box around the detected person
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_out, (x + x1, y + y1), (x + x2, y + y2), (0, 0, 255), 2)

                    # Display pose recognition result
                    cv2.putText(
                        frame_out,
                        label,
                        (x + x1, y + y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2
                    )

        # Draw bounding box around the moving object
        cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 200), 3)

    # Display the processed frame
    cv2.imshow('Human Pose Estimation', frame_out)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()