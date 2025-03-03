import cv2
from ultralytics import YOLO
import numpy as np

def get_shoulder_type(angle):

    if not (0 <= angle <= 180):
        raise ValueError("Angle must be within the range [0, 180].")

    if 0 <= angle < 45:
        return 'pressed'
    elif 45 <= angle < 90:
        return 'semi-open'
    elif 90 <= angle < 135:
        return 'semi-pressed'
    elif 135 <= angle <= 180:
        return 'fully_open'

def get_elbow_type(angle):

    if not (0 <= angle <= 180):
        raise ValueError("Angle must be within the range [0, 180].")

    if 135 <= angle <= 180:
        return 'pressed'
    elif 90 <= angle < 135:
        return 'semi-pressed'
    elif 45 <= angle < 90:
        return 'semi-open'
    elif 0 <= angle < 45:
        return 'open'        


class PoseAnalysis:
    def __init__(self, keypoints):
        self.keypoints = keypoints
        self.validate_keypoints()

    def validate_keypoints(self):
        """
        Validates the presence of keypoints. If some keypoints are missing,
        log a warning instead of raising an error.
        """
        required_keys = [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]
        missing_keys = [key for key in required_keys if key not in self.keypoints]
        if missing_keys:
            print(f"Warning: Missing keypoints: {missing_keys}. Some calculations may be skipped.")

    def calculate_b_axis(self):
        """
        Calculate the B-axis (shoulder axis) and its center.
        Skips calculation if required keypoints are missing.
        """
        left_shoulder = self.keypoints.get('left_shoulder')
        right_shoulder = self.keypoints.get('right_shoulder')

        if left_shoulder is None or right_shoulder is None:
            print("Skipping B-axis calculation due to missing keypoints.")
            return None, None

        left_shoulder = np.array(left_shoulder)
        right_shoulder = np.array(right_shoulder)
        center_of_shoulders = (left_shoulder + right_shoulder) / 2
        b_axis = right_shoulder - left_shoulder
        return center_of_shoulders, b_axis

    def calculate_a_axis(self):
        """
        Calculate the A-axis (hip axis) and its center.
        Skips calculation if required keypoints are missing.
        """
        left_hip = self.keypoints.get('left_hip')
        right_hip = self.keypoints.get('right_hip')

        if left_hip is None or right_hip is None:
            print("Skipping A-axis calculation due to missing keypoints.")
            return None, None

        left_hip = np.array(left_hip)
        right_hip = np.array(right_hip)
        center_of_hips = (left_hip + right_hip) / 2
        a_axis = right_hip - left_hip
        return center_of_hips, a_axis

    def calculate_reference_y_axis(self):
        """
        Calculate the reference Y-axis (vertical axis between shoulder and hip centers).
        Skips calculation if required keypoints are missing.
        """
        center_of_shoulders, _ = self.calculate_b_axis()
        center_of_hips, _ = self.calculate_a_axis()

        if center_of_shoulders is None or center_of_hips is None:
            print("Skipping reference Y-axis calculation due to missing keypoints.")
            return None

        reference_y_axis = center_of_hips - center_of_shoulders
        return reference_y_axis

    def calculate_arm_segments(self):
        """
        Calculate arm segments (shoulder - elbow) for left and right.
        Skips calculation if required keypoints are missing.
        """
        left_shoulder = self.keypoints.get('left_shoulder')
        right_shoulder = self.keypoints.get('right_shoulder')
        left_elbow = self.keypoints.get('left_elbow')
        right_elbow = self.keypoints.get('right_elbow')

        arm_segments = {}

        if left_shoulder is not None and left_elbow is not None:
            left_shoulder = np.array(left_shoulder)
            left_elbow = np.array(left_elbow)
            hl_vector = left_elbow - left_shoulder
            arm_segments['HL'] = {'start': left_shoulder, 'vector': hl_vector}

        if right_shoulder is not None and right_elbow is not None:
            right_shoulder = np.array(right_shoulder)
            right_elbow = np.array(right_elbow)
            hr_vector = right_elbow - right_shoulder
            arm_segments['HR'] = {'start': right_shoulder, 'vector': hr_vector}

        if not arm_segments:
            print("Skipping arm segment calculation due to missing keypoints.")
        return arm_segments

    def calculate_forearm_segments(self):
        """
        Calculate forearm segments (elbow - wrist) for left and right.
        Skips calculation if required keypoints are missing.
        """
        left_elbow = self.keypoints.get('left_elbow')
        right_elbow = self.keypoints.get('right_elbow')
        left_wrist = self.keypoints.get('left_wrist')
        right_wrist = self.keypoints.get('right_wrist')

        forearm_segments = {}

        if left_elbow is not None and left_wrist is not None:
            left_elbow = np.array(left_elbow)
            left_wrist = np.array(left_wrist)
            fal_vector = left_wrist - left_elbow
            forearm_segments['FAL'] = {'start': left_elbow, 'vector': fal_vector}

        if right_elbow is not None and right_wrist is not None:
            right_elbow = np.array(right_elbow)
            right_wrist = np.array(right_wrist)
            far_vector = right_wrist - right_elbow
            forearm_segments['FAR'] = {'start': right_elbow, 'vector': far_vector}

        if not forearm_segments:
            print("Skipping forearm segment calculation due to missing keypoints.")
        return forearm_segments

    def calculate_leg_segments(self):
        """
        Calculate leg segments (hip - knee) for left and right.
        Skips calculation if required keypoints are missing.
        """
        left_hip = self.keypoints.get('left_hip')
        right_hip = self.keypoints.get('right_hip')
        left_knee = self.keypoints.get('left_knee')
        right_knee = self.keypoints.get('right_knee')

        leg_segments = {}

        if left_hip is not None and left_knee is not None:
            left_hip = np.array(left_hip)
            left_knee = np.array(left_knee)
            ll_vector = left_knee - left_hip
            leg_segments['LL'] = {'start': left_hip, 'vector': ll_vector}

        if right_hip is not None and right_knee is not None:
            right_hip = np.array(right_hip)
            right_knee = np.array(right_knee)
            lr_vector = right_knee - right_hip
            leg_segments['LR'] = {'start': right_hip, 'vector': lr_vector}

        if not leg_segments:
            print("Skipping leg segment calculation due to missing keypoints.")
        return leg_segments

    def calculate_shin_segments(self):
        """
        Calculate shin segments (knee - ankle) for left and right.
        Skips calculation if required keypoints are missing.
        """
        left_knee = self.keypoints.get('left_knee')
        right_knee = self.keypoints.get('right_knee')
        left_ankle = self.keypoints.get('left_ankle')
        right_ankle = self.keypoints.get('right_ankle')

        shin_segments = {}

        if left_knee is not None and left_ankle is not None:
            left_knee = np.array(left_knee)
            left_ankle = np.array(left_ankle)
            sl_vector = left_ankle - left_knee
            shin_segments['SL'] = {'start': left_knee, 'vector': sl_vector}

        if right_knee is not None and right_ankle is not None:
            right_knee = np.array(right_knee)
            right_ankle = np.array(right_ankle)
            sr_vector = right_ankle - right_knee
            shin_segments['SR'] = {'start': right_knee, 'vector': sr_vector}

        if not shin_segments:
            print("Skipping shin segment calculation due to missing keypoints.")
        return shin_segments

    def calculate_angle_between_vectors(self, v1, v2):
        """
        Calculate the angle between two vectors in degrees, ensuring the result is in the range [0, 180].
        """
        if v1 is None or v2 is None:
            return None

        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Handle floating-point precision issues
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def calculate_angles(self):
        """
        Calculate all specified angles:
        - Angle between HL/HR and y-axis
        - Angle between FAL/FAR and HL/HR
        - Angle between LL/LR and y-axis
        - Angle between SL/SR and LL/LR
        """
        angles = {}

        # Reference y-axis
        reference_y_axis = self.calculate_reference_y_axis()

        # Arm segments (HL, HR)
        arm_segments = self.calculate_arm_segments()
        for side in ['HL', 'HR']:
            vector = arm_segments.get(side, {}).get('vector')
            angle = self.calculate_angle_between_vectors(reference_y_axis, vector)
            if angle is not None:
                angles[f"{side}_to_y"] = angle

        # Forearm segments (FAL, FAR)
        forearm_segments = self.calculate_forearm_segments()
        side_mapping = {'FAL': 'HL', 'FAR': 'HR'}
        for side in ['FAL', 'FAR']:
            forearm_vector = forearm_segments.get(side, {}).get('vector')
            arm_side = side_mapping.get(side)
            arm_vector = arm_segments.get(arm_side, {}).get('vector')
            angle = self.calculate_angle_between_vectors(arm_vector, forearm_vector)
            if angle is not None:
                angles[f"{side}_to_{side[:-1]}"] = angle

        # Leg segments (LL, LR)
        leg_segments = self.calculate_leg_segments()
        for side in ['LL', 'LR']:
            vector = leg_segments.get(side, {}).get('vector')
            angle = self.calculate_angle_between_vectors(reference_y_axis, vector)
            if angle is not None:
                angles[f"{side}_to_y"] = angle

        # Shin segments (SL, SR)
        side_mapping = {'SL': 'LL', 'SR': 'LR'}
        shin_segments = self.calculate_shin_segments()
        for side in ['SL', 'SR']:
            shin_vector = shin_segments.get(side, {}).get('vector')
            leg_side = side_mapping.get(side)
            leg_vector = leg_segments.get(leg_side, {}).get('vector')  # LL for SL, LR for SR
            angle = self.calculate_angle_between_vectors(leg_vector, shin_vector)
            if angle is not None:
                angles[f"{side}_to_{side[:-1]}"] = angle
        return angles

    def get_segments(self):
        """
        Get all calculated segments: B axis, A axis, reference y-axis, arms, forearms, legs, and shins.
        Returns partial results if some segments cannot be calculated.
        """
        center_of_shoulders, b_axis = self.calculate_b_axis()
        center_of_hips, a_axis = self.calculate_a_axis()
        reference_y_axis = self.calculate_reference_y_axis()
        arm_segments = self.calculate_arm_segments()
        forearm_segments = self.calculate_forearm_segments()
        leg_segments = self.calculate_leg_segments()
        shin_segments = self.calculate_shin_segments()

        return {
            'B_axis': {'center': center_of_shoulders, 'vector': b_axis} if center_of_shoulders is not None else None,
            'A_axis': {'center': center_of_hips, 'vector': a_axis} if center_of_hips is not None else None,
            'reference_y_axis': reference_y_axis,
            'arms': arm_segments if arm_segments else None,
            'forearms': forearm_segments if forearm_segments else None,
            'legs': leg_segments if leg_segments else None,
            'shins': shin_segments if shin_segments else None
        }
# Define the video stream URL
video_stream_url = 'rtsp://admin:!Qwerty1234@192.168.1.233:554/Streaming/Channels/101'
# video_stream_url = '3.mp4'

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

def is_valid_keypoint(keypoint):
    """Check if a keypoint is valid (not at (0, 0))."""
    return keypoint[0] != 0 or keypoint[1] != 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video stream.")
        break

    # Perform pose estimation on the entire frame
    results = pose_model(frame, conf=0.5)  # Adjust confidence threshold as needed

    # Create a copy of the frame for drawing results
    frame_out = frame.copy()

    for result in results:
        keypoints = result.keypoints.xy[0]  # Get keypoints (x, y coordinates) for the first person detected
        boxes = result.boxes  # Get bounding boxes

        if keypoints.shape[0] < len(KEYPOINT_INDEX):
            continue  # Skip if not all keypoints are detected

        # Extract keypoints into a dictionary
        try:
            keypoints_dict = {
                'left_shoulder': keypoints[KEYPOINT_INDEX['left_shoulder']].tolist(),
                'right_shoulder': keypoints[KEYPOINT_INDEX['right_shoulder']].tolist(),
                'left_elbow': keypoints[KEYPOINT_INDEX['left_elbow']].tolist(),
                'right_elbow': keypoints[KEYPOINT_INDEX['right_elbow']].tolist(),
                'left_wrist': keypoints[KEYPOINT_INDEX['left_wrist']].tolist(),
                'right_wrist': keypoints[KEYPOINT_INDEX['right_wrist']].tolist(),
                'left_hip': keypoints[KEYPOINT_INDEX['left_hip']].tolist(),
                'right_hip': keypoints[KEYPOINT_INDEX['right_hip']].tolist(),
                'left_knee': keypoints[KEYPOINT_INDEX['left_knee']].tolist(),
                'right_knee': keypoints[KEYPOINT_INDEX['right_knee']].tolist(),
                'left_ankle': keypoints[KEYPOINT_INDEX['left_ankle']].tolist(),
                'right_ankle': keypoints[KEYPOINT_INDEX['right_ankle']].tolist()
            }
        except IndexError:
            continue  # Skip if keypoints are not fully detected

        # Validate keypoints
        if not all(is_valid_keypoint(keypoints_dict[key]) for key in keypoints_dict):
            continue  # Skip if any keypoint is invalid

        # Perform analysis
        pose_analysis = PoseAnalysis(keypoints_dict)
        segments = pose_analysis.get_segments()
        angles = pose_analysis.calculate_angles()  # Calculate angles

        # Draw B-axis (X-axis) if available
        if segments['B_axis'] is not None:
            b_center = segments['B_axis']['center']
            b_vector = segments['B_axis']['vector']
            if is_valid_keypoint(b_center):
                b_start = tuple(map(int, b_center - b_vector / 2))
                b_end = tuple(map(int, b_center + b_vector / 2))
                cv2.line(frame_out, b_start, b_end, (255, 0, 0), 2)  # Blue line for B-axis

        # Draw reference Y-axis if available
        if segments['reference_y_axis'] is not None and segments['B_axis'] is not None:
            y_center = segments['B_axis']['center']
            if is_valid_keypoint(y_center):
                y_vector = segments['reference_y_axis']
                y_end = tuple(map(int, y_center + y_vector))
                cv2.line(frame_out, tuple(map(int, y_center)), y_end, (0, 0, 255), 2)  # Red line for Y-axis

        # Draw arm segments (HL and HR) if available
        if segments['arms'] is not None:
            for arm_name, arm_data in segments['arms'].items():
                start = arm_data['start']
                vector = arm_data['vector']
                if is_valid_keypoint(start) and is_valid_keypoint(start + vector):
                    end = tuple(map(int, start + vector))
                    start = tuple(map(int, start))
                    color = (0, 255, 255) if arm_name == 'HL' else (255, 255, 0)  # Yellow for HL, Cyan for HR
                    cv2.line(frame_out, start, end, color, 2)

                    # Define a dictionary to map angle types to colors
                    shoulder_type_colors = {
                        "fully_open": (0, 0, 255),   # Red for fully pressed
                        "semi-open": (0, 255, 255), # Yellow for partially open
                        "semi-pressed": (255, 165, 0), # Orange for semi-pressed
                        "pressed": (0, 255, 0)       # Green for fully open
                    }

                    # Display angle between HL/HR and y-axis
                    angle_key = f"{arm_name}_to_y"
                    if angle_key in angles:
                        angle_text = get_shoulder_type(angles[angle_key])  # Get the angle type as a string

                        # Draw the text on the frame with the corresponding color
                        cv2.putText(
                            frame_out,
                            angle_text,
                            (start[0], start[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            shoulder_type_colors.get(angle_text, (255, 255, 255)),  # Use the mapped color
                            1
                        )

        # Draw forearm segments (FAL and FAR) if available
        if segments['forearms'] is not None:
            for forearm_name, forearm_data in segments['forearms'].items():
                start = forearm_data['start']
                vector = forearm_data['vector']
                if is_valid_keypoint(start) and is_valid_keypoint(start + vector):
                    end = tuple(map(int, start + vector))
                    start = tuple(map(int, start))
                    color = (0, 255, 0) if forearm_name == 'FAL' else (0, 128, 0)  # Green for FAL, Dark Green for FAR
                    cv2.line(frame_out, start, end, color, 2)

                    # Display angle between FAL/FAR and HL/HR
                    # angle_key = f"{forearm_name}_to_{forearm_name[:-1]}"
                    # if angle_key in angles:
                        # angle_text = f"{angle_key}: {angles[angle_key]:.1f}°"
                        # cv2.putText(frame_out, angle_text, (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    elbow_type_colors = {
                        "pressed": (0, 0, 255),   # Red for fully pressed
                        "semi-pressed": (0, 255, 255), # Yellow for partially open
                        "semi-open": (255, 165, 0), # Orange for semi-pressed
                        "open": (0, 255, 0)       # Green for fully open
                    }

                    # Display angle between HL/HR and y-axis
                    # angle_key = f"{arm_name}_to_y"
                    angle_key = f"{forearm_name}_to_{forearm_name[:-1]}"
                    # print(angle_key)
                    if angle_key in angles:
                        angle_text = get_elbow_type(angles[angle_key])  # Get the angle type as a string
                        # angle_text = f"{angles[angle_key]}"  # Get the angle type as a string
                        # Draw the text on the frame with the corresponding color
                        cv2.putText(
                            frame_out,
                            angle_text,
                            (start[0], start[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            elbow_type_colors.get(angle_text, (255, 255, 255)),  # Use the mapped color
                            1
                        )                        

        # Draw leg segments (LL and LR) if available
        if segments['legs'] is not None:
            for leg_name, leg_data in segments['legs'].items():
                start = leg_data['start']
                vector = leg_data['vector']
                if is_valid_keypoint(start) and is_valid_keypoint(start + vector):
                    end = tuple(map(int, start + vector))
                    start = tuple(map(int, start))
                    color = (255, 0, 255) if leg_name == 'LL' else (255, 255, 0)  # Magenta for LL, Cyan for LR
                    cv2.line(frame_out, start, end, color, 2)

                    # Display angle between LL/LR and y-axis
                    angle_key = f"{leg_name}_to_y"
                    if angle_key in angles:
                        angle_text = f"{angle_key}: {angles[angle_key]:.1f}°"
                        # cv2.putText(frame_out, angle_text, (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw shin segments (SL and SR) if available
        if segments['shins'] is not None:
            for shin_name, shin_data in segments['shins'].items():
                start = shin_data['start']
                vector = shin_data['vector']
                if is_valid_keypoint(start) and is_valid_keypoint(start + vector):
                    end = tuple(map(int, start + vector))
                    start = tuple(map(int, start))
                    color = (255, 165, 0) if shin_name == 'SL' else (255, 200, 0)  # Orange for SL, Light Orange for SR
                    cv2.line(frame_out, start, end, color, 2)

                    # Display angle between SL/SR and LL/LR
                    angle_key = f"{shin_name}_to_{shin_name[:-1]}"
                    if angle_key in angles:
                        angle_text = f"{angle_key}: {angles[angle_key]:.1f}°"
                        # cv2.putText(frame_out, angle_text, (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Optionally, draw bounding boxes around detected persons
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw bounding box in red

        # Draw keypoints
        for i, (kpx, kpy) in enumerate(keypoints):
            if kpx != 0 or kpy != 0:  # Skip invalid keypoints
                color = (0, 255, 0)  # Default color is green
                cv2.circle(frame_out, (int(kpx), int(kpy)), 1, color, -1)

    # Display the processed frame
    cv2.imshow('Human Pose Estimation', frame_out)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()