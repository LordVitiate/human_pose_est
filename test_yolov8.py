from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Make sure to download the model weights if you haven't already
print("Model loaded successfully!")
