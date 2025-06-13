# detection.py

from ultralytics import YOLO
import os
import cv2
import numpy as np

# Get absolute path to yolov8n-face.pt
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/yolov8n-face.pt"))
assert os.path.exists(model_path), f"Model file not found at {model_path}"

model = YOLO(model_path)

def detect_face(image):
    """
    Detects the most prominent face and returns a cropped face (RGB).
    Assumes input image is RGB format.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    # YOLOv8 expects RGB images, so no color conversion is needed
    results = model.predict(image, conf=0.25, verbose=False)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        # Find the largest face by area
        areas = [(float(b.xyxy[0][2] - b.xyxy[0][0]) * float(b.xyxy[0][3] - b.xyxy[0][1])) for b in boxes]
        biggest_idx = int(np.argmax(areas))
        biggest = boxes[biggest_idx]
        x1, y1, x2, y2 = map(int, biggest.xyxy[0])

        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            face_crop = image[y1:y2, x1:x2]
            return face_crop

    return None

