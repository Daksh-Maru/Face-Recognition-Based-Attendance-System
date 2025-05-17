# detection.py

from ultralytics import YOLO
import os
import cv2
import numpy as np
from .super_resolution import SuperResolution

# Get absolute path to yolov8n-face.pt
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/yolov8n-face.pt"))
assert os.path.exists(model_path), f"Model file not found at {model_path}"

yolo_model = YOLO(model_path)
sr_model = SuperResolution(model_name="espcn", scale=2)


def detect_face(image, apply_sr=True):
    """
    Detect the largest face in an image using YOLOv8

    Args:
        image: Input RGB image
        apply_sr: Whether to apply super-resolution

    Returns:
        Cropped face image or None if no face detected
    """
    # Convert to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Apply super-resolution if requested
    if apply_sr:
        img_bgr = sr_model.upsample(img_bgr)

    # Detect faces
    results = yolo_model.predict(img_bgr, conf=0.2, verbose=False)

    # Extract face bounding boxes
    faces = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    if len(faces) == 0:
        return None

    # Select the largest face
    largest_face = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    x1, y1, x2, y2 = map(int, largest_face)

    # Convert back to RGB for further processing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Extract face region
    face_img = img_rgb[y1:y2, x1:x2]

    return face_img