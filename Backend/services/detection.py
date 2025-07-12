from ultralytics import YOLO
import os
import cv2
import numpy as np
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get absolute path to yolov8n-face.pt
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/yolov8n-face.pt"))
assert os.path.exists(model_path), f"Model file not found at {model_path}"

# Initialize YOLOv8 face model
model = YOLO(model_path)

def detect_face(
    image: np.ndarray,
    min_confidence: float = 0.4,
    min_face_size: int = 40,
    padding_ratio: float = 0.10
) -> Optional[np.ndarray]:
    """
    Detects and crops the most confident face from an RGB image.
    Returns the face crop or None if detection fails.
    """
    if image is None or image.size == 0:
        logger.error("Empty or invalid image input.")
        return None

    # Single-pass detection with higher confidence threshold
    results = model.predict(image, conf=min_confidence, iou=0.50, imgsz=640, verbose=False)
    boxes = results[0].boxes
    if not boxes or len(boxes) == 0:
        logger.debug("No faces detected at conf ≥ %.2f", min_confidence)
        return None

    # Select box with highest confidence
    best_box = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Enforce minimum face size
    fw, fh = x2 - x1, y2 - y1
    if fw < min_face_size or fh < min_face_size:
        logger.debug("Discarded small face: %dx%d px.", fw, fh)
        return None

    # Apply adaptive padding
    px, py = int(fw * padding_ratio), int(fh * padding_ratio)
    x1p, y1p = max(0, x1 - px), max(0, y1 - py)
    x2p, y2p = min(w, x2 + px), min(h, y2 + py)
    crop = image[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        logger.warning("Zero-area crop after padding.")
        return None

    logger.info("Face detected (conf=%.2f)", float(best_box.conf[0]))
    return crop
