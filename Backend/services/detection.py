# detection.py

from ultralytics import YOLO
import os
import cv2
import numpy as np
from super_resolution import SuperResolution
from preprocessing import handle_occlusions, preprocess_image_with_occlusion_handling

# Get absolute path to yolov8n-face.pt
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/yolov8n-face.pt"))

# Initialize models with error handling
yolo_model = None
sr_model = None


def initialize_models():
    """Initialize YOLO and super-resolution models with error handling"""
    global yolo_model, sr_model

    try:
        if os.path.exists(model_path):
            yolo_model = YOLO(model_path)
            print(f"✅ YOLO model loaded from {model_path}")
        else:
            print(f"⚠️ YOLO model not found at {model_path}")
            print("Face detection will be disabled.")

        # Initialize super-resolution model
        sr_model = SuperResolution(model_name="espcn", scale=2)
        print("✅ Super-resolution model initialized")

    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        yolo_model = None
        sr_model = None


# Initialize models on import
initialize_models()


def detect_face(image, apply_sr=True, quality_threshold=0.3):
    """
    Enhanced face detection with quality assessment and improved occlusion handling

    Args:
        image: Input RGB image
        apply_sr: Whether to apply super-resolution
        quality_threshold: Minimum quality score threshold

    Returns:
        tuple: (enhanced_face, occlusion_mask, occlusion_info) or (None, None, error_message)
    """
    try:
        if yolo_model is None:
            return None, None, "YOLO model not available"

        # Convert RGB to BGR for OpenCV/YOLO
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Apply super-resolution if requested and available
        if apply_sr and sr_model is not None:
            try:
                img_bgr = sr_model.upsample(img_bgr)
            except Exception as e:
                print(f"⚠️ Super-resolution failed: {e}")

        # Detect faces
        results = yolo_model.predict(img_bgr, conf=0.2, verbose=False)

        # Extract face coordinates
        faces = []
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            faces = results[0].boxes.xyxy.cpu().numpy()

        if len(faces) == 0:
            return None, None, "No faces detected"

        # Select the largest face
        largest_face = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        x1, y1, x2, y2 = map(int, largest_face)

        # Ensure coordinates are within image bounds
        h, w = img_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Check if face region is valid
        if x2 <= x1 or y2 <= y1:
            return None, None, "Invalid face coordinates"

        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        face_img = img_rgb[y1:y2, x1:x2]

        # Check if face image is valid
        if face_img.size == 0:
            return None, None, "Empty face region"

        # Quality assessment
        quality_score = assess_face_quality(face_img)

        if quality_score < quality_threshold:
            return None, None, f"Low quality: {quality_score:.2f}"

        # Enhanced occlusion handling
        try:
            enhanced_face, occlusion_mask, occlusion_info = handle_occlusions_enhanced(face_img)
        except Exception as e:
            print(f"⚠️ Enhanced occlusion handling failed: {e}")
            # Fallback to basic occlusion handling
            try:
                enhanced_face, occlusion_mask, occlusion_info = handle_occlusions(face_img)
            except Exception as e2:
                print(f"⚠️ Basic occlusion handling failed: {e2}")
                # Return basic processed image
                enhanced_face = face_img
                occlusion_mask = np.zeros((face_img.shape[0], face_img.shape[1]), dtype=np.uint8)
                occlusion_info = {
                    'upper_occluded': False,
                    'lower_occluded': False,
                    'upper_confidence': 0.0,
                    'lower_confidence': 0.0,
                    'total_occlusion_percentage': 0.0,
                    'occlusion_level': 'minimal'
                }

        return enhanced_face, occlusion_mask, occlusion_info

    except Exception as e:
        print(f"❌ Error in detect_face: {e}")
        return None, None, f"Detection error: {str(e)}"


# In detection.py - Update the handle_occlusions_enhanced function
def handle_occlusions_enhanced(face_img):
    """Enhanced occlusion handling - wrapper function"""
    try:
        # Use the heuristic approach from preprocessing
        return handle_occlusions(face_img)
    except Exception as e:
        print(f"Enhanced occlusion handling failed: {e}")
        # Return basic fallback
        enhanced_face = face_img
        mask = np.zeros((face_img.shape[0], face_img.shape[1]), dtype=np.float32)
        occlusion_info = {
            'upper_occluded': False,
            'lower_occluded': False,
            'upper_confidence': 0.0,
            'lower_confidence': 0.0,
            'total_occlusion_percentage': 0.0,
            'occlusion_level': 'minimal'
        }
        return enhanced_face, mask, occlusion_info



def detect_face_enhanced(image, apply_sr=True, quality_threshold=0.3):
    """
    Alias for detect_face with enhanced functionality
    This is the function expected by test scripts
    """
    return detect_face(image, apply_sr, quality_threshold)


def assess_face_quality(face_img):
    """Assess face image quality"""
    try:
        if face_img.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)

        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness / 100.0)

        # Brightness
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 128) / 128.0

        # Contrast
        contrast = gray.std()
        contrast_score = min(1.0, contrast / 50.0)

        # Overall quality
        quality_score = (sharpness_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)

        return max(0.0, min(1.0, quality_score))  # Ensure score is between 0 and 1

    except Exception as e:
        print(f"Quality assessment failed: {e}")
        return 0.5  # Return medium quality as fallback


def multi_scale_face_detection(image, scales=[0.5, 1.0, 1.5]):
    """Enhanced detection across multiple scales with NMS"""
    try:
        if yolo_model is None:
            return []

        all_detections = []

        for scale in scales:
            # Resize image
            height, width = image.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)

            if new_width <= 0 or new_height <= 0:
                continue

            scaled_img = cv2.resize(image, (new_width, new_height))

            # Convert to BGR for YOLO
            scaled_bgr = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2BGR)

            # Run detection on scaled image
            results = yolo_model.predict(scaled_bgr, conf=0.2, verbose=False)

            # Process results and scale back coordinates
            if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()

                for box, score in zip(boxes, scores):
                    # Scale coordinates back to original image size
                    x1, y1, x2, y2 = box
                    x1 = x1 / scale
                    y1 = y1 / scale
                    x2 = x2 / scale
                    y2 = y2 / scale

                    all_detections.append([x1, y1, x2, y2, score, 0])  # class_id = 0 for face

        # Apply NMS to remove overlapping detections
        if len(all_detections) > 0:
            # Simple NMS implementation
            filtered_detections = apply_simple_nms(all_detections, iou_threshold=0.5)
            return filtered_detections

        return []

    except Exception as e:
        print(f"Multi-scale detection failed: {e}")
        return []


def apply_simple_nms(detections, iou_threshold=0.5):
    """Simple Non-Maximum Suppression implementation"""
    try:
        if len(detections) == 0:
            return []

        detections = np.array(detections)

        # Sort by confidence score (descending)
        sorted_indices = np.argsort(detections[:, 4])[::-1]
        detections = detections[sorted_indices]

        keep = []

        while len(detections) > 0:
            # Take the detection with highest confidence
            current = detections[0]
            keep.append(current)

            if len(detections) == 1:
                break

            # Calculate IoU with remaining detections
            ious = []
            for i in range(1, len(detections)):
                iou = calculate_iou(current[:4], detections[i][:4])
                ious.append(iou)

            # Keep detections with IoU below threshold
            ious = np.array(ious)
            keep_indices = np.where(ious < iou_threshold)[0] + 1
            detections = detections[keep_indices]

        return keep

    except Exception as e:
        print(f"NMS failed: {e}")
        return detections if len(detections) > 0 else []


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    try:
        # Get intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate intersection area
        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    except Exception as e:
        print(f"IoU calculation failed: {e}")
        return 0.0


def test_detection():
    """Test the detection pipeline"""
    try:
        print("🧪 Testing face detection pipeline...")

        # Create a synthetic test image
        test_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)

        # Test basic detection
        face, mask, info = detect_face(test_img)

        if face is not None:
            print("✅ Face detection test passed!")
            print(f"Face shape: {face.shape}")
            print(f"Mask shape: {mask.shape}")
            print(f"Info: {info}")
        else:
            print("⚠️ No face detected in test image (expected for random image)")

        # Test quality assessment
        quality = assess_face_quality(test_img)
        print(f"✅ Quality assessment: {quality:.3f}")

        print("🎉 Detection pipeline test completed!")

    except Exception as e:
        print(f"❌ Detection test failed: {e}")


if __name__ == "__main__":
    test_detection()
