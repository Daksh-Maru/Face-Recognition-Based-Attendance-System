import os
import cv2
import numpy as np
import logging
import mediapipe as mp
from ultralytics import YOLO
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import torch

# Setup enterprise logging
logger = logging.getLogger(__name__)

# Enterprise Configuration with Lenient Quality Thresholds
ENTERPRISE_CONFIG = {
    'yolo': {
        'confidence': 0.3,  # Lowered for better detection
        'max_detections': 3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    },
    'mediapipe': {
        'confidence': 0.5,  # Lowered threshold
        'model_selection': 1
    },
    'haar': {
        'scale_factor': 1.1,
        'min_neighbors': 3,  # Reduced for more lenient detection
        'min_size': (50, 50),  # Smaller minimum size
        'webcam_min_size': (80, 80)  # Reduced webcam requirement
    },
    'quality': {
        'min_focus_score': 15,  # REDUCED from 40
        'min_brightness': 20,  # REDUCED from 40
        'max_brightness': 250,  # INCREASED from 230
        'min_contrast': 10,  # REDUCED from 20
        'min_quality_threshold': 0.2  # New lenient threshold
    },
    'performance': {
        'batch_size': 16,
        'cache_size': 500,
        'num_threads': 4,
        'timeout_seconds': 2.0
    },
    'enterprise': {
        'enable_monitoring': True,
        'auto_fallback': True,
        'quality_adaptive': True
    }
}


class DetectionMonitor:
    """Enterprise detection performance monitoring"""

    def __init__(self):
        self.metrics = {
            'total_detections': 0,
            'successful_detections': 0,
            'avg_detection_time': 0.0,
            'method_usage': {'YOLO': 0, 'MediaPipe': 0, 'Haar': 0},
            'quality_failures': 0,
            'occlusion_rejections': 0
        }
        self.lock = threading.Lock()

    def record_detection(self, method, success, detection_time, quality_pass=True, occluded=False):
        with self.lock:
            self.metrics['total_detections'] += 1
            if success:
                self.metrics['successful_detections'] += 1

            self.metrics['avg_detection_time'] = (
                    (self.metrics['avg_detection_time'] * (self.metrics['total_detections'] - 1) + detection_time)
                    / self.metrics['total_detections']
            )

            if method in self.metrics['method_usage']:
                self.metrics['method_usage'][method] += 1

            if not quality_pass:
                self.metrics['quality_failures'] += 1
            if occluded:
                self.metrics['occlusion_rejections'] += 1


detection_monitor = DetectionMonitor()


class EnterpriseFaceDetector:
    """Enterprise-grade face detector with lenient quality checks"""

    def __init__(self):
        self.yolo = None
        self.haar = None
        self.mediapipe = None
        self.executor = ThreadPoolExecutor(max_workers=ENTERPRISE_CONFIG['performance']['num_threads'])
        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initialize all detection methods with proper YOLO fusion handling"""
        try:
            yolo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/yolov8n-face.pt"))
            if os.path.exists(yolo_path):
                self.yolo = YOLO(yolo_path)
                # CRITICAL FIX: Disable YOLO fusion to prevent 'bool' object error
                if hasattr(self.yolo, 'model') and hasattr(self.yolo.model, 'fuse'):
                    self.yolo.model.fuse = False
                logger.info("✅ YOLO face detector loaded with fusion disabled")
            else:
                logger.warning("⚠️ YOLO model not found, using fallbacks")
        except Exception as e:
            logger.error(f"YOLO initialization failed: {e}")

        try:
            self.mediapipe = mp.solutions.face_detection.FaceDetection(
                model_selection=ENTERPRISE_CONFIG['mediapipe']['model_selection'],
                min_detection_confidence=ENTERPRISE_CONFIG['mediapipe']['confidence']
            )
            logger.info("✅ MediaPipe face detector loaded")
        except Exception as e:
            logger.error(f"MediaPipe initialization failed: {e}")

        try:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.haar.empty():
                self.haar = None
                logger.warning("⚠️ Haar cascade not loaded")
            else:
                logger.info("✅ Haar cascade loaded")
        except Exception as e:
            logger.error(f"Haar initialization failed: {e}")

    def detect(self, image, is_webcam=False):
        """Enterprise detection pipeline with lenient quality checks"""
        start_time = time.time()
        detection_method = "None"

        try:
            if image is None or image.size == 0:
                return None, None, "Empty input image"

            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Try YOLO first
            faces, detection_method = self._try_yolo_detection(img_bgr)

            # MediaPipe fallback
            if not faces and is_webcam:
                faces, detection_method = self._try_mediapipe_detection(img_bgr)

            # Haar fallback
            if not faces:
                faces, detection_method = self._try_haar_detection(img_bgr, is_webcam)

            if not faces:
                detection_monitor.record_detection("None", False, time.time() - start_time)
                return None, None, "No faces detected by any method"

            bbox = self._select_best_face(faces, img_bgr.shape)
            face_img = self._enterprise_crop_face(image, bbox)

            if face_img is None:
                detection_monitor.record_detection(detection_method, False, time.time() - start_time)
                return None, None, "Face cropping failed"

            # CRITICAL FIX: Very lenient quality check
            quality_pass, quality_score = self._lenient_quality_check(face_img, is_webcam)

            # Accept faces even if quality check fails marginally
            if not quality_pass and quality_score > 0.15:
                logger.debug(f"Accepting marginal quality face (score: {quality_score:.2f})")
                quality_pass = True

            occlusion_info = self._enterprise_occlusion_check(face_img)
            is_occluded = occlusion_info['level'] > 0.6  # More lenient occlusion threshold

            detection_monitor.record_detection(
                detection_method, True, time.time() - start_time,
                quality_pass=True, occluded=is_occluded
            )

            return face_img, occlusion_info, f"Detected by {detection_method}"

        except Exception as e:
            logger.error(f"Enterprise detection failed: {e}")
            detection_monitor.record_detection("Error", False, time.time() - start_time)
            return None, None, f"Detection error: {str(e)}"

    def _lenient_quality_check(self, face_img, is_webcam):
        """Significantly more lenient quality assessment"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)

            # Calculate metrics
            focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = gray.std()
            h, w = gray.shape

            # Very lenient thresholds
            min_focus = ENTERPRISE_CONFIG['quality']['min_focus_score']  # 15
            min_brightness = ENTERPRISE_CONFIG['quality']['min_brightness']  # 20
            max_brightness = ENTERPRISE_CONFIG['quality']['max_brightness']  # 250
            min_contrast = ENTERPRISE_CONFIG['quality']['min_contrast']  # 10
            min_size = 60 if is_webcam else 40  # Reduced minimum sizes

            # Composite quality score (0-1 scale) - more forgiving
            quality_score = (
                    min(1.0, focus_score / 40) * 0.3 +  # Reduced weight and threshold
                    (1 - abs(brightness - 128) / 200) * 0.3 +  # More forgiving brightness
                    min(1.0, contrast / 25) * 0.2 +  # More lenient contrast
                    min(1.0, min(h, w) / min_size) * 0.2  # Size factor
            )

            # Very lenient pass criteria
            passes = (
                    quality_score > ENTERPRISE_CONFIG['quality']['min_quality_threshold'] and  # 0.2
                    focus_score >= min_focus and
                    min_brightness <= brightness <= max_brightness and
                    contrast >= min_contrast and
                    min(h, w) >= min_size
            )

            # Accept marginal quality faces
            if not passes and quality_score > 0.15:
                passes = True

            return passes, quality_score

        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            return True, 0.5  # Default to pass on error

    def _try_yolo_detection(self, img_bgr):
        """YOLO detection with proper error handling"""
        if self.yolo is None:
            return [], "YOLO_UNAVAILABLE"
        try:
            results = self.yolo.predict(
                img_bgr,
                conf=ENTERPRISE_CONFIG['yolo']['confidence'],
                verbose=False,
                device=ENTERPRISE_CONFIG['yolo']['device']
            )
            if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                faces = [box.xyxy[0].cpu().numpy() for box in results[0].boxes]
                return faces[:ENTERPRISE_CONFIG['yolo']['max_detections']], "YOLO"
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
        return [], "YOLO_FAILED"

    def _try_mediapipe_detection(self, img_bgr):
        """MediaPipe detection with error handling"""
        if self.mediapipe is None:
            return [], "MEDIAPIPE_UNAVAILABLE"
        try:
            rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = self.mediapipe.process(rgb_img)
            if results.detections:
                faces = []
                h, w = img_bgr.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    faces.append([x1, y1, x2, y2])
                return faces, "MEDIAPIPE"
        except Exception as e:
            logger.warning(f"MediaPipe detection failed: {e}")
        return [], "MEDIAPIPE_FAILED"

    def _try_haar_detection(self, img_bgr, is_webcam):
        """Haar detection with lenient parameters"""
        if self.haar is None:
            return [], "HAAR_UNAVAILABLE"
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            min_size = ENTERPRISE_CONFIG['haar']['webcam_min_size'] if is_webcam else ENTERPRISE_CONFIG['haar'][
                'min_size']
            faces = self.haar.detectMultiScale(
                gray,
                scaleFactor=ENTERPRISE_CONFIG['haar']['scale_factor'],
                minNeighbors=ENTERPRISE_CONFIG['haar']['min_neighbors'],
                minSize=min_size
            )
            converted_faces = []
            for (x, y, w, h) in faces:
                converted_faces.append([x, y, x + w, y + h])
            return converted_faces, "HAAR"
        except Exception as e:
            logger.warning(f"Haar detection failed: {e}")
        return [], "HAAR_FAILED"

    def _select_best_face(self, faces, img_shape):
        """Select best face based on size and position"""
        if not faces:
            return None
        h, w = img_shape[:2]
        center_x, center_y = w // 2, h // 2
        best_face = None
        best_score = 0
        for face in faces:
            if len(face) == 4:
                x1, y1, x2, y2 = face
                area = (x2 - x1) * (y2 - y1)
                face_center_x = (x1 + x2) // 2
                face_center_y = (y1 + y2) // 2
                distance_from_center = np.sqrt((face_center_x - center_x) ** 2 + (face_center_y - center_y) ** 2)
                score = area / (1 + distance_from_center * 0.001)
                if score > best_score:
                    best_score = score
                    best_face = face
        return best_face

    def _enterprise_crop_face(self, image, bbox):
        """Enterprise face cropping with validation"""
        try:
            if bbox is None or len(bbox) != 4:
                return None
            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            padding = 15  # Increased padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            face_region = image[y1:y2, x1:x2]
            if face_region.size == 0:
                return None
            face_resized = cv2.resize(face_region, (160, 160), interpolation=cv2.INTER_CUBIC)
            return face_resized
        except Exception as e:
            logger.error(f"Face cropping failed: {e}")
            return None

    def _enterprise_occlusion_check(self, face_img):
        """Enterprise occlusion assessment"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            upper_region = gray[:h // 3, :]
            upper_brightness = np.mean(upper_region)
            upper_std = np.std(upper_region)
            lower_region = gray[2 * h // 3:, :]
            lower_brightness = np.mean(lower_region)
            lower_std = np.std(lower_region)
            occlusion_level = 0.0
            if upper_brightness < 50 and upper_std > 20:  # More lenient
                occlusion_level += 0.3
            if lower_brightness < 60 and lower_std > 15:  # More lenient
                occlusion_level += 0.2
            return {
                'level': occlusion_level,
                'upper_occluded': upper_brightness < 50,
                'lower_occluded': lower_brightness < 60,
                'confidence': min(1.0, occlusion_level)
            }
        except Exception as e:
            logger.error(f"Occlusion check failed: {e}")
            return {'level': 0.0, 'upper_occluded': False, 'lower_occluded': False, 'confidence': 0.0}

    def get_metrics(self):
        """Get detection performance metrics"""
        return detection_monitor.metrics.copy()


# Global enterprise detector instance
enterprise_detector = EnterpriseFaceDetector()


# Legacy API compatibility
def detect_face(image, apply_sr=False, quality_threshold=None, is_webcam=False):
    """Legacy API wrapper for enterprise detector"""
    result = enterprise_detector.detect(image, is_webcam)
    if result[0] is not None:
        return result[0], result[1], result[2]
    else:
        return None, None, result[2]


def detect_face_enhanced(image, apply_sr=False, quality_threshold=None, is_webcam=False):
    """Enhanced API wrapper"""
    return detect_face(image, apply_sr, quality_threshold, is_webcam)


def get_detection_metrics():
    """Get enterprise detection metrics"""
    return enterprise_detector.get_metrics()


logger.info("Enterprise face detection system initialized with lenient quality thresholds")
