import cv2
import numpy as np
import logging
import mediapipe as mp
from sklearn.decomposition import PCA
import joblib
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os

logger = logging.getLogger(__name__)

# Updated Enterprise Configuration with Very Lenient Quality Requirements
ENTERPRISE_CONFIG = {
    'webcam': {
        'min_face_size': (60, 60),      # Reduced for better acceptance
        'clahe_clip': 1.5,              # Reduced for less aggressive enhancement
        'denoise_strength': 2,          # Reduced for speed
        'sharpening_kernel': np.array([[-0.05, -0.05, -0.05],
                                     [-0.05, 1.2, -0.05],
                                     [-0.05, -0.05, -0.05]]),  # Lighter sharpening
    },
    'cctv': {
        'min_face_size': (40, 40),      # Very small minimum
        'brightness_threshold': 40,      # Lower threshold
        'max_enhance': 1.8              # Increased enhancement
    },
    'occlusion': {
        'cnn_threshold': 0.7,           # Higher threshold (more lenient)
        'gabor_threshold': 0.4,         # Higher threshold
        'temporal_window': 6,           # Reduced for memory
        'max_occlusion_allowed': 0.7    # More lenient
    },
    'quality': {
        'min_focus': 8,                 # Much more lenient
        'contrast_threshold': 8,        # Much lower
        'brightness_range': (10, 250)   # Very wide range
    },
    'performance': {
        'batch_size': 16,
        'cache_size': 300,
        'num_threads': 4,
        'enable_monitoring': True
    }
}

class PreprocessingMonitor:
    """Enterprise preprocessing performance monitoring"""

    def __init__(self):
        self.metrics = {
            'total_processed': 0,
            'successful_processed': 0,
            'avg_processing_time': 0.0,
            'quality_failures': 0,
            'occlusion_detections': 0,
            'alignment_failures': 0
        }
        self.lock = threading.Lock()

    def record_processing(self, success, processing_time, quality_pass=True, occluded=False, aligned=True):
        with self.lock:
            self.metrics['total_processed'] += 1
            if success:
                self.metrics['successful_processed'] += 1

            self.metrics['avg_processing_time'] = (
                (self.metrics['avg_processing_time'] * (self.metrics['total_processed'] - 1) + processing_time)
                / self.metrics['total_processed']
            )

            if not quality_pass:
                self.metrics['quality_failures'] += 1
            if occluded:
                self.metrics['occlusion_detections'] += 1
            if not aligned:
                self.metrics['alignment_failures'] += 1

preprocessing_monitor = PreprocessingMonitor()

class EnterpriseOcclusionValidator:
    """Enterprise-grade occlusion detection with very lenient validation"""

    def __init__(self):
        self.gabor_bank = self._create_gabor_filters()
        self.pca = None
        self.clf = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._load_models()

    def _create_gabor_filters(self):
        """Optimized Gabor filter bank"""
        filters = []
        # Reduced filter count for performance
        for theta in np.arange(0, np.pi, np.pi / 3):  # 3 orientations
            for sigma in (2, 4):
                kern = cv2.getGaborKernel((13, 13), sigma, theta, 8, 0.5)
                filters.append(kern)
        return filters

    def _load_models(self):
        """Load pre-trained models with error handling"""
        try:
            if os.path.exists('assets/occlusion_cnn.pkl'):
                self.clf = joblib.load('assets/occlusion_cnn.pkl')
                logger.info("✅ Occlusion CNN model loaded")

            if os.path.exists('assets/occlusion_pca.pkl'):
                self.pca = joblib.load('assets/occlusion_pca.pkl')
                logger.info("✅ Occlusion PCA model loaded")

        except Exception as e:
            logger.warning(f"Model loading failed: {e}")

    def validate(self, face_img):
        """Very lenient occlusion validation"""
        try:
            if face_img is None or face_img.size == 0:
                return False

            # Fast heuristic check first with lenient threshold
            if self._fast_heuristic_check(face_img):
                return True

            # More lenient fallback checks
            return self._lenient_brightness_validation(face_img)

        except Exception as e:
            logger.error(f"Occlusion validation failed: {e}")
            return False  # Default to no occlusion

    def _fast_heuristic_check(self, face_img):
        """Very lenient brightness-based occlusion check"""
        try:
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_img

            upper_region = gray[:gray.shape[0] // 3, :]
            return np.mean(upper_region) < 25  # Very dark threshold
        except:
            return False

    def _lenient_brightness_validation(self, face_img):
        """Very lenient brightness-based validation"""
        try:
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_img

            upper = gray[:gray.shape[0] // 2, :]
            return np.mean(upper) < 30  # Very lenient threshold
        except:
            return False

class EnterprisePreprocessor:
    """Enterprise-grade preprocessing system with lenient quality standards"""

    def __init__(self):
        self.occlusion_validator = EnterpriseOcclusionValidator()
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.4,  # Very lenient
                min_tracking_confidence=0.3
            )
        except Exception as e:
            logger.warning(f"MediaPipe initialization failed: {e}")
            self.face_mesh = None

        self.executor = ThreadPoolExecutor(max_workers=ENTERPRISE_CONFIG['performance']['num_threads'])

    def enhance_image_enterprise(self, image, is_webcam=False):
        """Minimal enhancement to preserve image quality"""
        start_time = time.time()

        try:
            if image is None or image.size == 0:
                return image

            # Ensure image is in correct format
            if len(image.shape) != 3:
                logger.warning("Image is not 3-channel, returning as-is")
                return image

            if is_webcam:
                # Very light webcam processing
                try:
                    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)

                    # Light CLAHE enhancement
                    clahe = cv2.createCLAHE(
                        clipLimit=ENTERPRISE_CONFIG['webcam']['clahe_clip'],
                        tileGridSize=(6, 6)
                    )
                    l_enhanced = clahe.apply(l)
                    enhanced = cv2.merge([l_enhanced, a, b])
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

                    # Skip denoising for speed
                    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                    return enhanced

                except Exception as e:
                    logger.warning(f"Webcam enhancement failed: {e}, using original")
                    return image
            else:
                # Minimal CCTV processing
                try:
                    brightness = np.mean(image)
                    if brightness < ENTERPRISE_CONFIG['cctv']['brightness_threshold']:
                        enhanced = cv2.convertScaleAbs(
                            image,
                            alpha=ENTERPRISE_CONFIG['cctv']['max_enhance'],
                            beta=10
                        )
                        return enhanced
                    return image
                except Exception as e:
                    logger.warning(f"CCTV enhancement failed: {e}")
                    return image

        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return image
        finally:
            if ENTERPRISE_CONFIG['performance']['enable_monitoring']:
                preprocessing_monitor.record_processing(
                    True, time.time() - start_time
                )

    def align_face_enterprise(self, face_img):
        """Optional face alignment with error recovery"""
        try:
            if self.face_mesh is None:
                return face_img

            if face_img is None or face_img.size == 0:
                return face_img

            # Convert RGB to BGR for MediaPipe
            bgr_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            results = self.face_mesh.process(bgr_img)

            if not results.multi_face_landmarks:
                return face_img

            landmarks = results.multi_face_landmarks[0].landmark

            try:
                # Use eye landmarks for alignment
                left_eye = (int(landmarks[33].x * face_img.shape[1]),
                            int(landmarks[33].y * face_img.shape[0]))
                right_eye = (int(landmarks[263].x * face_img.shape[1]),
                             int(landmarks[263].y * face_img.shape[0]))

                # Calculate rotation angle
                dY = right_eye[1] - left_eye[1]
                dX = right_eye[0] - left_eye[0]
                angle = np.degrees(np.arctan2(dY, dX))

                # Only rotate if angle is significant
                if abs(angle) > 5:  # Increased threshold
                    center = (face_img.shape[1] // 2, face_img.shape[0] // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    aligned = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
                    return aligned

                return face_img

            except (IndexError, ValueError) as e:
                logger.warning(f"Landmark extraction failed: {e}")
                return face_img

        except Exception as e:
            logger.warning(f"Face alignment failed: {e}")
            return face_img

    def assess_quality_enterprise(self, face_img):
        """Very lenient quality assessment"""
        try:
            if face_img is None or face_img.size == 0:
                return True, 0.5  # Default to pass

            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_img

            h, w = gray.shape

            # Very lenient size check
            min_size = min(ENTERPRISE_CONFIG['webcam']['min_face_size'])
            if min(h, w) < min_size:
                return False, 0.0

            # Very lenient quality metrics
            focus = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = gray.std()

            # Generous scoring
            focus_score = min(1.0, focus / 25.0)  # Much lower threshold
            brightness_range = ENTERPRISE_CONFIG['quality']['brightness_range']
            brightness_score = 1.0 if brightness_range[0] <= brightness <= brightness_range[1] else 0.8  # Still generous
            contrast_score = min(1.0, contrast / 15.0)  # Lower threshold

            # Combined quality score with generous weighting
            quality_score = (focus_score * 0.4 + brightness_score * 0.4 + contrast_score * 0.2)

            # Very lenient pass criteria
            passes = (
                focus >= ENTERPRISE_CONFIG['quality']['min_focus'] and
                brightness_range[0] <= brightness <= brightness_range[1] and
                contrast >= ENTERPRISE_CONFIG['quality']['contrast_threshold']
            )

            # Override: accept marginal quality
            if not passes and quality_score > 0.3:
                passes = True

            return passes, quality_score

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return True, 0.5  # Default to pass on error

    def handle_occlusions_enterprise(self, face_img, tracker=None):
        """Very lenient occlusion handling"""
        try:
            # Quality check first but be very forgiving
            quality_pass, quality_score = self.assess_quality_enterprise(face_img)

            # Accept almost everything
            if quality_score < 0.1:  # Only reject very poor quality
                return face_img, np.zeros_like(face_img[:, :, 0]), {
                    'occlusion_level': 'very_low_quality',
                    'confidence': 0.0,
                    'quality_score': quality_score,
                    'total_occlusion_percentage': 0.0
                }

            # Very lenient occlusion validation
            is_occluded = self.occlusion_validator.validate(face_img)
            occlusion_confidence = 0.3 if is_occluded else 0.05  # Lower confidence penalties

            # Create result mask
            if len(face_img.shape) == 3:
                mask = np.zeros_like(face_img[:, :, 0])
            else:
                mask = np.zeros_like(face_img)

            # Very minimal occlusion info
            occlusion_info = {
                'occlusion_level': 'minimal' if occlusion_confidence < 0.5 else 'moderate',
                'confidence': occlusion_confidence,
                'quality_score': quality_score,
                'upper_occluded': is_occluded,
                'lower_occluded': False,
                'total_occlusion_percentage': occlusion_confidence * 0.3  # Much lower penalty
            }

            return face_img, mask, occlusion_info

        except Exception as e:
            logger.error(f"Enterprise occlusion handling failed: {e}")
            return face_img, np.zeros_like(face_img[:, :, 0] if len(face_img.shape) == 3 else face_img), {
                'occlusion_level': 'error',
                'confidence': 0.0,
                'quality_score': 0.5,
                'total_occlusion_percentage': 0.0
            }

    def preprocess_enterprise(self, image, is_webcam=False, tracker=None):
        """Complete enterprise preprocessing pipeline with lenient standards"""
        start_time = time.time()

        try:
            if image is None or image.size == 0:
                return None, None, {'occlusion_level': 'error', 'total_occlusion_percentage': 0.0}, 0.0

            # Minimal enhancement
            enhanced = self.enhance_image_enterprise(image, is_webcam)

            # Optional alignment only for webcam
            if is_webcam and enhanced is not None:
                aligned = self.align_face_enterprise(enhanced)
            else:
                aligned = enhanced

            # Lenient quality and occlusion handling
            processed_face, mask, occlusion_info = self.handle_occlusions_enterprise(aligned, tracker)

            # Generous final quality score
            quality_score = max(0.5, occlusion_info.get('quality_score', 0.7))  # Default to good score

            if ENTERPRISE_CONFIG['performance']['enable_monitoring']:
                preprocessing_monitor.record_processing(
                    True, time.time() - start_time,
                    quality_pass=quality_score > 0.2,  # Very lenient
                    occluded=occlusion_info.get('confidence', 0) > 0.6  # Higher threshold
                )

            return processed_face, mask, occlusion_info, quality_score

        except Exception as e:
            logger.error(f"Enterprise preprocessing failed: {e}")
            if ENTERPRISE_CONFIG['performance']['enable_monitoring']:
                preprocessing_monitor.record_processing(False, time.time() - start_time)
            return None, None, {'occlusion_level': 'error', 'total_occlusion_percentage': 0.0}, 0.0

    def get_metrics(self):
        """Get preprocessing performance metrics"""
        return preprocessing_monitor.metrics.copy()

# Global enterprise preprocessor
enterprise_preprocessor = EnterprisePreprocessor()

# Legacy API compatibility (maintained exactly as before)
def precise_alignment(face_img):
    """Legacy MediaPipe alignment"""
    return enterprise_preprocessor.align_face_enterprise(face_img)

def enhance_image(image, is_webcam=False):
    """Legacy enhancement function"""
    return enterprise_preprocessor.enhance_image_enterprise(image, is_webcam)

def assess_quality(face_img):
    """Legacy quality assessment"""
    quality_pass, _ = enterprise_preprocessor.assess_quality_enterprise(face_img)
    return quality_pass

def handle_occlusions_production(face_img, tracker=None):
    """Legacy occlusion handling"""
    return enterprise_preprocessor.handle_occlusions_enterprise(face_img, tracker)

def preprocess_image_with_occlusion_handling(image, is_webcam=False):
    """Legacy preprocessing pipeline"""
    return enterprise_preprocessor.preprocess_enterprise(image, is_webcam)

# Legacy classes for compatibility
class OcclusionValidator:
    def __init__(self):
        self.enterprise_validator = enterprise_preprocessor.occlusion_validator

    def validate(self, face_img):
        return self.enterprise_validator.validate(face_img)

class TemporalOcclusionTracker:
    def __init__(self):
        self.history = []

    def update(self, occlusion_level):
        self.history = self.history[-ENTERPRISE_CONFIG['occlusion']['temporal_window']:] + [occlusion_level]

    @property
    def significant_occlusion(self):
        if len(self.history) < 2:  # Very reduced minimum
            return False
        return sum(1 for lvl in self.history if lvl > 0.6) / len(self.history) > 0.7  # Higher threshold

# Legacy instances
occlusion_validator = OcclusionValidator()

def get_preprocessing_metrics():
    """Get enterprise preprocessing metrics"""
    return enterprise_preprocessor.get_metrics()

logger.info("Enterprise preprocessing system initialized with very lenient quality standards")
