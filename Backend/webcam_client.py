import cv2
import numpy as np
import os
import sys
import time
import logging
from datetime import datetime
import requests
from collections import deque, defaultdict
import threading
import json
from concurrent.futures import ThreadPoolExecutor

# Add services to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from services.detection import enterprise_detector, get_detection_metrics
from services.preprocessing import enterprise_preprocessor, get_preprocessing_metrics, TemporalOcclusionTracker
from services.recognition import recognizer, get_recognition_metrics

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enterprise_face_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enterprise Configuration for 1000+ Employees
ENTERPRISE_CONFIG = {
    'cctv': {
        'url': "rtsp://admin:password@192.168.1.100:554/stream",
        'frame_size': (1280, 720),
        'fps': 15,
        'buffer_size': 1,  # Reduced for real-time
        'reconnect_interval': 5.0  # Auto-reconnect
    },
    'webcam': {
        'device_id': 0,
        'exposure': -6,  # Optimized for office lighting
        'focus': 40,  # Fixed focus for consistency
        'frame_size': (640, 480),
        'auto_white_balance': False
    },
    'recognition': {
        'temporal_window': 2.0,  # Increased for enterprise stability
        'min_detections': 6,  # Higher consensus requirement
        'confidence_threshold': 0.75,  # Higher threshold for enterprise
        'unknown_threshold': 0.6,  # Separate threshold for unknown
        'max_recognition_time': 3.0  # Timeout for recognition
    },
    'performance': {
        'frame_skip': 2,  # Process every 2nd frame
        'max_retries': 5,  # More retries for enterprise
        'batch_size': 4,  # Batch processing
        'max_concurrent_streams': 10,  # Support multiple cameras
        'memory_limit_mb': 2048  # Memory management
    },
    'enterprise': {
        'enable_api_fallback': True,  # API fallback for unknown faces
        'enable_registration': False,  # Disable for production
        'enable_monitoring': True,  # Performance monitoring
        'save_unknown_faces': True,  # Save for later analysis
        'audit_logging': True  # Compliance logging
    }
}


class EnterpriseTemporalTracker:
    """Advanced temporal consistency tracker for enterprise deployment"""

    def __init__(self, window_seconds=2.0, min_detections=6):
        self.window_seconds = window_seconds
        self.min_detections = min_detections
        self.recognition_history = deque()
        self.identity_scores = defaultdict(list)
        self.confidence_history = deque(maxlen=50)
        self.lock = threading.Lock()

    def add_recognition(self, identity, confidence, has_occlusion=False, quality_score=1.0):
        """Add recognition result with metadata"""
        current_time = time.time()

        with self.lock:
            # Add to history with metadata
            self.recognition_history.append({
                'identity': identity,
                'confidence': confidence,
                'has_occlusion': has_occlusion,
                'quality_score': quality_score,
                'timestamp': current_time
            })

            # Clean old entries
            cutoff_time = current_time - self.window_seconds
            while self.recognition_history and self.recognition_history[0]['timestamp'] < cutoff_time:
                self.recognition_history.popleft()

            # Update confidence history for monitoring
            self.confidence_history.append(confidence)

            # Update identity scores
            self._update_identity_scores()

    def _update_identity_scores(self):
        """Update weighted scores for each identity"""
        self.identity_scores.clear()

        for entry in self.recognition_history:
            identity = entry['identity']
            confidence = entry['confidence']
            quality = entry['quality_score']

            # Weight by quality and recency
            age_weight = 1.0  # Could add time decay here
            quality_weight = quality
            occlusion_penalty = 0.9 if entry['has_occlusion'] else 1.0

            weighted_confidence = confidence * age_weight * quality_weight * occlusion_penalty
            self.identity_scores[identity].append(weighted_confidence)

    def get_best_identity(self):
        """Get most confident identity with enterprise validation"""
        with self.lock:
            if not self.identity_scores:
                return "Unknown", 0.0

            best_identity = "Unknown"
            best_score = 0.0

            for identity, confidences in self.identity_scores.items():
                if len(confidences) >= self.min_detections:
                    # Use weighted average with recency bias
                    weights = np.linspace(0.7, 1.0, len(confidences))
                    weighted_avg = np.average(confidences, weights=weights)

                    # Stability bonus for consistent detections
                    stability_bonus = min(0.1, len(confidences) * 0.01)
                    total_score = weighted_avg + stability_bonus

                    if total_score > best_score:
                        best_score = total_score
                        best_identity = identity

            return best_identity, best_score

    def get_detection_count(self, identity):
        """Get number of detections for an identity"""
        return len(self.identity_scores.get(identity, []))

    def get_avg_confidence(self):
        """Get average confidence over recent detections"""
        if not self.confidence_history:
            return 0.0
        return np.mean(list(self.confidence_history))

    def clear(self):
        """Clear all tracking data"""
        with self.lock:
            self.recognition_history.clear()
            self.identity_scores.clear()


class EnterpriseVideoClient:
    """Enterprise-grade video client for 1000+ employee deployment"""

    def __init__(self, config=None, use_cctv=False):
        self.config = config or ENTERPRISE_CONFIG
        self.use_cctv = use_cctv
        self.cap = None
        self.tracker = EnterpriseTemporalTracker(
            window_seconds=self.config['recognition']['temporal_window'],
            min_detections=self.config['recognition']['min_detections']
        )
        self.occlusion_tracker = TemporalOcclusionTracker()
        self.frame_count = 0
        self.last_recognition_time = 0
        self.recognition_cooldown = 0.5  # Seconds between recognitions
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False

        # Enterprise monitoring
        self.session_stats = {
            'start_time': time.time(),
            'total_frames': 0,
            'successful_recognitions': 0,
            'unknown_faces': 0,
            'errors': 0
        }

    def initialize_camera(self):
        """Enterprise camera initialization with comprehensive error handling"""
        for attempt in range(self.config['performance']['max_retries']):
            try:
                if self.use_cctv:
                    logger.info(f"Connecting to CCTV: {self.config['cctv']['url']}")
                    self.cap = cv2.VideoCapture(self.config['cctv']['url'])

                    # CCTV optimizations
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['cctv']['frame_size'][0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['cctv']['frame_size'][1])
                    self.cap.set(cv2.CAP_PROP_FPS, self.config['cctv']['fps'])
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['cctv']['buffer_size'])
                else:
                    logger.info(f"Connecting to webcam: {self.config['webcam']['device_id']}")
                    self.cap = cv2.VideoCapture(self.config['webcam']['device_id'])

                    # Webcam optimizations
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['webcam']['frame_size'][0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['webcam']['frame_size'][1])
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config['webcam']['exposure'])
                    self.cap.set(cv2.CAP_PROP_FOCUS, self.config['webcam']['focus'])

                    if not self.config['webcam']['auto_white_balance']:
                        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

                if self.cap.isOpened():
                    # Verify camera properties
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = self.cap.get(cv2.CAP_PROP_FPS)

                    logger.info(f"✅ Camera initialized: {width}x{height} @ {fps} FPS")
                    return True

            except Exception as e:
                logger.error(f"Camera init attempt {attempt + 1} failed: {e}")
                if attempt < self.config['performance']['max_retries'] - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        logger.error("❌ Failed to initialize camera after all attempts")
        return False

    def process_frame(self, frame):
        """Enterprise frame processing with comprehensive pipeline"""
        try:
            current_time = time.time()

            # Skip if in cooldown period
            if current_time - self.last_recognition_time < self.recognition_cooldown:
                return None

            # Frame skipping for performance
            self.frame_count += 1
            if self.frame_count % self.config['performance']['frame_skip'] != 0:
                return None

            self.session_stats['total_frames'] += 1

            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face detection with enterprise detector
            face_result = enterprise_detector.detect(img_rgb, is_webcam=not self.use_cctv)

            if face_result[0] is None:
                logger.debug(f"No face detected: {face_result[2]}")
                return None

            face_img, occlusion_info, detection_method = face_result
            logger.debug(f"Face detected by {detection_method}")

            # Preprocessing with enterprise preprocessor
            processed_result = enterprise_preprocessor.preprocess_enterprise(
                face_img,
                is_webcam=not self.use_cctv,
                tracker=self.occlusion_tracker
            )

            if processed_result[0] is None:
                logger.debug("Preprocessing failed")
                return None

            enhanced_face, mask, preprocessing_info, quality_score = processed_result

            # Check if face quality is acceptable
            if quality_score < 0.4:
                logger.debug(f"Low quality face: {quality_score:.2f}")
                return None

            # Recognition with enterprise recognizer
            identity, confidence = recognizer.recognize_face(
                recognizer.get_embedding(enhanced_face),
                confidence_threshold=self.config['recognition']['confidence_threshold']
            )

            # Apply enterprise thresholds
            if confidence < self.config['recognition']['unknown_threshold']:
                identity = "Unknown"

            # Update temporal tracker
            has_occlusion = preprocessing_info.get('confidence', 0) > 0.3
            self.tracker.add_recognition(identity, confidence, has_occlusion, quality_score)

            # Get temporal result
            temporal_identity, temporal_confidence = self.tracker.get_best_identity()
            detection_count = self.tracker.get_detection_count(temporal_identity)

            # Final confidence check
            if temporal_confidence < self.config['recognition']['confidence_threshold']:
                temporal_identity = "Unknown"

            # Update session stats
            if temporal_identity != "Unknown":
                self.session_stats['successful_recognitions'] += 1
            else:
                self.session_stats['unknown_faces'] += 1

                # Save unknown face for analysis if enabled
                if self.config['enterprise']['save_unknown_faces']:
                    self._save_unknown_face(enhanced_face)

            result = {
                'identity': temporal_identity,
                'confidence': temporal_confidence,
                'detection_count': detection_count,
                'quality_score': quality_score,
                'has_occlusion': has_occlusion,
                'detection_method': detection_method,
                'face_image': enhanced_face
            }

            self.last_recognition_time = current_time
            return result

        except Exception as e:
            logger.error(f"❌ Error during frame processing: {e}")
            self.session_stats['errors'] += 1
            return None

    def _save_unknown_face(self, face_img):
        """Save unknown faces for later analysis"""
        try:
            unknown_dir = "unknown_faces"
            os.makedirs(unknown_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"unknown_{timestamp}.jpg"
            filepath = os.path.join(unknown_dir, filename)

            face_uint8 = np.clip(face_img, 0, 255).astype(np.uint8)
            cv2.imwrite(filepath, cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR))

        except Exception as e:
            logger.error(f"Failed to save unknown face: {e}")

    def try_api_recognition(self, face_image):
        """Enterprise API fallback with timeout"""
        if not self.config['enterprise']['enable_api_fallback']:
            return None

        try:
            logger.info("🌐 Trying API recognition...")
            _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
            files = {'file': img_encoded.tobytes()}

            response = requests.post(
                "http://127.0.0.1:8000/recognize",
                files=files,
                timeout=self.config['recognition']['max_recognition_time']
            )

            if response.status_code == 200:
                result = response.json()
                api_identity = result.get('identity', 'Unknown')
                if api_identity != "Unknown":
                    logger.info(f"✅ API recognition: {api_identity}")
                    return api_identity

        except requests.exceptions.Timeout:
            logger.warning("⚠️ API request timed out")
        except Exception as e:
            logger.warning(f"⚠️ API request failed: {e}")

        return None

    def get_system_metrics(self):
        """Get comprehensive system metrics"""
        runtime = time.time() - self.session_stats['start_time']

        metrics = {
            'session': self.session_stats.copy(),
            'runtime_seconds': runtime,
            'fps': self.session_stats['total_frames'] / runtime if runtime > 0 else 0,
            'recognition_rate': self.session_stats['successful_recognitions'] / max(1,
                                                                                    self.session_stats['total_frames']),
            'temporal_tracker': {
                'avg_confidence': self.tracker.get_avg_confidence(),
                'active_identities': len(self.tracker.identity_scores)
            },
            'detection': get_detection_metrics(),
            'preprocessing': get_preprocessing_metrics(),
            'recognition': get_recognition_metrics()
        }

        return metrics

    def run(self):
        """Enterprise main loop with comprehensive monitoring"""
        if not self.initialize_camera():
            return False

        self.running = True
        logger.info("🚀 Enterprise video client started")
        logger.info("Press 'r' to recognize, 'c' to clear tracking, 'm' for metrics, 'q' to quit")

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    if self.use_cctv:
                        # Auto-reconnect for CCTV
                        logger.info("Attempting CCTV reconnection...")
                        time.sleep(self.config['cctv']['reconnect_interval'])
                        if not self.initialize_camera():
                            break
                    continue

                # Display frame with enterprise info
                display_frame = frame.copy()
                self._add_enterprise_overlay(display_frame)

                cv2.imshow("Enterprise Face Recognition", display_frame)
                key = cv2.waitKey(1) & 0xFF

                # Process frame on 'r' key
                if key == ord('r'):
                    result = self.process_frame(frame)

                    if result:
                        identity = result['identity']
                        confidence = result['confidence']
                        detection_count = result['detection_count']

                        logger.info(f"🎯 Recognition: {identity} (confidence: {confidence:.2f}, "
                                    f"detections: {detection_count})")

                        # Try API if unknown and enabled
                        if identity == "Unknown":
                            api_identity = self.try_api_recognition(result['face_image'])
                            if api_identity:
                                identity = api_identity

                        logger.info(f"✅ Final result: {identity}")

                # Clear tracking on 'c' key
                elif key == ord('c'):
                    self.tracker.clear()
                    logger.info("🔄 Tracking data cleared")

                # Show metrics on 'm' key
                elif key == ord('m'):
                    metrics = self.get_system_metrics()
                    logger.info(f"📊 System Metrics: {json.dumps(metrics, indent=2)}")

                # Quit on 'q' key
                elif key == ord('q'):
                    logger.info("👋 Shutting down...")
                    self.running = False
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()

        return True

    def _add_enterprise_overlay(self, frame):
        """Add enterprise information overlay to frame"""
        try:
            # Add system info
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            runtime = time.time() - self.session_stats['start_time']
            fps = self.session_stats['total_frames'] / runtime if runtime > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add recognition stats
            success_rate = (self.session_stats['successful_recognitions'] /
                            max(1, self.session_stats['total_frames']) * 100)
            cv2.putText(frame, f"Success: {success_rate:.1f}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            logger.error(f"Overlay error: {e}")

    def cleanup(self):
        """Enterprise cleanup with comprehensive resource management"""
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown(wait=True)

            # Log final metrics
            final_metrics = self.get_system_metrics()
            logger.info(f"📊 Final Session Metrics: {json.dumps(final_metrics, indent=2)}")
            logger.info("📹 Enterprise video client cleaned up")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Main entry point for enterprise deployment"""
    import argparse

    parser = argparse.ArgumentParser(description='Enterprise Face Recognition Client')
    parser.add_argument('--cctv', action='store_true', help='Use CCTV mode')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()

    # Load custom config if provided
    config = ENTERPRISE_CONFIG
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)

    # Create and run client
    client = EnterpriseVideoClient(config, use_cctv=args.cctv)
    success = client.run()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
