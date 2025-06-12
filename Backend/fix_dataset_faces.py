import os
import cv2
import numpy as np
import sys
import logging
import time
import gc
from pathlib import Path

# Configure logging WITHOUT Unicode characters for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_preprocessing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Force UTF-8 encoding for console output on Windows
if sys.platform.startswith('win'):
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Enterprise Configuration - MEMORY OPTIMIZED for Windows
ENTERPRISE_CONFIG = {
    'enable_super_resolution': False,
    'quality_threshold': 0.15,
    'min_face_size': (60, 60),
    'max_attempts': 3,  # Reduced to save memory
    'save_original': False,  # Disabled to save memory
    'output_format': 'jpg',
    'batch_size': 10,  # Much smaller batches
    'enable_enhancement': False,  # Disabled to save memory
    'skip_processed': True,
    'create_backup': False,
    'progress_interval': 50,
    'memory_cleanup_interval': 25,  # Force garbage collection
    'max_image_size': (1024, 1024),  # Limit image size
    'process_timeout': 30  # Timeout per image
}

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
services_dir = os.path.join(current_dir, 'services')
sys.path.insert(0, current_dir)
sys.path.insert(0, services_dir)

# Memory-efficient imports with fallbacks
detection_available = False
preprocessing_available = False

try:
    from detection import detect_face

    detection_available = True
    logger.info("SUCCESS: Detection module loaded")
except ImportError:
    logger.warning("Detection module not available, using fallback")

try:
    from preprocessing import preprocess_image_with_occlusion_handling

    preprocessing_available = True
    logger.info("SUCCESS: Preprocessing module loaded")
except ImportError:
    logger.warning("Preprocessing module not available, using fallback")


def memory_efficient_detect_face(img):
    """Memory-efficient face detection with fallbacks"""
    try:
        # Try enterprise detection first if available
        if detection_available:
            try:
                result = detect_face(img, is_webcam=False)
                if isinstance(result, tuple) and len(result) >= 1:
                    return result[0], "enterprise"
                return None, "enterprise_failed"
            except Exception as e:
                logger.debug(f"Enterprise detection failed: {e}")

        # Fallback to simple OpenCV detection
        try:
            # Use smaller image for detection to save memory
            h, w = img.shape[:2]
            if h > 800 or w > 800:
                scale = min(800 / w, 800 / h)
                small_img = cv2.resize(img, (int(w * scale), int(h * scale)))
            else:
                small_img = img
                scale = 1.0

            # Convert to grayscale for detection
            if len(small_img.shape) == 3:
                gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = small_img

            # Load cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Detect faces with conservative parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30),
                maxSize=(300, 300)
            )

            if len(faces) > 0:
                # Get largest face
                areas = [w * h for (x, y, w, h) in faces]
                largest_idx = np.argmax(areas)
                x, y, w, h = faces[largest_idx]

                # Scale back to original size
                if scale != 1.0:
                    x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)

                # Extract face with padding
                pad = 20
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(img.shape[1], x + w + pad)
                y2 = min(img.shape[0], y + h + pad)

                face_img = img[y1:y2, x1:x2]
                return face_img, "opencv_fallback"

            return None, "no_face_detected"

        except Exception as e:
            logger.debug(f"OpenCV detection failed: {e}")
            return None, f"opencv_error: {e}"

    except Exception as e:
        logger.error(f"All detection methods failed: {e}")
        return None, f"critical_error: {e}"


def memory_efficient_preprocess(img):
    """Memory-efficient preprocessing"""
    try:
        if preprocessing_available:
            try:
                result = preprocess_image_with_occlusion_handling(img, is_webcam=False)
                if isinstance(result, tuple) and len(result) >= 4:
                    return result[0], result[3]  # enhanced_face, quality_score
                return img, 0.5
            except Exception as e:
                logger.debug(f"Enterprise preprocessing failed: {e}")

        # Simple fallback preprocessing
        if img is None:
            return None, 0.0

        # Simple quality score based on contrast
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        quality = min(1.0, gray.std() / 50.0)  # Simple contrast-based quality

        return img, quality

    except Exception as e:
        logger.error(f"All preprocessing failed: {e}")
        return img, 0.3


def force_memory_cleanup():
    """Force memory cleanup"""
    try:
        gc.collect()
        cv2.destroyAllWindows()  # Close any OpenCV windows
    except:
        pass


class MemoryEfficientProcessor:
    """Memory-efficient dataset processor"""

    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed_detection': 0,
            'failed_quality': 0,
            'failed_processing': 0,
            'skipped_existing': 0,
            'start_time': time.time()
        }
        self.last_cleanup = 0

    def update_stats(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] += value

    def cleanup_if_needed(self):
        """Periodic memory cleanup"""
        if self.stats['total_processed'] - self.last_cleanup >= ENTERPRISE_CONFIG['memory_cleanup_interval']:
            force_memory_cleanup()
            self.last_cleanup = self.stats['total_processed']
            logger.debug(f"Memory cleanup at {self.stats['total_processed']} processed images")


processor = MemoryEfficientProcessor()


def process_single_image(img_path, output_path=None):
    """Memory-efficient single image processing"""
    try:
        # Cleanup check
        processor.cleanup_if_needed()

        if output_path is None:
            output_path = img_path

        # Skip if already processed
        if (ENTERPRISE_CONFIG['skip_processed'] and
                output_path != img_path and
                os.path.exists(output_path)):
            processor.update_stats(skipped_existing=1)
            return 1

        # Read image with size limit
        img = None
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not read image")

            # Limit image size to prevent memory issues
            h, w = img.shape[:2]
            max_h, max_w = ENTERPRISE_CONFIG['max_image_size']
            if h > max_h or w > max_w:
                scale = min(max_w / w, max_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.debug(f"Resized large image: {w}x{h} -> {new_w}x{new_h}")

        except Exception as e:
            logger.debug(f"Failed to read {img_path}: {e}")
            processor.update_stats(failed_processing=1)
            return 0

        # Convert to RGB
        try:
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            logger.debug(f"Color conversion failed for {img_path}: {e}")
            processor.update_stats(failed_processing=1)
            return 0

        # Face detection
        face_img, detection_method = memory_efficient_detect_face(img_rgb)

        if face_img is None:
            logger.debug(f"No face detected in {os.path.basename(img_path)}")
            processor.update_stats(failed_detection=1)
            return 0

        # Validate face size
        h, w = face_img.shape[:2]
        min_h, min_w = ENTERPRISE_CONFIG['min_face_size']
        if h < min_h or w < min_w:
            # Try to resize small faces
            scale = max(min_h / h, min_w / w)
            if scale <= 2.0:  # Only reasonable scaling
                new_h, new_w = int(h * scale), int(w * scale)
                face_img = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                logger.debug(f"Face too small in {os.path.basename(img_path)}: {w}x{h}")
                processor.update_stats(failed_quality=1)
                return 0

        # Preprocessing
        enhanced_face, quality_score = memory_efficient_preprocess(face_img)

        if enhanced_face is None:
            logger.debug(f"Preprocessing failed for {os.path.basename(img_path)}")
            processor.update_stats(failed_processing=1)
            return 0

        # Quality check
        if quality_score < ENTERPRISE_CONFIG['quality_threshold']:
            logger.debug(f"Low quality face in {os.path.basename(img_path)}: {quality_score:.2f}")
            processor.update_stats(failed_quality=1)
            return 0

        # Save processed image
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Convert to uint8 and BGR for saving
            if enhanced_face.dtype != np.uint8:
                if enhanced_face.max() <= 1.0:
                    enhanced_uint8 = (enhanced_face * 255).astype(np.uint8)
                else:
                    enhanced_uint8 = np.clip(enhanced_face, 0, 255).astype(np.uint8)
            else:
                enhanced_uint8 = enhanced_face

            if len(enhanced_uint8.shape) == 3:
                enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
            else:
                enhanced_bgr = enhanced_uint8

            # Save with compression
            success = cv2.imwrite(output_path, enhanced_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

            if not success:
                logger.warning(f"Failed to save {output_path}")
                processor.update_stats(failed_processing=1)
                return 0

            processor.update_stats(successful=1)
            return 1

        except Exception as e:
            logger.error(f"Save error for {img_path}: {e}")
            processor.update_stats(failed_processing=1)
            return 0

    except Exception as e:
        logger.error(f"Critical error processing {img_path}: {e}")
        processor.update_stats(failed_processing=1)
        return 0
    finally:
        processor.update_stats(total_processed=1)
        # Force cleanup of variables
        try:
            del img, img_rgb, face_img, enhanced_face
        except:
            pass


def preprocess_organized_dataset(dataset_dir="dataset"):
    """Memory-efficient organized dataset preprocessing"""

    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        return 0, 0

    logger.info("Starting organized dataset preprocessing...")
    logger.info(f"Dataset: {dataset_dir}")

    # Get person folders with error handling
    person_folders = []
    try:
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isdir(item_path):
                person_folders.append(item)
    except Exception as e:
        logger.error(f"Error reading dataset directory: {e}")
        return 0, 0

    if not person_folders:
        logger.error(f"No person folders found in {dataset_dir}")
        return 0, 0

    logger.info(f"Found {len(person_folders)} person folders")

    # Process each person with error handling
    for person_idx, person_name in enumerate(person_folders):
        try:
            person_folder = os.path.join(dataset_dir, person_name)
            logger.info(f"Processing {person_idx + 1}/{len(person_folders)}: {person_name}")

            # Get images for this person
            person_images = []
            try:
                for img_name in os.listdir(person_folder):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        person_images.append(img_name)
            except Exception as e:
                logger.warning(f"Error reading folder {person_name}: {e}")
                continue

            if not person_images:
                logger.warning(f"No images found for {person_name}")
                continue

            # Process images for this person
            person_start = processor.stats['successful']

            for img_name in person_images:
                try:
                    img_path = os.path.join(person_folder, img_name)
                    process_single_image(img_path)
                except Exception as e:
                    logger.debug(f"Error processing {img_name}: {e}")
                    continue

            person_processed = processor.stats['successful'] - person_start
            logger.info(f"   DONE: {person_name}: {person_processed}/{len(person_images)} images processed")

            # Force cleanup after each person
            force_memory_cleanup()

        except Exception as e:
            logger.error(f"Error processing person {person_name}: {e}")
            continue

    # Final report
    total = processor.stats['total_processed']
    successful = processor.stats['successful']
    success_rate = (successful / total * 100) if total > 0 else 0

    logger.info(f"\nDataset preprocessing complete!")
    logger.info(f"Results: {successful}/{total} images successfully processed")
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info(f"Failed detection: {processor.stats['failed_detection']}")
    logger.info(f"Failed quality: {processor.stats['failed_quality']}")
    logger.info(f"Failed processing: {processor.stats['failed_processing']}")

    return successful, total


def main():
    """Main function with memory-optimized settings"""
    import argparse

    parser = argparse.ArgumentParser(description='Memory-Efficient Dataset Face Preprocessing')
    parser.add_argument('--input', type=str, default='dataset', help='Input directory path')
    parser.add_argument('--quality', type=float, default=0.15, help='Quality threshold')
    parser.add_argument('--min-size', type=int, default=60, help='Minimum face size')

    args = parser.parse_args()

    # Update configuration
    ENTERPRISE_CONFIG['quality_threshold'] = args.quality
    ENTERPRISE_CONFIG['min_face_size'] = (args.min_size, args.min_size)

    logger.info(f"Configuration:")
    logger.info(f"  Quality threshold: {ENTERPRISE_CONFIG['quality_threshold']}")
    logger.info(f"  Minimum face size: {ENTERPRISE_CONFIG['min_face_size']}")
    logger.info(f"  Memory cleanup interval: {ENTERPRISE_CONFIG['memory_cleanup_interval']}")

    # Run preprocessing
    start_time = time.time()
    processed, total = preprocess_organized_dataset(args.input)
    elapsed_time = time.time() - start_time

    # Final summary
    success_rate = (processed / total * 100) if total > 0 else 0
    logger.info(f"\nFinal Summary:")
    logger.info(f"  Total time: {elapsed_time:.1f} seconds")
    logger.info(f"  Success rate: {success_rate:.1f}%")
    logger.info(f"  Ready for embedding generation: {processed} images")


if __name__ == "__main__":
    main()
