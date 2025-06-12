import cv2
import numpy as np
import logging
from .preprocessing import enhance_image

# Setup logging
logger = logging.getLogger(__name__)

# Production Configuration - More lenient for better compatibility
PRODUCTION_CONFIG = {
    'default_target_size': (160, 160),
    'enable_enhancement': True,
    'debug_mode': False,
    'max_image_size': (4096, 4096),  # Increased for high-res images
    'min_image_size': (24, 24),  # Reduced minimum size
    'quality_checks': True,
    'memory_optimization': True,
    'batch_processing': True
}


def load_image_from_bytes(img_bytes):
    """
    Production image loading from bytes with enhanced error handling.
    """
    try:
        if img_bytes is None or len(img_bytes) == 0:
            logger.warning("Empty image bytes provided")
            return None

        # Convert bytes to numpy array with better error handling
        try:
            np_arr = np.frombuffer(img_bytes, np.uint8)
        except Exception as e:
            logger.error(f"Failed to convert bytes to numpy array: {e}")
            return None

        if np_arr.size == 0:
            logger.warning("Invalid image bytes - empty array")
            return None

        # Decode image with multiple attempts
        img = None
        for flag in [cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED, cv2.IMREAD_ANYCOLOR]:
            try:
                img = cv2.imdecode(np_arr, flag)
                if img is not None:
                    break
            except Exception as e:
                logger.debug(f"Decode attempt failed with flag {flag}: {e}")
                continue

        if img is None:
            logger.warning("Failed to decode image with all methods")
            return None

        # Handle different image formats
        if len(img.shape) == 3 and img.shape[2] == 4:
            # Convert RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            # Convert grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            logger.warning(f"Unsupported image format: {img.shape}")
            return None

        # Validate image size - more lenient
        h, w = img.shape[:2]
        max_h, max_w = PRODUCTION_CONFIG['max_image_size']
        min_h, min_w = PRODUCTION_CONFIG['min_image_size']

        if h > max_h or w > max_w:
            # Resize large images instead of rejecting
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized large image from {w}x{h} to {new_w}x{new_h}")

        if h < min_h or w < min_w:
            logger.warning(f"Image too small: {w}x{h}, min required: {min_w}x{min_h}")
            return None

        # Apply enhancement if enabled
        if PRODUCTION_CONFIG['enable_enhancement']:
            try:
                enhanced_img = production_enhance_image(img)
                if enhanced_img is not None:
                    img = enhanced_img
            except Exception as e:
                logger.warning(f"Enhancement failed, using original: {e}")

        if PRODUCTION_CONFIG['debug_mode']:
            logger.debug(f"Image loaded and processed: {img.shape}")

        return img

    except Exception as e:
        logger.error(f"Critical error loading image from bytes: {e}")
        return None


def production_enhance_image(image):
    """
    Production image enhancement with fallback to original preprocessing
    """
    try:
        # Try to use the existing enhance_image function
        enhanced = enhance_image(image, is_webcam=False)
        if enhanced is not None:
            return enhanced
    except Exception as e:
        logger.debug(f"Preprocessing enhancement failed: {e}")

    # Fallback to simple enhancement
    try:
        if image is None or image.size == 0:
            return image

        # Simple brightness and contrast adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return enhanced

    except Exception as e:
        logger.warning(f"Fallback enhancement failed: {e}")
        return image


def production_preprocess_face(face, target_size=None):
    """
    Production face preprocessing optimized for FaceNet - CRITICAL: Must match recognition.py
    """
    try:
        if target_size is None:
            target_size = PRODUCTION_CONFIG['default_target_size']

        if face is None or face.size == 0:
            logger.warning("Empty face image provided")
            return None

        # Validate input shape
        if len(face.shape) != 3 or face.shape[2] != 3:
            logger.warning(f"Invalid face shape: {face.shape}, expected (H, W, 3)")
            return None

        # Ensure face is in correct data type
        if face.dtype != np.uint8:
            if face.max() <= 1.0:
                face = (face * 255).astype(np.uint8)
            else:
                face = face.astype(np.uint8)

        # Resize to target size with high-quality interpolation
        if face.shape[:2] != target_size[::-1]:  # target_size is (W, H), shape is (H, W)
            face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_CUBIC)
        else:
            face_resized = face.copy()

        # CRITICAL: This preprocessing must match what's used in recognition.py
        # Convert to PIL Image format for consistency with recognition pipeline
        # The actual normalization will be done by the transform in recognition.py

        if PRODUCTION_CONFIG['debug_mode']:
            logger.debug(f"Face preprocessed: {face_resized.shape}, dtype: {face_resized.dtype}")

        return face_resized

    except Exception as e:
        logger.error(f"Error preprocessing face: {e}")
        return None


def validate_image_production(image):
    """
    Production image validation with enhanced checks.
    """
    try:
        if image is None:
            return False

        if not isinstance(image, np.ndarray):
            return False

        if image.size == 0:
            return False

        if len(image.shape) not in [2, 3]:
            return False

        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False

        # Check size constraints - more lenient
        h, w = image.shape[:2]
        max_h, max_w = PRODUCTION_CONFIG['max_image_size']
        min_h, min_w = PRODUCTION_CONFIG['min_image_size']

        if h > max_h or w > max_w:
            # Allow large images - they can be resized
            logger.debug(f"Large image detected: {w}x{h}")

        if h < min_h or w < min_w:
            return False

        # Check data type - more permissive
        if image.dtype not in [np.uint8, np.float32, np.float64, np.int32, np.int64]:
            logger.warning(f"Unusual image dtype: {image.dtype}")
            return False

        # Check for reasonable pixel values
        if image.dtype == np.uint8:
            if image.min() < 0 or image.max() > 255:
                return False
        elif image.dtype in [np.float32, np.float64]:
            if image.max() > 255 and image.min() >= 0:
                # Likely 0-255 range in float
                pass
            elif image.max() <= 1.0 and image.min() >= 0:
                # Likely 0-1 range
                pass
            elif image.max() <= 1.0 and image.min() >= -1.0:
                # Likely normalized range
                pass
            else:
                logger.warning(f"Unusual pixel value range: [{image.min()}, {image.max()}]")

        return True

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False


def convert_to_rgb_production(image):
    """
    Production RGB conversion with enhanced error handling.
    """
    try:
        if not validate_image_production(image):
            return None

        # Handle different input formats
        if len(image.shape) == 2:
            # Grayscale to RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            if image.shape[2] == 1:
                # Single channel to RGB
                image_2d = image.squeeze(axis=2)
                return cv2.cvtColor(image_2d, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                # Could be BGR or RGB - assume BGR and convert
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                # RGBA to RGB
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                logger.warning(f"Unsupported number of channels: {image.shape[2]}")
                return None
        else:
            logger.warning(f"Unsupported image shape: {image.shape}")
            return None

    except Exception as e:
        logger.error(f"Error converting to RGB: {e}")
        return None


def resize_with_aspect_ratio_production(image, target_size, pad_color=(0, 0, 0)):
    """
    Production aspect-ratio preserving resize with padding.
    """
    try:
        if not validate_image_production(image):
            return None

        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)

        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image with appropriate interpolation
        if scale < 1.0:
            # Downscaling - use INTER_AREA for better quality
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Upscaling - use INTER_CUBIC for better quality
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Create padded image
        if len(image.shape) == 3:
            padded = np.full((target_h, target_w, image.shape[2]), pad_color, dtype=image.dtype)
        else:
            padded = np.full((target_h, target_w), pad_color[0], dtype=image.dtype)

        # Calculate padding offsets (center the image)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        # Place resized image in center
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return padded

    except Exception as e:
        logger.error(f"Error resizing with aspect ratio: {e}")
        return None


def normalize_image_production(image, method='facenet'):
    """
    Production image normalization - CRITICAL: Must match recognition pipeline
    """
    try:
        if not validate_image_production(image):
            return None

        # Convert to float32
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        elif image.dtype in [np.float64]:
            img_float = image.astype(np.float32)
        else:
            img_float = image.astype(np.float32)

        # Ensure values are in expected range
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

        if method == 'facenet':
            # FaceNet normalization: [-1, 1] - CRITICAL: Must match recognition.py
            return (img_float - 0.5) / 0.5
        elif method == 'standard':
            # Standard normalization: mean=0, std=1
            mean = img_float.mean()
            std = img_float.std()
            if std < 1e-8:
                return img_float - mean
            return (img_float - mean) / std
        elif method == 'minmax':
            # Min-max normalization: [0, 1]
            min_val = img_float.min()
            max_val = img_float.max()
            if max_val - min_val < 1e-8:
                return img_float
            return (img_float - min_val) / (max_val - min_val)
        elif method == 'none':
            # No normalization - return as is
            return img_float
        else:
            logger.warning(f"Unknown normalization method: {method}, using 'none'")
            return img_float

    except Exception as e:
        logger.error(f"Error normalizing image: {e}")
        return None


def crop_face_production(image, bbox, padding=0.1):
    """
    Production face cropping with enhanced padding and validation.
    """
    try:
        if not validate_image_production(image):
            return None

        # Handle different bbox formats
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            logger.error(f"Invalid bbox format: {bbox}")
            return None

        h, w = image.shape[:2]

        # Validate bbox coordinates
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid bounding box coordinates")
            return None

        # Calculate padding
        face_w = x2 - x1
        face_h = y2 - y1

        # Use different padding for width and height if face is not square
        pad_w = int(face_w * padding)
        pad_h = int(face_h * padding)

        # Apply padding and clamp to image bounds
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)

        # Final validation
        if x2_pad <= x1_pad or y2_pad <= y1_pad:
            logger.warning("Invalid bounding box after padding")
            return None

        # Crop face
        face_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

        if face_crop.size == 0:
            logger.warning("Empty face crop")
            return None

        return face_crop

    except Exception as e:
        logger.error(f"Error cropping face: {e}")
        return None


def batch_preprocess_faces(faces, target_size=(160, 160)):
    """
    Batch preprocessing for multiple faces - enterprise optimization
    """
    try:
        if not faces:
            return []

        processed_faces = []
        for i, face in enumerate(faces):
            try:
                processed = production_preprocess_face(face, target_size)
                if processed is not None:
                    processed_faces.append(processed)
                else:
                    logger.warning(f"Failed to preprocess face {i}")
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")

        return processed_faces

    except Exception as e:
        logger.error(f"Batch preprocessing failed: {e}")
        return []


def safe_image_operation(func, *args, **kwargs):
    """
    Safe wrapper for image operations with error handling
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Image operation {func.__name__} failed: {e}")
        return None


# Legacy functions for backward compatibility
def preprocess_face(face, target_size=(160, 160)):
    """Legacy function - redirects to production preprocessing."""
    return production_preprocess_face(face, target_size)


def preprocess_face_for_facenet(face, target_size=(160, 160)):
    """Legacy function - redirects to production preprocessing."""
    return production_preprocess_face(face, target_size)


def validate_image(image):
    """Legacy function - redirects to production validation."""
    return validate_image_production(image)


def convert_to_rgb(image):
    """Legacy function - redirects to production RGB conversion."""
    return convert_to_rgb_production(image)


def resize_with_aspect_ratio(image, target_size, pad_color=(0, 0, 0)):
    """Legacy function - redirects to production resize."""
    return resize_with_aspect_ratio_production(image, target_size, pad_color)


def normalize_image(image, method='facenet'):
    """Legacy function - redirects to production normalization."""
    return normalize_image_production(image, method)


def crop_face(image, bbox, padding=0.1):
    """Legacy function - redirects to production cropping."""
    return crop_face_production(image, bbox, padding)


def test_production_utils():
    """Test the production utility functions with realistic scenarios."""
    try:
        logger.info("🧪 Testing production utility functions...")

        # Test with realistic image sizes and formats
        test_cases = [
            (200, 200, 3),  # Standard RGB
            (160, 160, 3),  # FaceNet size
            (100, 150, 3),  # Non-square
            (50, 50, 3),  # Small face
        ]

        for h, w, c in test_cases:
            logger.info(f"Testing with {w}x{h}x{c} image...")

            # Create test image with realistic pixel values
            test_img = np.random.randint(50, 200, (h, w, c), dtype=np.uint8)

            # Test validation
            is_valid = validate_image_production(test_img)
            logger.info(f"  ✅ Validation: {is_valid}")

            if not is_valid:
                continue

            # Test preprocessing
            processed = production_preprocess_face(test_img)
            if processed is not None:
                logger.info(f"  ✅ Preprocessing: {processed.shape}")
            else:
                logger.warning(f"  ❌ Preprocessing failed")

            # Test RGB conversion
            rgb_img = convert_to_rgb_production(test_img)
            if rgb_img is not None:
                logger.info(f"  ✅ RGB conversion: {rgb_img.shape}")
            else:
                logger.warning(f"  ❌ RGB conversion failed")

            # Test normalization
            normalized = normalize_image_production(test_img, 'facenet')
            if normalized is not None:
                logger.info(f"  ✅ Normalization: range [{normalized.min():.3f}, {normalized.max():.3f}]")
            else:
                logger.warning(f"  ❌ Normalization failed")

        # Test face cropping with various bbox sizes
        test_img = np.random.randint(80, 180, (300, 300, 3), dtype=np.uint8)
        test_bboxes = [
            (50, 50, 150, 150),  # Square face
            (40, 60, 120, 180),  # Rectangular face
            (10, 10, 80, 80),  # Small face
            (200, 200, 290, 290),  # Large face near edge
        ]

        for bbox in test_bboxes:
            cropped = crop_face_production(test_img, bbox)
            if cropped is not None:
                logger.info(f"  ✅ Face cropping {bbox}: {cropped.shape}")
            else:
                logger.warning(f"  ❌ Face cropping failed for {bbox}")

        # Test batch processing
        batch_faces = [np.random.randint(80, 180, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        batch_result = batch_preprocess_faces(batch_faces)
        logger.info(f"  ✅ Batch processing: {len(batch_result)}/{len(batch_faces)} successful")

        logger.info("🎉 Production utility tests completed!")
        return True

    except Exception as e:
        logger.error(f"❌ Utility test failed: {e}")
        return False


# Legacy test function
def test_utils():
    """Legacy test function - redirects to production test."""
    return test_production_utils()


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    test_production_utils()
