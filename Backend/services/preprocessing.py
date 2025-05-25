# preprocessing.py

import cv2
import numpy as np
import os
from occlusion_detection import OcclusionDetector


# Super-resolution model class
class SuperResolution:
    def __init__(self, model_name="espcn", scale=2):
        """
        Initialize super-resolution model

        Args:
            model_name: Model architecture ('espcn', 'fsrcnn', or 'lapsrn')
            scale: Upscaling factor (2, 3, or 4 depending on model)
        """
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()

        # Define model paths
        models_dir = os.path.join("..", "assets", "sr_models")
        os.makedirs(models_dir, exist_ok=True)

        # Model file paths based on architecture and scale
        model_files = {
            "espcn": f"ESPCN_x{scale}.pb",
            "fsrcnn": f"FSRCNN_x{scale}.pb",
            "lapsrn": f"LapSRN_x{scale}.pb"
        }

        if model_name not in model_files:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model_path = os.path.join(models_dir, model_files[model_name])

        # Check if model exists
        if not os.path.exists(self.model_path):
            print(f"Warning: SR model not found at {self.model_path}")
            print("Super-resolution will be disabled.")
            self.sr = None
            return

        try:
            # Load the model
            self.sr.readModel(self.model_path)
            self.sr.setModel(model_name, scale)
        except Exception as e:
            print(f"Error loading SR model: {e}")
            self.sr = None

    def upsample(self, image):
        """
        Apply super-resolution to an image

        Args:
            image: Input image (grayscale or color)

        Returns:
            Super-resolved image
        """
        if self.sr is None:
            return image

        try:
            return self.sr.upsample(image)
        except Exception as e:
            print(f"Super-resolution failed: {e}")
            return image


# Initialize super-resolution model (lazy loading)
_sr_model = None


def get_sr_model():
    """Lazy initialization of super-resolution model"""
    global _sr_model
    if _sr_model is None:
        try:
            _sr_model = SuperResolution(model_name="espcn", scale=2)
        except Exception as e:
            print(f"Could not initialize super-resolution model: {e}")
            return None
    return _sr_model


def apply_super_resolution(image):
    """Apply super-resolution to enhance image details"""
    sr_model = get_sr_model()
    if sr_model is None or sr_model.sr is None:
        return image

    # Process based on image type
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Color image - split channels and process individually
        b, g, r = cv2.split(image)
        try:
            b_sr = sr_model.upsample(b)
            g_sr = sr_model.upsample(g)
            r_sr = sr_model.upsample(r)
            # Merge channels
            return cv2.merge([b_sr, g_sr, r_sr])
        except Exception as e:
            print(f"Color super-resolution failed: {e}")
            return image
    else:
        # Grayscale image
        try:
            return sr_model.upsample(image)
        except Exception as e:
            print(f"Grayscale super-resolution failed: {e}")
            return image


def extract_multi_scale_gabor_features_enhanced(img_patch, scales=[0.8, 1.0, 1.2]):
    """Enhanced multi-scale Gabor feature extraction"""
    if len(img_patch.shape) == 3:
        img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY)

    all_features = []

    for scale in scales:
        # Resize patch
        h, w = img_patch.shape
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h <= 0 or new_w <= 0:
            continue

        scaled_patch = cv2.resize(img_patch, (new_w, new_h))
        scaled_patch = cv2.resize(scaled_patch, (64, 64))  # Normalize back to 64x64

        # Extract enhanced Gabor features
        gabor_features = []
        for theta in np.arange(0, np.pi, np.pi / 6):  # 6 orientations
            for sigma in [2, 4, 6]:  # 3 scales
                for lambd in [8, 12, 16]:  # 3 wavelengths
                    try:
                        kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
                        filtered = cv2.filter2D(scaled_patch, cv2.CV_8UC3, kernel)

                        # Enhanced statistical features
                        gabor_features.extend([
                            np.mean(filtered),
                            np.std(filtered),
                            np.max(filtered),
                            np.min(filtered),
                            np.median(filtered)
                        ])
                    except Exception as e:
                        print(f"Gabor filter error: {e}")
                        # Add zeros if filter fails
                        gabor_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        all_features.extend(gabor_features)

    return np.array(all_features)


def create_hierarchical_occlusion_mask(face_img, occlusion_detector):
    """Create detailed occlusion mask with hierarchical regions"""
    h, w = face_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    # Define facial regions
    regions = {
        'forehead': (0, h // 4, 0, w),
        'eyes': (h // 4, h // 2, 0, w),
        'nose': (h // 3, 2 * h // 3, w // 3, 2 * w // 3),
        'mouth': (2 * h // 3, h, w // 4, 3 * w // 4),
        'left_cheek': (h // 3, 2 * h // 3, 0, w // 3),
        'right_cheek': (h // 3, 2 * h // 3, 2 * w // 3, w),
        'chin': (3 * h // 4, h, 0, w)
    }

    # Analyze each region
    for region_name, (y1, y2, x1, x2) in regions.items():
        # Ensure coordinates are within bounds
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)

        if y2 <= y1 or x2 <= x1:
            continue

        region_patch = face_img[y1:y2, x1:x2]

        if region_patch.size > 0:
            try:
                # Extract enhanced features
                features = extract_multi_scale_gabor_features_enhanced(region_patch)
                if features.size == 0:
                    continue

                features = features.reshape(1, -1)

                # Use appropriate classifier
                if (hasattr(occlusion_detector, 'pca') and
                        hasattr(occlusion_detector, 'upper_classifier') and
                        hasattr(occlusion_detector, 'lower_classifier')):

                    features_pca = occlusion_detector.pca.transform(features)
                    if region_name in ['forehead', 'eyes']:
                        occlusion_prob = occlusion_detector.upper_classifier.predict_proba(features_pca)[0][1]
                    else:
                        occlusion_prob = occlusion_detector.lower_classifier.predict_proba(features_pca)[0][1]
                else:
                    occlusion_prob = 0.0

            except Exception as e:
                print(f"Error processing region {region_name}: {e}")
                occlusion_prob = 0.0

            mask[y1:y2, x1:x2] = occlusion_prob

    # Apply adaptive thresholding and morphological operations
    binary_mask = (mask > 0.5).astype(np.uint8)

    # Enhanced MRF-like refinement
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)

    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    binary_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (5, 5), 1.0)

    return binary_mask, mask


def detect_occlusions_enhanced(face_img, occlusion_detector):
    """Enhanced occlusion detection with confidence scores"""
    try:
        h, w = face_img.shape[:2]

        # Extract upper region
        upper_patch = face_img[0:h // 2, :]
        upper_features = extract_multi_scale_gabor_features_enhanced(upper_patch)

        if upper_features.size == 0:
            raise ValueError("No features extracted from upper region")

        upper_features = upper_features.reshape(1, -1)

        if (hasattr(occlusion_detector, 'pca') and
                hasattr(occlusion_detector, 'upper_classifier')):
            upper_features_pca = occlusion_detector.pca.transform(upper_features)
            upper_proba = occlusion_detector.upper_classifier.predict_proba(upper_features_pca)[0]
            upper_occluded = upper_proba[1] > 0.5
            upper_confidence = upper_proba[1]
        else:
            upper_occluded = False
            upper_confidence = 0.0

        # Extract lower region
        lower_patch = face_img[h // 2:, :]
        lower_features = extract_multi_scale_gabor_features_enhanced(lower_patch)

        if lower_features.size == 0:
            raise ValueError("No features extracted from lower region")

        lower_features = lower_features.reshape(1, -1)

        if (hasattr(occlusion_detector, 'pca') and
                hasattr(occlusion_detector, 'lower_classifier')):
            lower_features_pca = occlusion_detector.pca.transform(lower_features)
            lower_proba = occlusion_detector.lower_classifier.predict_proba(lower_features_pca)[0]
            lower_occluded = lower_proba[1] > 0.5
            lower_confidence = lower_proba[1]
        else:
            lower_occluded = False
            lower_confidence = 0.0

        return {
            'upper_occluded': upper_occluded,
            'lower_occluded': lower_occluded,
            'upper_confidence': upper_confidence,
            'lower_confidence': lower_confidence
        }
    except Exception as e:
        print(f"Enhanced occlusion detection failed: {e}")
        return {
            'upper_occluded': False,
            'lower_occluded': False,
            'upper_confidence': 0.0,
            'lower_confidence': 0.0
        }


def apply_adaptive_enhancement(face_img, prob_mask):
    """Apply adaptive enhancement based on occlusion probability"""
    try:
        enhanced_face = face_img.copy().astype(np.float32)

        # Create attention map from occlusion mask
        attention_map = 1.0 - prob_mask
        attention_map = cv2.GaussianBlur(attention_map, (5, 5), 0)

        # Normalize attention weights
        if np.max(attention_map) > 0:
            attention_map = attention_map / np.max(attention_map)

        # Apply CLAHE more strongly to non-occluded regions
        lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Apply enhancement with spatial weighting
        l_channel = lab[:, :, 0]
        enhanced_l = clahe.apply(l_channel)

        # Blend based on attention map
        final_l = l_channel * prob_mask + enhanced_l * attention_map
        lab[:, :, 0] = final_l.astype(np.uint8)

        enhanced_face = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced_face
    except Exception as e:
        print(f"Adaptive enhancement failed: {e}")
        return face_img


def categorize_occlusion_level(occlusion_percentage):
    """Categorize occlusion level"""
    if occlusion_percentage < 0.1:
        return "minimal"
    elif occlusion_percentage < 0.3:
        return "moderate"
    elif occlusion_percentage < 0.5:
        return "high"
    else:
        return "severe"


def assess_face_quality(face_img):
    """Assess face image quality"""
    try:
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

        return quality_score
    except Exception as e:
        print(f"Quality assessment failed: {e}")
        return 0.5  # Return medium quality as fallback


def apply_hist_eq(image):
    """Apply histogram equalization to improve contrast"""
    try:
        # Convert to YCrCb and equalize only Y channel to preserve color
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)
    except Exception as e:
        print(f"Histogram equalization failed: {e}")
        return image


def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    except Exception as e:
        print(f"CLAHE failed: {e}")
        return image


def apply_gamma(image, gamma=1.5):
    """Apply gamma correction"""
    try:
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)
    except Exception as e:
        print(f"Gamma correction failed: {e}")
        return image


def apply_denoise(image):
    """Apply bilateral filter for denoising"""
    try:
        # Reduced parameters for faster processing while still effective
        return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
    except Exception as e:
        print(f"Denoising failed: {e}")
        return image


def apply_sharpen(image):
    """Apply sharpening to enhance facial features"""
    try:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    except Exception as e:
        print(f"Sharpening failed: {e}")
        return image


def normalize_brightness(image):
    """Normalize brightness to mid-range"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # Calculate current brightness
        mean_v = np.mean(v)

        # Target mid-brightness (128)
        target = 128

        # Calculate adjustment factor
        if mean_v > 0:
            factor = target / mean_v
            v = np.clip(v * factor, 0, 255).astype(np.uint8)

        hsv_adjusted = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)
    except Exception as e:
        print(f"Brightness normalization failed: {e}")
        return image


def enhance_image(image, use_sr=False):
    """
    Adaptive image enhancement based on image conditions

    Args:
        image: Input RGB image
        use_sr: Whether to apply super-resolution

    Returns:
        Enhanced image
    """
    try:
        # Apply super-resolution first if requested
        if use_sr:
            image = apply_super_resolution(image)

        # Get image brightness
        brightness = np.mean(image)

        # Different processing pipelines based on lighting conditions
        if brightness < 80:  # Dark images
            # For dark images, brighten first then enhance details
            image = apply_gamma(image, 2.5)  # Stronger gamma correction
            image = apply_clahe(image)
            image = apply_hist_eq(image)  # Add histogram equalization
            image = apply_sharpen(image)  # Add sharpening
            image = apply_denoise(image)  # Denoise at the end

        elif brightness > 200:  # Bright/overexposed images
            # For bright images, reduce brightness and enhance details
            image = apply_gamma(image, 0.8)  # Gamma < 1 darkens the image
            image = apply_clahe(image)
            image = apply_denoise(image)

        else:  # Normal lighting
            # Standard pipeline with added sharpening
            image = apply_gamma(image, 1.5)
            image = apply_clahe(image)
            image = apply_sharpen(image)  # Add sharpening for better feature definition
            image = apply_denoise(image)

        return image
    except Exception as e:
        print(f"Image enhancement failed: {e}")
        return image


# In preprocessing.py - Replace the existing handle_occlusions function
def handle_occlusions(face_img):
    """Working heuristic occlusion detection with adjusted thresholds"""
    try:
        enhanced_face = enhance_image(face_img)
        h, w = face_img.shape[:2]

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        mask = np.zeros((h, w), dtype=np.float32)

        # Upper region analysis (glasses detection) - RELAXED THRESHOLDS
        upper_region = gray[:h // 2, :]
        upper_mean = np.mean(upper_region)
        upper_std = np.std(upper_region)

        # More sensitive glasses detection
        glasses_detected = False
        glasses_confidence = 0.1

        # Method 1: Check for darkness (relaxed threshold)
        if upper_mean < 100:  # Changed from 80 to 100
            glasses_confidence = max(0.6, (100 - upper_mean) / 100)
            glasses_detected = True
            mask[:h // 2, :] = glasses_confidence

        # Method 2: Check for horizontal dark bands
        elif upper_std > 25:  # High contrast might indicate glasses frames
            horizontal_profile = np.mean(upper_region, axis=1)
            if np.min(horizontal_profile) < upper_mean * 0.7:  # Relative darkness
                glasses_confidence = 0.7
                glasses_detected = True
                mask[:h // 2, :] = glasses_confidence

        # Method 3: Edge-based detection for glasses frames
        edges_upper = cv2.Canny(upper_region, 30, 100)
        edge_density_upper = np.sum(edges_upper > 0) / edges_upper.size
        if edge_density_upper > 0.15:  # High edge density = frames
            glasses_confidence = max(glasses_confidence, 0.6)
            glasses_detected = True
            mask[:h // 2, :] = glasses_confidence

        # Lower region analysis (beard/mask detection)
        lower_region = gray[h // 2:, :]
        lower_mean = np.mean(lower_region)

        beard_detected = False
        beard_confidence = 0.1

        # Beard/mask detection with relaxed thresholds
        if lower_mean < 110:  # Slightly relaxed from 100
            edges = cv2.Canny(lower_region, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size

            if edge_density > 0.06:  # Reduced from 0.08
                beard_confidence = min(0.8, edge_density * 10)
                beard_detected = True
                mask[h // 2:, :] = beard_confidence

        # Create detailed occlusion info
        occlusion_info = {
            'upper_occluded': glasses_detected,
            'lower_occluded': beard_detected,
            'upper_confidence': glasses_confidence,
            'lower_confidence': beard_confidence,
            'total_occlusion_percentage': float(np.mean(mask)),
            'occlusion_level': 'high' if np.mean(mask) > 0.5 else 'moderate' if np.mean(mask) > 0.2 else 'minimal',
            # Debug info
            'debug_upper_mean': float(upper_mean),
            'debug_upper_std': float(upper_std),
            'debug_lower_mean': float(lower_mean)
        }

        return enhanced_face, mask, occlusion_info

    except Exception as e:
        print(f"Heuristic occlusion detection failed: {e}")
        # Return safe defaults
        enhanced_face = enhance_image(face_img) if 'enhance_image' in globals() else face_img
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


# Remove or comment out these functions (no longer needed):
# - create_hierarchical_occlusion_mask
# - detect_occlusions_enhanced
# - extract_multi_scale_gabor_features_enhanced


def preprocess_image_with_occlusion_handling(image, quality_threshold=0.3):
    """
    Complete enhanced preprocessing pipeline with occlusion handling
    Returns: preprocessed image, occlusion mask, occlusion info, quality score
    """
    try:
        # Assess image quality first
        quality_score = assess_face_quality(image)

        if quality_score < quality_threshold:
            # Return basic processing for low quality images
            basic_enhanced = enhance_image(image)
            dummy_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            dummy_info = {
                'upper_occluded': False,
                'lower_occluded': False,
                'upper_confidence': 0.0,
                'lower_confidence': 0.0,
                'total_occlusion_percentage': 0.0,
                'occlusion_level': 'minimal'
            }
            return basic_enhanced, dummy_mask, dummy_info, quality_score

        # Apply enhanced occlusion handling
        enhanced_img, occlusion_mask, occlusion_info = handle_occlusions(image)

        return enhanced_img, occlusion_mask, occlusion_info, quality_score
    except Exception as e:
        print(f"Preprocessing pipeline failed: {e}")
        # Return basic fallback
        basic_enhanced = enhance_image(image)
        dummy_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        dummy_info = {
            'upper_occluded': False,
            'lower_occluded': False,
            'upper_confidence': 0.0,
            'lower_confidence': 0.0,
            'total_occlusion_percentage': 0.0,
            'occlusion_level': 'minimal'
        }
        quality_score = 0.5
        return basic_enhanced, dummy_mask, dummy_info, quality_score


def adaptive_enhance(image, use_sr=False):
    """
    Try multiple enhancement strategies and return the original plus enhanced versions
    This allows the face detector to try multiple versions

    Args:
        image: Input RGB image
        use_sr: Whether to apply super-resolution

    Returns:
        List of enhanced images
    """
    try:
        results = [image.copy()]  # Always include the original image

        # Apply super-resolution if requested
        if use_sr:
            sr_image = apply_super_resolution(image)
            results.append(sr_image)

            # Also add enhanced version of SR image
            enhanced_sr = enhance_image(sr_image)
            results.append(enhanced_sr)

        # Basic enhancement
        basic = enhance_image(image)
        results.append(basic)

        # Get image brightness
        brightness = np.mean(image)

        # For dark images
        if brightness < 100:
            # Add stronger gamma correction
            dark_enhanced = apply_gamma(image, 2.0)
            dark_enhanced = apply_clahe(dark_enhanced)
            results.append(dark_enhanced)

        # For bright images
        elif brightness > 180:
            # Add gamma < 1 to reduce brightness
            bright_enhanced = apply_gamma(image, 0.8)
            bright_enhanced = apply_clahe(bright_enhanced)
            results.append(bright_enhanced)

        return results
    except Exception as e:
        print(f"Adaptive enhancement failed: {e}")
        return [image]  # Return original image as fallback


# Test function
def test_preprocessing():
    """Test the preprocessing pipeline"""
    try:
        # Create a test image
        test_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        print("Testing preprocessing pipeline...")
        enhanced_img, mask, info, quality = preprocess_image_with_occlusion_handling(test_img)

        print(f"Quality Score: {quality:.3f}")
        print(f"Occlusion Info: {info}")
        print(f"Enhanced image shape: {enhanced_img.shape}")
        print(f"Mask shape: {mask.shape}")
        print("✅ Preprocessing test completed successfully!")

    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")


if __name__ == "__main__":
    test_preprocessing()
