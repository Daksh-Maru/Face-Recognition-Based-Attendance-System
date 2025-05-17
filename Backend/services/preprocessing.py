import cv2
import numpy as np
import os


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
            "espcn": f"ESPCN_x{scale}.pb"
        }  #

        self.model_path = os.path.join(models_dir, model_files[model_name])

        # Check if model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SR model not found at {self.model_path}")

        # Load the model
        self.sr.readModel(self.model_path)
        self.sr.setModel(model_name, scale)

    def upsample(self, image):
        """
        Apply super-resolution to an image

        Args:
            image: Input image (grayscale or single channel)

        Returns:
            Super-resolved image
        """
        return self.sr.upsample(image)


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
    if sr_model is None:
        return image

    # Process based on image type
    if len(image.shape) == 3 and image.shape[2] == 3:  # Fixed condition check
        # Split channels and process individually
        b, g, r = cv2.split(image)
        b_sr = sr_model.upsample(b)
        g_sr = sr_model.upsample(g)
        r_sr = sr_model.upsample(r)
        # Merge channels
        return cv2.merge([b_sr, g_sr, r_sr])
    else:
        # Grayscale image
        return sr_model.upsample(image)


def apply_hist_eq(image):
    """
    Apply histogram equalization to improve contrast
    """
    # Convert to YCrCb and equalize only Y channel to preserve color
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


def apply_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_denoise(image):
    # Reduced parameters for faster processing while still effective
    return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)


def apply_sharpen(image):
    """
    Apply sharpening to enhance facial features
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def normalize_brightness(image):
    # Normalize brightness to mid-range
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


def enhance_image(image, use_sr=False):  # Added use_sr parameter
    """
    Adaptive image enhancement based on image conditions

    Args:
        image: Input RGB image
        use_sr: Whether to apply super-resolution

    Returns:
        Enhanced image
    """
    # Apply super-resolution first if requested
    if use_sr:
        try:
            image = apply_super_resolution(image)
        except Exception as e:
            print(f"Super-resolution failed: {e}")

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
    results = [image.copy()]  # Always include the original image

    # Apply super-resolution if requested
    if use_sr:
        try:
            sr_image = apply_super_resolution(image)
            results.append(sr_image)

            # Also add enhanced version of SR image
            enhanced_sr = enhance_image(sr_image)
            results.append(enhanced_sr)
        except Exception as e:
            print(f"Super-resolution failed: {e}")

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
