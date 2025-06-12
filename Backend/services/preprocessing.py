import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Union
import time

# Setup logging
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Optimized image preprocessing for face recognition systems
    Compatible with LFW and standard datasets
    """

    def __init__(self, adaptive_mode: bool = True):
        """
        Initialize preprocessor with configurable parameters

        Args:
            adaptive_mode: Enable adaptive enhancement based on image quality
        """
        self.adaptive_mode = adaptive_mode

        # Optimized parameters for face recognition
        self.clahe_params = {
            'clipLimit': 1.5,  # Reduced from 2.0 for gentler enhancement
            'tileGridSize': (6, 6)  # Smaller tiles for better local adaptation
        }

        self.bilateral_params = {
            'd': 7,  # Reduced from 9 for better detail preservation
            'sigmaColor': 50,  # Reduced from 75
            'sigmaSpace': 50  # Reduced from 75
        }

        # Adaptive sharpening kernel (gentler than original)
        self.sharpen_kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ])

    def assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Assess image quality metrics for adaptive processing

        Args:
            image: Input RGB image

        Returns:
            Dictionary containing quality metrics
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Calculate quality metrics
            brightness = np.mean(gray)
            contrast = gray.std()
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Normalize metrics
            brightness_score = 1.0 - abs(brightness - 128) / 128
            contrast_score = min(1.0, contrast / 50.0)
            sharpness_score = min(1.0, sharpness / 100.0)

            return {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness,
                'brightness_score': brightness_score,
                'contrast_score': contrast_score,
                'sharpness_score': sharpness_score,
                'overall_quality': (brightness_score + contrast_score + sharpness_score) / 3.0
            }

        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return {'overall_quality': 0.5}

    def apply_hist_eq(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization with validation

        Args:
            image: Input RGB image

        Returns:
            Histogram equalized image
        """
        try:
            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.warning("Invalid image format for histogram equalization")
                return image

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            equalized = cv2.equalizeHist(gray)
            return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

        except Exception as e:
            logger.error(f"Error in histogram equalization: {e}")
            return image

    def apply_clahe(self, image: np.ndarray, adaptive: bool = True) -> np.ndarray:
        """
        Apply CLAHE with adaptive parameters

        Args:
            image: Input RGB image
            adaptive: Whether to use adaptive parameters

        Returns:
            CLAHE enhanced image
        """
        try:
            if len(image.shape) != 3 or image.shape[2] != 3:
                return image

            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Adaptive CLAHE parameters
            if adaptive and self.adaptive_mode:
                quality_metrics = self.assess_image_quality(image)
                # Adjust clip limit based on image contrast
                if quality_metrics['contrast'] < 20:
                    clip_limit = 2.5  # Higher for low contrast images
                elif quality_metrics['contrast'] > 60:
                    clip_limit = 1.0  # Lower for high contrast images
                else:
                    clip_limit = self.clahe_params['clipLimit']
            else:
                clip_limit = self.clahe_params['clipLimit']

            # Apply CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=self.clahe_params['tileGridSize']
            )
            cl = clahe.apply(l)

            # Merge channels and convert back
            enhanced_lab = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        except Exception as e:
            logger.error(f"Error in CLAHE enhancement: {e}")
            return image

    def apply_gamma(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """
        Apply gamma correction with optimized default value

        Args:
            image: Input RGB image
            gamma: Gamma correction value (optimized default: 1.2)

        Returns:
            Gamma corrected image
        """
        try:
            # Use more conservative gamma for face recognition
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(256)
            ]).astype("uint8")

            return cv2.LUT(image, table)

        except Exception as e:
            logger.error(f"Error in gamma correction: {e}")
            return image

    def apply_adaptive_gamma(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive gamma correction based on image brightness

        Args:
            image: Input RGB image

        Returns:
            Adaptively gamma corrected image
        """
        try:
            quality_metrics = self.assess_image_quality(image)
            brightness = quality_metrics['brightness']

            # Adaptive gamma selection
            if brightness < 80:  # Dark image
                gamma = 0.8  # Brighten
            elif brightness > 180:  # Bright image
                gamma = 1.4  # Darken slightly
            else:  # Normal brightness
                gamma = 1.0  # No correction needed

            return self.apply_gamma(image, gamma) if gamma != 1.0 else image

        except Exception as e:
            logger.error(f"Error in adaptive gamma correction: {e}")
            return image

    def apply_denoise(self, image: np.ndarray, adaptive: bool = True) -> np.ndarray:
        """
        Apply bilateral filtering with adaptive parameters

        Args:
            image: Input RGB image
            adaptive: Whether to use adaptive parameters

        Returns:
            Denoised image
        """
        try:
            if adaptive and self.adaptive_mode:
                quality_metrics = self.assess_image_quality(image)
                # Adjust denoising strength based on image quality
                if quality_metrics['sharpness'] < 30:  # Blurry image
                    d = 5  # Light denoising to preserve details
                    sigma_color = 30
                    sigma_space = 30
                else:  # Sharp image
                    d = self.bilateral_params['d']
                    sigma_color = self.bilateral_params['sigmaColor']
                    sigma_space = self.bilateral_params['sigmaSpace']
            else:
                d = self.bilateral_params['d']
                sigma_color = self.bilateral_params['sigmaColor']
                sigma_space = self.bilateral_params['sigmaSpace']

            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        except Exception as e:
            logger.error(f"Error in denoising: {e}")
            return image

    def apply_sharpen(self, image: np.ndarray, adaptive: bool = True) -> np.ndarray:
        """
        Apply sharpening with adaptive intensity

        Args:
            image: Input RGB image
            adaptive: Whether to use adaptive sharpening

        Returns:
            Sharpened image
        """
        try:
            if adaptive and self.adaptive_mode:
                quality_metrics = self.assess_image_quality(image)
                # Only sharpen if image is sufficiently blurry
                if quality_metrics['sharpness'] < 50:
                    kernel = self.sharpen_kernel
                else:
                    return image  # Skip sharpening for already sharp images
            else:
                kernel = self.sharpen_kernel

            sharpened = cv2.filter2D(image, -1, kernel)

            # Blend with original to avoid over-sharpening
            alpha = 0.7  # Reduced from full strength
            return cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0)

        except Exception as e:
            logger.error(f"Error in sharpening: {e}")
            return image

    def apply_unsharp_mask(self, image: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """
        Apply unsharp masking for better detail enhancement

        Args:
            image: Input RGB image
            amount: Sharpening amount (0.0 to 1.0)

        Returns:
            Unsharp masked image
        """
        try:
            # Create Gaussian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 1.0)

            # Create unsharp mask
            unsharp_mask = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

            return np.clip(unsharp_mask, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.error(f"Error in unsharp masking: {e}")
            return image

    def enhance_image(self, image: np.ndarray,
                      enhancement_level: str = "moderate") -> np.ndarray:
        """
        Main image enhancement pipeline optimized for face recognition

        Args:
            image: Input RGB image
            enhancement_level: "light", "moderate", or "aggressive"

        Returns:
            Enhanced image optimized for face recognition
        """
        try:
            if image is None or image.size == 0:
                return image

            # Validate input
            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.warning("Invalid image format for enhancement")
                return image

            # Ensure uint8 format
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)

            enhanced = image.copy()

            # Apply enhancement based on level
            if enhancement_level == "light":
                # Minimal processing for high-quality images
                if self.adaptive_mode:
                    enhanced = self.apply_adaptive_gamma(enhanced)
                else:
                    enhanced = self.apply_gamma(enhanced, 1.1)
                enhanced = self.apply_clahe(enhanced, adaptive=True)

            elif enhancement_level == "moderate":
                # Standard processing for most face recognition scenarios
                enhanced = self.apply_adaptive_gamma(enhanced)
                enhanced = self.apply_clahe(enhanced, adaptive=True)
                enhanced = self.apply_denoise(enhanced, adaptive=True)

            elif enhancement_level == "aggressive":
                # Heavy processing for challenging images
                enhanced = self.apply_gamma(enhanced, 1.3)
                enhanced = self.apply_clahe(enhanced, adaptive=False)
                enhanced = self.apply_denoise(enhanced, adaptive=False)
                enhanced = self.apply_sharpen(enhanced, adaptive=True)

            return enhanced

        except Exception as e:
            logger.error(f"Error in image enhancement: {e}")
            return image

    def preprocess_for_recognition(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Complete preprocessing pipeline for face recognition

        Args:
            image: Input face image

        Returns:
            Tuple of (processed_image, processing_info)
        """
        start_time = time.time()

        try:
            # Assess initial quality
            initial_quality = self.assess_image_quality(image)

            # Determine enhancement level based on quality
            if initial_quality['overall_quality'] > 0.7:
                enhancement_level = "light"
            elif initial_quality['overall_quality'] > 0.4:
                enhancement_level = "moderate"
            else:
                enhancement_level = "aggressive"

            # Apply enhancement
            processed_image = self.enhance_image(image, enhancement_level)

            # Assess final quality
            final_quality = self.assess_image_quality(processed_image)

            processing_info = {
                'enhancement_level': enhancement_level,
                'initial_quality': initial_quality,
                'final_quality': final_quality,
                'quality_improvement': final_quality['overall_quality'] - initial_quality['overall_quality'],
                'processing_time': time.time() - start_time
            }

            return processed_image, processing_info

        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            return image, {'error': str(e)}


# Global instance for backward compatibility
preprocessor = ImagePreprocessor(adaptive_mode=True)


# Backward compatibility functions
def apply_hist_eq(image):
    """Backward compatible histogram equalization"""
    return preprocessor.apply_hist_eq(image)


def apply_clahe(image):
    """Backward compatible CLAHE"""
    return preprocessor.apply_clahe(image)


def apply_gamma(image, gamma=1.2):  # Changed default from 1.5 to 1.2
    """Backward compatible gamma correction with optimized default"""
    return preprocessor.apply_gamma(image, gamma)


def apply_denoise(image):
    """Backward compatible denoising"""
    return preprocessor.apply_denoise(image)


def apply_sharpen(image):
    """Backward compatible sharpening"""
    return preprocessor.apply_sharpen(image)


def enhance_image(image):
    """Enhanced backward compatible function"""
    return preprocessor.enhance_image(image, "moderate")


# New optimized function for face recognition
def preprocess_face_image(image, adaptive=True):
    """
    Optimized preprocessing specifically for face recognition

    Args:
        image: Input face image
        adaptive: Whether to use adaptive enhancement

    Returns:
        Preprocessed image ready for recognition
    """
    global preprocessor
    if adaptive:
        preprocessor.adaptive_mode = True
        processed_image, _ = preprocessor.preprocess_for_recognition(image)
        return processed_image
    else:
        return preprocessor.enhance_image(image, "moderate")


if __name__ == "__main__":
    # Example usage and testing
    try:
        # Create test image
        test_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

        # Test preprocessing
        processor = ImagePreprocessor(adaptive_mode=True)
        enhanced_image, info = processor.preprocess_for_recognition(test_image)

        print("Preprocessing Test Results:")
        print(f"Enhancement level: {info['enhancement_level']}")
        print(f"Quality improvement: {info['quality_improvement']:.3f}")
        print(f"Processing time: {info['processing_time']:.3f}s")

    except Exception as e:
        print(f"Error in preprocessing test: {e}")
