import cv2
import numpy as np

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
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def apply_sharpen(image):
    """
    Apply sharpening to enhance facial features
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)



def enhance_image(image):
    """
    Adaptive image enhancement based on image conditions
    """
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
