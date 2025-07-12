import cv2
import numpy as np

def apply_hist_eq(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2RGB)

def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def apply_gamma(image: np.ndarray, gamma: float=1.5) -> np.ndarray:
    invGamma = 1.0/gamma
    table = np.array([((i/255.0)**invGamma)*255 for i in range(256)], dtype="uint8")
    return cv2.LUT(image, table)

def apply_denoise(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def apply_sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)

def enhance_image(image: np.ndarray) -> np.ndarray:
    brightness = np.mean(image)
    if brightness < 80:
        image = apply_gamma(image, 2.5)
        image = apply_clahe(image)
        image = apply_hist_eq(image)
        image = apply_sharpen(image)
        image = apply_denoise(image)
    elif brightness > 200:
        image = apply_gamma(image, 0.8)
        image = apply_clahe(image)
        image = apply_denoise(image)
    else:
        image = apply_gamma(image, 1.5)
        image = apply_clahe(image)
        image = apply_sharpen(image)
        image = apply_denoise(image)
    return image
