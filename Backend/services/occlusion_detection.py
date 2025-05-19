# occlusion_detection.py

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import os
import pickle


class OcclusionDetector:
    def __init__(self, model_path=None):
        self.pca = PCA(n_components=30)
        self.upper_classifier = None  # For eyes/glasses region
        self.lower_classifier = None  # For mouth/beard region

        # Load pre-trained models if available
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models = pickle.load(f)
                self.pca = models['pca']
                self.upper_classifier = models['upper_classifier']
                self.lower_classifier = models['lower_classifier']

    def extract_gabor_features(self, img_patch):
        """Extract Gabor features from an image patch"""
        # Convert to grayscale if needed
        if len(img_patch.shape) == 3:
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY)

        # Resize for consistency
        img_patch = cv2.resize(img_patch, (64, 64))

        # Create Gabor kernels at different scales and orientations
        gabor_features = []
        for theta in np.arange(0, np.pi, np.pi / 4):  # 4 orientations
            for sigma in [3, 5]:  # 2 scales
                for lambd in [10, 15]:  # 2 wavelengths
                    kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(img_patch, cv2.CV_8UC3, kernel)
                    # Extract statistical features from filtered image
                    gabor_features.append(np.mean(filtered))
                    gabor_features.append(np.std(filtered))

        return np.array(gabor_features)

    def train(self, upper_occluded_images, upper_clean_images,
              lower_occluded_images, lower_clean_images, save_path="assets/occlusion_models.pkl"):
        """Train the occlusion detectors with labeled data"""
        # Process upper region (eyewear) data
        upper_features = []
        upper_labels = []

        # Extract features from occluded upper regions
        for img in upper_occluded_images:
            h, w = img.shape[:2]
            upper_patch = img[0:h // 2, :]  # Upper half of face
            features = self.extract_gabor_features(upper_patch)
            upper_features.append(features)
            upper_labels.append(1)  # 1 = occluded

        # Extract features from clean upper regions
        for img in upper_clean_images:
            h, w = img.shape[:2]
            upper_patch = img[0:h // 2, :]
            features = self.extract_gabor_features(upper_patch)
            upper_features.append(features)
            upper_labels.append(0)  # 0 = not occluded

        # Fit PCA on upper features
        upper_features_array = np.array(upper_features)
        self.pca.fit(upper_features_array)
        upper_features_pca = self.pca.transform(upper_features_array)

        # Train SVM for upper region
        self.upper_classifier = SVC(kernel='rbf', probability=True)
        self.upper_classifier.fit(upper_features_pca, upper_labels)

        # Process lower region (beard/mask) data
        lower_features = []
        lower_labels = []

        # Extract features from occluded lower regions
        for img in lower_occluded_images:
            h, w = img.shape[:2]
            lower_patch = img[h // 2:, :]  # Lower half of face
            features = self.extract_gabor_features(lower_patch)
            lower_features.append(features)
            lower_labels.append(1)  # 1 = occluded

        # Extract features from clean lower regions
        for img in lower_clean_images:
            h, w = img.shape[:2]
            lower_patch = img[h // 2:, :]
            features = self.extract_gabor_features(lower_patch)
            lower_features.append(features)
            lower_labels.append(0)  # 0 = not occluded

        # Apply same PCA to lower features
        lower_features_array = np.array(lower_features)
        lower_features_pca = self.pca.transform(lower_features_array)

        # Train SVM for lower region
        self.lower_classifier = SVC(kernel='rbf', probability=True)
        self.lower_classifier.fit(lower_features_pca, lower_labels)

        # Save the trained models
        models = {
            'pca': self.pca,
            'upper_classifier': self.upper_classifier,
            'lower_classifier': self.lower_classifier
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(models, f)

        return True

    def detect_occlusions(self, face_img):
        """Detect occlusions in upper and lower face regions"""
        if self.upper_classifier is None or self.lower_classifier is None:
            raise ValueError("Classifiers not trained. Train or load models first.")

        h, w = face_img.shape[:2]

        # Extract and process upper region
        upper_patch = face_img[0:h // 2, :]
        upper_features = self.extract_gabor_features(upper_patch)
        upper_features = upper_features.reshape(1, -1)
        upper_features_pca = self.pca.transform(upper_features)
        upper_occluded = self.upper_classifier.predict(upper_features_pca)[0] == 1

        # Extract and process lower region
        lower_patch = face_img[h // 2:, :]
        lower_features = self.extract_gabor_features(lower_patch)
        lower_features = lower_features.reshape(1, -1)
        lower_features_pca = self.pca.transform(lower_features)
        lower_occluded = self.lower_classifier.predict(lower_features_pca)[0] == 1

        return upper_occluded, lower_occluded

    def create_occlusion_mask(self, face_img):
        """Create a binary occlusion mask (1=occluded, 0=non-occluded)"""
        h, w = face_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        upper_occluded, lower_occluded = self.detect_occlusions(face_img)

        # Mark upper region if occluded
        if upper_occluded:
            mask[0:h // 2, :] = 1

        # Mark lower region if occluded
        if lower_occluded:
            mask[h // 2:, :] = 1

        # Apply MRF-based refinement (simplified version)
        # In a full implementation, you would use a proper MRF library
        # This is a simplified approach using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask


# Function to extract LGBPHS features from non-occluded regions
def extract_selective_lgbphs(face_img, occlusion_mask):
    """Extract LGBPHS features only from non-occluded regions"""
    # Convert to grayscale
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_img

    # Apply Gabor filters
    gabor_features = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        for sigma in [3, 5]:
            kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)

            # Apply LBP to filtered image (simplified)
            lbp = np.zeros_like(filtered)
            for i in range(1, filtered.shape[0] - 1):
                for j in range(1, filtered.shape[1] - 1):
                    if occlusion_mask[i, j] == 0:  # Only process non-occluded pixels
                        center = filtered[i, j]
                        code = 0
                        if filtered[i - 1, j - 1] > center: code += 1
                        if filtered[i - 1, j] > center: code += 2
                        if filtered[i - 1, j + 1] > center: code += 4
                        if filtered[i, j + 1] > center: code += 8
                        if filtered[i + 1, j + 1] > center: code += 16
                        if filtered[i + 1, j] > center: code += 32
                        if filtered[i + 1, j - 1] > center: code += 64
                        if filtered[i, j - 1] > center: code += 128
                        lbp[i, j] = code

            # Compute histogram for this filtered+LBP image
            # Only include non-occluded regions in histogram
            non_occluded_lbp = lbp[occlusion_mask == 0]
            if len(non_occluded_lbp) > 0:  # Only if we have non-occluded pixels
                hist, _ = np.histogram(non_occluded_lbp, bins=256, range=(0, 256))
                hist = hist.astype(float) / len(non_occluded_lbp)  # Normalize
                gabor_features.extend(hist)

    return np.array(gabor_features)
