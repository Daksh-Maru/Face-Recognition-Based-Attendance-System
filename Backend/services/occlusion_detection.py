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
            try:
                with open(model_path, 'rb') as f:
                    models = pickle.load(f)
                    self.pca = models['pca']
                    self.upper_classifier = models['upper_classifier']
                    self.lower_classifier = models['lower_classifier']
                print(f"✅ Loaded occlusion models from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading models from {model_path}: {e}")

    def extract_gabor_features(self, img_patch):
        """Extract Gabor features from an image patch"""
        try:
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
        except Exception as e:
            print(f"Error in extract_gabor_features: {e}")
            # Return zero features if extraction fails
            return np.zeros(32)  # 4*2*2*2 = 32 features

    def extract_multi_scale_gabor_features(self, img_patch, scales=[1.0]):  # Single scale only
        """Simplified Gabor features to match training"""
        try:
            if len(img_patch.shape) == 3:
                img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY)

            # Resize to fixed size
            img_patch = cv2.resize(img_patch, (64, 64))

            gabor_features = []
            # Reduced parameters to match training
            for theta in np.arange(0, np.pi, np.pi / 4):  # 4 orientations
                for sigma in [3, 5]:  # 2 scales
                    for lambd in [10, 15]:  # 2 wavelengths
                        try:
                            kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
                            filtered = cv2.filter2D(img_patch, cv2.CV_8UC3, kernel)

                            # Only basic features (2 per filter)
                            gabor_features.append(np.mean(filtered))
                            gabor_features.append(np.std(filtered))
                        except:
                            gabor_features.extend([0.0, 0.0])

            return np.array(gabor_features)  # Should be 4*2*2*2 = 32 features
        except Exception as e:
            print(f"Error in extract_multi_scale_gabor_features: {e}")
            return np.zeros(32)  # Return exactly 32 features

    def train(self, upper_occluded_images, upper_clean_images,
              lower_occluded_images, lower_clean_images, save_path="assets/occlusion_models.pkl"):
        """Train the occlusion detectors with labeled data"""
        try:
            print("🎯 Training occlusion detector...")

            # Process upper region (eyewear) data
            upper_features = []
            upper_labels = []

            print(f"Processing {len(upper_occluded_images)} occluded upper images...")
            # Extract features from occluded upper regions
            for img in upper_occluded_images:
                h, w = img.shape[:2]
                upper_patch = img[0:h // 2, :]  # Upper half of face
                features = self.extract_multi_scale_gabor_features(upper_patch)
                upper_features.append(features)
                upper_labels.append(1)  # 1 = occluded

            print(f"Processing {len(upper_clean_images)} clean upper images...")
            # Extract features from clean upper regions
            for img in upper_clean_images:
                h, w = img.shape[:2]
                upper_patch = img[0:h // 2, :]
                features = self.extract_multi_scale_gabor_features(upper_patch)
                upper_features.append(features)
                upper_labels.append(0)  # 0 = not occluded

            # Fit PCA on upper features
            upper_features_array = np.array(upper_features)
            print(f"Upper features shape: {upper_features_array.shape}")

            # Adjust PCA components based on available data
            n_samples, n_features = upper_features_array.shape
            n_components = min(50, n_samples - 1, n_features)
            self.pca = PCA(n_components=n_components)

            self.pca.fit(upper_features_array)
            upper_features_pca = self.pca.transform(upper_features_array)

            # Train SVM for upper region
            self.upper_classifier = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
            self.upper_classifier.fit(upper_features_pca, upper_labels)

            # Process lower region (beard/mask) data
            lower_features = []
            lower_labels = []

            print(f"Processing {len(lower_occluded_images)} occluded lower images...")
            # Extract features from occluded lower regions
            for img in lower_occluded_images:
                h, w = img.shape[:2]
                lower_patch = img[h // 2:, :]  # Lower half of face
                features = self.extract_multi_scale_gabor_features(lower_patch)
                lower_features.append(features)
                lower_labels.append(1)  # 1 = occluded

            print(f"Processing {len(lower_clean_images)} clean lower images...")
            # Extract features from clean lower regions
            for img in lower_clean_images:
                h, w = img.shape[:2]
                lower_patch = img[h // 2:, :]
                features = self.extract_multi_scale_gabor_features(lower_patch)
                lower_features.append(features)
                lower_labels.append(0)  # 0 = not occluded

            # Apply same PCA to lower features
            lower_features_array = np.array(lower_features)
            print(f"Lower features shape: {lower_features_array.shape}")
            lower_features_pca = self.pca.transform(lower_features_array)

            # Train SVM for lower region
            self.lower_classifier = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
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

            print(f"✅ Models saved to {save_path}")
            return True

        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False

    def detect_occlusions(self, face_img):
        """Detect occlusions in upper and lower face regions"""
        if self.upper_classifier is None or self.lower_classifier is None:
            raise ValueError("Classifiers not trained. Train or load models first.")

        try:
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
        except Exception as e:
            print(f"Error in detect_occlusions: {e}")
            return False, False

    def detect_occlusions_enhanced(self, face_img):
        """Enhanced occlusion detection with confidence scores"""
        if self.upper_classifier is None or self.lower_classifier is None:
            raise ValueError("Classifiers not trained. Train or load models first.")

        try:
            h, w = face_img.shape[:2]

            # Extract and process upper region with multi-scale features
            upper_patch = face_img[0:h // 2, :]
            upper_features = self.extract_multi_scale_gabor_features(upper_patch)
            upper_features = upper_features.reshape(1, -1)
            upper_features_pca = self.pca.transform(upper_features)

            # Get probability scores instead of just binary prediction
            upper_proba = self.upper_classifier.predict_proba(upper_features_pca)[0]
            upper_occluded = upper_proba[1] > 0.5  # Class 1 is occluded
            upper_confidence = upper_proba[1]

            # Extract and process lower region
            lower_patch = face_img[h // 2:, :]
            lower_features = self.extract_multi_scale_gabor_features(lower_patch)
            lower_features = lower_features.reshape(1, -1)
            lower_features_pca = self.pca.transform(lower_features)

            lower_proba = self.lower_classifier.predict_proba(lower_features_pca)[0]
            lower_occluded = lower_proba[1] > 0.5
            lower_confidence = lower_proba[1]

            return {
                'upper_occluded': upper_occluded,
                'lower_occluded': lower_occluded,
                'upper_confidence': upper_confidence,
                'lower_confidence': lower_confidence
            }
        except Exception as e:
            print(f"Error in detect_occlusions_enhanced: {e}")
            return {
                'upper_occluded': False,
                'lower_occluded': False,
                'upper_confidence': 0.0,
                'lower_confidence': 0.0
            }

    def create_occlusion_mask(self, face_img):
        """Create a binary occlusion mask (1=occluded, 0=non-occluded)"""
        try:
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
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            return mask
        except Exception as e:
            print(f"Error in create_occlusion_mask: {e}")
            h, w = face_img.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

    def create_hierarchical_occlusion_mask(self, face_img):
        """Create detailed occlusion mask with hierarchical regions"""
        try:
            h, w = face_img.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)

            # Define facial regions more precisely
            regions = {
                'forehead': (0, h // 4, 0, w),
                'eyes': (h // 4, h // 2, 0, w),
                'nose': (h // 3, 2 * h // 3, w // 3, 2 * w // 3),
                'mouth': (2 * h // 3, h, w // 4, 3 * w // 4),
                'left_cheek': (h // 3, 2 * h // 3, 0, w // 3),
                'right_cheek': (h // 3, 2 * h // 3, 2 * w // 3, w),
                'chin': (3 * h // 4, h, 0, w)
            }

            # Analyze each region separately
            for region_name, (y1, y2, x1, x2) in regions.items():
                # Ensure coordinates are within bounds
                y1, y2 = max(0, y1), min(h, y2)
                x1, x2 = max(0, x1), min(w, x2)

                if y2 <= y1 or x2 <= x1:
                    continue

                region_patch = face_img[y1:y2, x1:x2]

                if region_patch.size > 0:
                    # Extract features for this region
                    features = self.extract_multi_scale_gabor_features(region_patch)
                    if features.size == 0:
                        continue

                    features = features.reshape(1, -1)

                    # Use appropriate classifier based on region
                    if region_name in ['forehead', 'eyes']:
                        features_pca = self.pca.transform(features)
                        occlusion_prob = self.upper_classifier.predict_proba(features_pca)[0][1]
                    else:
                        features_pca = self.pca.transform(features)
                        occlusion_prob = self.lower_classifier.predict_proba(features_pca)[0][1]

                    # Fill mask with probability values
                    mask[y1:y2, x1:x2] = occlusion_prob

            # Apply adaptive thresholding
            adaptive_threshold = 0.5
            binary_mask = (mask > adaptive_threshold).astype(np.uint8)

            # Enhanced MRF-like refinement
            kernel_open = np.ones((3, 3), np.uint8)
            kernel_close = np.ones((7, 7), np.uint8)

            # Remove small noise
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
            # Fill small holes
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)

            # Apply Gaussian smoothing for soft boundaries
            binary_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (5, 5), 1.0)

            return binary_mask, mask  # Return both binary and probability masks
        except Exception as e:
            print(f"Error in create_hierarchical_occlusion_mask: {e}")
            h, w = face_img.shape[:2]
            dummy_mask = np.zeros((h, w), dtype=np.float32)
            return dummy_mask, dummy_mask

    def extract_selective_lgbphs(self, face_img, occlusion_mask):
        """Extract LGBPHS features only from non-occluded regions"""
        try:
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
        except Exception as e:
            print(f"Error in extract_selective_lgbphs: {e}")
            return np.zeros(2048)  # 4*2*256 = 2048 features


# Standalone function for testing
def test_occlusion_detector():
    """Test the occlusion detector with synthetic data"""
    try:
        print("🧪 Testing OcclusionDetector...")

        # Create synthetic test images
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Initialize detector
        detector = OcclusionDetector()

        # Test feature extraction
        features = detector.extract_gabor_features(test_img)
        print(f"✅ Gabor features shape: {features.shape}")

        # Test multi-scale features
        ms_features = detector.extract_multi_scale_gabor_features(test_img)
        print(f"✅ Multi-scale features shape: {ms_features.shape}")

        # Test occlusion mask creation (without trained models)
        try:
            mask = detector.create_occlusion_mask(test_img)
            print(f"⚠️ Mask creation should fail without trained models")
        except ValueError as e:
            print(f"✅ Expected error: {e}")

        print("🎉 OcclusionDetector test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_occlusion_detector()
