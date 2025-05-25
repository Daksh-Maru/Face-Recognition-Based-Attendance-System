# train_occlusion_detector.py
# In train_occlusion_detector.py - Add this at the top:
"""
DEPRECATED: Training is no longer required.
The system now uses heuristic-based occlusion detection.

This file is kept for reference but is not used in the current system.
"""
print("⚠️ Training is no longer required. Using heuristic approach.")
exit()

import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from occlusion_detection import OcclusionDetector

# Define directories for training data (update these paths to your actual directories)
glasses_dir = '../training_data/glasses'
no_glasses_dir = '../training_data/no_glasses'
beard_dir = '../training_data/beard'
no_beard_dir = '../training_data/no_beard'


def load_images_from_directory(directory):
    """Load images from a directory"""
    images = []

    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist!")
        return images

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_formats):
            filepath = os.path.join(directory, filename)
            img = cv2.imread(filepath)
            if img is not None:
                # Convert to RGB for consistency
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
            else:
                print(f"Warning: Could not load image {filepath}")

    print(f"Loaded {len(images)} images from {directory}")
    return images


def augment_training_data(images, augmentation_factor=3):
    """Augment training data with variations"""
    if len(images) == 0:
        print("Warning: No images to augment!")
        return []

    augmented_images = []

    for img in images[:100]:
        augmented_images.append(img)  # Original

        for _ in range(augmentation_factor):
            try:
                # Random brightness
                brightness = np.random.uniform(0.7, 1.3)
                bright_img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)

                # Random rotation
                angle = np.random.uniform(-10, 10)
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_img = cv2.warpAffine(img, M, (w, h))

                # Random noise
                noise = np.random.normal(0, 5, img.shape).astype(np.int16)
                noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                # Random horizontal flip
                flipped_img = cv2.flip(img, 1)

                augmented_images.extend([bright_img, rotated_img, noisy_img, flipped_img])

            except Exception as e:
                print(f"Warning: Augmentation failed for an image: {e}")
                continue

    return augmented_images


def validate_model(detector, test_occluded, test_clean):
    """Validate the trained model"""
    if len(test_occluded) == 0 and len(test_clean) == 0:
        print("Warning: No test data available for validation!")
        return

    correct = 0
    total = len(test_occluded) + len(test_clean)

    # Test occluded images (should detect occlusion)
    for img in test_occluded:
        try:
            result = detector.detect_occlusions_enhanced(img)
            if result['upper_occluded']:
                correct += 1
        except Exception as e:
            print(f"Error during validation on occluded image: {e}")

    # Test clean images (should not detect occlusion)
    for img in test_clean:
        try:
            result = detector.detect_occlusions_enhanced(img)
            if not result['upper_occluded']:
                correct += 1
        except Exception as e:
            print(f"Error during validation on clean image: {e}")

    accuracy = correct / total if total > 0 else 0
    print(f"Validation Accuracy: {accuracy:.2f} ({correct}/{total})")

    return accuracy


def train_occlusion_detector_enhanced():
    """Enhanced training with data augmentation and better validation"""

    print("🚀 Starting Enhanced Occlusion Detector Training...")
    print("=" * 60)

    # Check if directories exist
    directories = [glasses_dir, no_glasses_dir, beard_dir, no_beard_dir]
    for directory in directories:
        if not os.path.exists(directory):
            print(f"❌ Error: Directory {directory} does not exist!")
            print("Please create the directory and add training images.")
            return None

    # Load training data
    print("📂 Loading training data...")
    glasses_images = load_images_from_directory(glasses_dir)
    no_glasses_images = load_images_from_directory(no_glasses_dir)
    beard_images = load_images_from_directory(beard_dir)
    no_beard_images = load_images_from_directory(no_beard_dir)

    # Check if we have enough data
    min_images_required = 5
    if (len(glasses_images) < min_images_required or
            len(no_glasses_images) < min_images_required or
            len(beard_images) < min_images_required or
            len(no_beard_images) < min_images_required):
        print(f"❌ Error: Need at least {min_images_required} images in each category!")
        print(f"Current counts - Glasses: {len(glasses_images)}, No-glasses: {len(no_glasses_images)}")
        print(f"Beard: {len(beard_images)}, No-beard: {len(no_beard_images)}")
        return None

    # Augment training data
    print("🔄 Augmenting training data...")
    glasses_images = augment_training_data(glasses_images)
    no_glasses_images = augment_training_data(no_glasses_images)
    beard_images = augment_training_data(beard_images)
    no_beard_images = augment_training_data(no_beard_images)

    print(f"📊 Augmented dataset sizes:")
    print(f"   Glasses: {len(glasses_images)}")
    print(f"   No-glasses: {len(no_glasses_images)}")
    print(f"   Beard: {len(beard_images)}")
    print(f"   No-beard: {len(no_beard_images)}")

    # Initialize detector with enhanced PCA
    print("🤖 Initializing enhanced occlusion detector...")
    detector = OcclusionDetector()

    # Set enhanced PCA with more components
    detector.pca = PCA(n_components=min(50, len(glasses_images) + len(no_glasses_images)))

    # Create assets directory if it doesn't exist
    assets_dir = "../assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        print(f"📁 Created assets directory: {assets_dir}")

    # Train with enhanced features
    print("🎯 Training the occlusion detector...")
    try:
        success = detector.train(
            upper_occluded_images=glasses_images,
            upper_clean_images=no_glasses_images,
            lower_occluded_images=beard_images,
            lower_clean_images=no_beard_images,
            save_path="../assets/enhanced_occlusion_models.pkl"
        )

        if success:
            print("✅ Enhanced occlusion detector trained successfully!")

            # Validate the model
            print("🔍 Validating the trained model...")
            test_size = min(10, len(glasses_images), len(no_glasses_images))
            accuracy = validate_model(
                detector,
                glasses_images[:test_size],
                no_glasses_images[:test_size]
            )

            if accuracy > 0.7:
                print(f"🎉 Great! Model achieved {accuracy:.1%} validation accuracy!")
            elif accuracy > 0.5:
                print(f"⚠️  Model achieved {accuracy:.1%} accuracy. Consider more training data.")
            else:
                print(f"❌ Low accuracy ({accuracy:.1%}). Please check your training data.")

        else:
            print("❌ Training failed!")
            return None

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

    return detector


def test_trained_model():
    """Test the trained model with sample images"""
    print("🧪 Testing the trained model...")

    model_path = "../assets/enhanced_occlusion_models.pkl"
    if not os.path.exists(model_path):
        print("❌ No trained model found. Please train the model first.")
        return

    try:
        detector = OcclusionDetector(model_path=model_path)

        # Test with a few sample images
        test_dirs = [glasses_dir, no_glasses_dir, beard_dir, no_beard_dir]

        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                images = load_images_from_directory(test_dir)
                if len(images) > 0:
                    test_img = images[0]  # Test first image
                    result = detector.detect_occlusions_enhanced(test_img)
                    print(f"📸 {os.path.basename(test_dir)}: Upper={result['upper_occluded']} "
                          f"(conf: {result['upper_confidence']:.2f}), "
                          f"Lower={result['lower_occluded']} (conf: {result['lower_confidence']:.2f})")

    except Exception as e:
        print(f"❌ Error testing model: {e}")


if __name__ == "__main__":
    print("🎯 Enhanced Occlusion Detector Training Script")
    print("=" * 60)

    # Train the enhanced model
    detector = train_occlusion_detector_enhanced()

    if detector is not None:
        print("\n" + "=" * 60)
        test_trained_model()
        print("\n🎉 Training and testing completed!")
    else:
        print("\n❌ Training failed. Please check your data and try again.")
