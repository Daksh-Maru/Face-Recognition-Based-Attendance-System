# test_detection2.py

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from detection import detect_face  # Import your actual function
from recognition import predict_face_with_occlusion_handling  # Import your actual function


def test_enhanced_detection():
    # Use a relative path that works in your project structure
    img_path = r"C:\Users\abhis\PycharmProjects\STPL_MAIN\Face-Recognition-Based-Attendance-System\Backend\training_data\glasses\2017-11-30-18-23-12.jpg"

    # Alternative: use an absolute path that exists in your system
    # img_path = "/path/to/your/actual/image.jpg"

    # Check if file exists before trying to load
    if not os.path.exists(img_path):
        print(f"❌ Image file not found: {img_path}")
        print("Please check the file path and ensure the image exists.")
        return

    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Could not load image from: {img_path}")
        print("The file exists but cannot be read. Check if it's a valid image file.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use enhanced detection
    try:
        face, occlusion_mask, occlusion_info = detect_face(img_rgb)
    except Exception as e:
        print(f"❌ Error in enhanced detection: {e}")
        return

    if face is not None:
        print("✅ Face detected! Enhanced processing...")

        # Debug information
        print(f"Face shape: {face.shape}")
        print(f"Mask type: {type(occlusion_mask)}")
        if occlusion_mask is not None:
            print(f"Mask shape: {occlusion_mask.shape}")
            print(f"Mask range: {np.min(occlusion_mask):.3f} - {np.max(occlusion_mask):.3f}")

        # Get detailed occlusion analysis
        if isinstance(occlusion_info, dict):
            print(
                f"Upper region occluded: {occlusion_info['upper_occluded']} (confidence: {occlusion_info['upper_confidence']:.2f})")
            print(
                f"Lower region occluded: {occlusion_info['lower_occluded']} (confidence: {occlusion_info['lower_confidence']:.2f})")
            print(f"Total occlusion: {occlusion_info['total_occlusion_percentage']:.1%}")
            print(f"Occlusion level: {occlusion_info['occlusion_level']}")
        else:
            print(f"Occlusion info type: {type(occlusion_info)}")
            print(f"Occlusion info: {occlusion_info}")

        # Recognize with enhanced handling
        try:
            identity = predict_face_with_occlusion_handling(face, occlusion_mask)
            print(f"🎯 Recognition result: {identity}")
        except Exception as e:
            print(f"⚠️ Recognition error: {e}")
            identity = "Recognition Failed"

        # Enhanced visualization
        plt.figure(figsize=(20, 5))

        plt.subplot(141)
        plt.imshow(face)
        plt.title("Detected Face")
        plt.axis('off')

        plt.subplot(142)
        # Handle different mask formats
        if isinstance(occlusion_mask, np.ndarray) and occlusion_mask.size > 0:
            # Ensure mask is 2D
            if len(occlusion_mask.shape) == 3:
                occlusion_mask_2d = np.mean(occlusion_mask, axis=2)
            else:
                occlusion_mask_2d = occlusion_mask

            # Resize mask to match face if needed
            if occlusion_mask_2d.shape != face.shape[:2]:
                occlusion_mask_2d = cv2.resize(occlusion_mask_2d, (face.shape[1], face.shape[0]))

            plt.imshow(occlusion_mask_2d, cmap='Reds', vmin=0, vmax=1)
            plt.colorbar(label='Occlusion Probability')
        else:
            # Create informative dummy mask
            dummy_mask = np.zeros((face.shape[0], face.shape[1]))
            plt.imshow(dummy_mask, cmap='Reds', vmin=0, vmax=1)
            plt.text(0.5, 0.5, 'No Occlusion\nDetected',
                     transform=plt.gca().transAxes, ha='center', va='center',
                     fontsize=12, color='red', weight='bold')
        plt.title("Occlusion Probability Map")
        plt.axis('off')

        plt.subplot(143)
        # Show non-occluded regions
        if isinstance(occlusion_mask, np.ndarray) and occlusion_mask.size > 0:
            # Ensure mask is 2D and same size as face
            if len(occlusion_mask.shape) == 3:
                binary_mask = np.mean(occlusion_mask, axis=2)
            else:
                binary_mask = occlusion_mask.copy()

            # Resize if needed
            if binary_mask.shape != face.shape[:2]:
                binary_mask = cv2.resize(binary_mask, (face.shape[1], face.shape[0]))

            # Create binary mask
            binary_mask = (binary_mask > 0.5).astype(np.uint8)

            # Apply mask to face
            masked_face = face.copy()
            for c in range(3):
                masked_face[:, :, c] = masked_face[:, :, c] * (1 - binary_mask)
            plt.imshow(masked_face)
        else:
            plt.imshow(face)  # Show original if no mask
        plt.title("Non-occluded Regions")
        plt.axis('off')

        plt.subplot(144)
        # Show confidence visualization
        confidence_viz = np.zeros_like(face)
        if isinstance(occlusion_info, dict):
            h, w = face.shape[:2]
            # Upper region confidence (red channel)
            confidence_viz[:h // 2, :, 0] = int(occlusion_info['upper_confidence'] * 255)
            # Lower region confidence (green channel)
            confidence_viz[h // 2:, :, 1] = int(occlusion_info['lower_confidence'] * 255)
        plt.imshow(confidence_viz)
        plt.title(
            f"Confidence Map\nUpper: {occlusion_info.get('upper_confidence', 0):.2f} | Lower: {occlusion_info.get('lower_confidence', 0):.2f}\nIdentified as: {identity}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Save the visualization
        plt.savefig('heuristic_occlusion_result.png', dpi=150, bbox_inches='tight')
        print("📊 Visualization saved as 'heuristic_occlusion_result.png'")

    else:
        print("❌ No face detected or quality too low.")


def test_with_sample_images():
    """Test with multiple sample images"""
    sample_paths = [
        r"C:\Users\abhis\PycharmProjects\STPL_MAIN\Face-Recognition-Based-Attendance-System\Backend\training_data\beard\10.jpg",
        r"C:\Users\abhis\PycharmProjects\STPL_MAIN\Face-Recognition-Based-Attendance-System\Backend\training_data\no_beard\1.jpg",
        r"C:\Users\abhis\PycharmProjects\STPL_MAIN\Face-Recognition-Based-Attendance-System\Backend\training_data\glasses\1.jpg",
        r"C:\Users\abhis\PycharmProjects\STPL_MAIN\Face-Recognition-Based-Attendance-System\Backend\training_data\no_glasses\1.jpg"
    ]

    for img_path in sample_paths:
        if os.path.exists(img_path):
            print(f"\n🔍 Testing with: {os.path.basename(img_path)}")
            test_enhanced_detection_single(img_path)
        else:
            print(f"⚠️ Skipping {os.path.basename(img_path)} - file not found")


def test_enhanced_detection_single(img_path):
    """Test enhanced detection on a single image"""
    img = cv2.imread(img_path)
    if img is None:
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        face, occlusion_mask, occlusion_info = detect_face(img_rgb)
        if face is not None and isinstance(occlusion_info, dict):
            print(
                f"  ✅ Face detected - Occlusion: {occlusion_info['occlusion_level']} ({occlusion_info['total_occlusion_percentage']:.1%})")
            print(f"     Upper: {occlusion_info['upper_occluded']} ({occlusion_info['upper_confidence']:.2f})")
            print(f"     Lower: {occlusion_info['lower_occluded']} ({occlusion_info['lower_confidence']:.2f})")
        else:
            print(f"  ❌ No face detected")
    except Exception as e:
        print(f"  ⚠️ Error: {e}")


def test_heuristic_functionality():
    """Test the heuristic occlusion detection with synthetic data"""
    print("\n🧪 Testing Heuristic Functionality")
    print("-" * 40)

    # Test 1: Normal face (should show minimal occlusion)
    normal_face = np.ones((200, 200, 3), dtype=np.uint8) * 120  # Gray face
    try:
        face, mask, info = detect_face(normal_face)
        if info:
            print(f"Normal face - Occlusion level: {info.get('occlusion_level', 'Unknown')}")
    except:
        print("Normal face test failed")

    # Test 2: Face with dark upper region (simulating glasses)
    glasses_face = np.ones((200, 200, 3), dtype=np.uint8) * 120
    glasses_face[:100, :] = 30  # Dark upper region
    try:
        face, mask, info = detect_face(glasses_face)
        if info:
            print(f"Glasses simulation - Upper occluded: {info.get('upper_occluded', False)}")
    except:
        print("Glasses simulation test failed")

    # Test 3: Face with textured lower region (simulating beard)
    beard_face = np.ones((200, 200, 3), dtype=np.uint8) * 120
    # Add random texture to lower region
    beard_face[100:, :] = np.random.randint(40, 80, (100, 200, 3))
    try:
        face, mask, info = detect_face(beard_face)
        if info:
            print(f"Beard simulation - Lower occluded: {info.get('lower_occluded', False)}")
    except:
        print("Beard simulation test failed")


if __name__ == "__main__":
    print("🚀 Starting Enhanced Face Detection Test")
    print("=" * 50)

    # Test single image
    test_enhanced_detection()

    # Test heuristic functionality
    test_heuristic_functionality()

    # Uncomment to test multiple images
    print("\n" + "=" * 50)
    print("Testing multiple sample images...")
    test_with_sample_images()
