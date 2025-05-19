import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def apply_gamma(image, gamma=1.0):
    """
    Apply gamma correction to improve brightness/contrast

    Args:
        image: Input RGB image
        gamma: Gamma value (< 1: lighter, > 1: darker)

    Returns:
        Gamma-corrected image
    """
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def test_gamma_correction(image_path):
    """
    Test gamma correction with various gamma values

    Args:
        image_path: Path to the input image
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert from BGR to RGB for proper display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a figure to display multiple gamma values
    plt.figure(figsize=(15, 12))

    # Test different gamma values
    gamma_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.2]

    # Display original image
    plt.subplot(3, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # Apply and display different gamma corrections
    for i, gamma in enumerate(gamma_values):
        # Skip gamma=1.0 as it doesn't change the image
        if gamma == 1.0:
            continue

        # Apply gamma correction
        gamma_image = apply_gamma(image_rgb, gamma)

        # Display the result
        plt.subplot(3, 3, i + 2)
        plt.imshow(gamma_image)
        plt.title(f'Gamma = {gamma}')
        plt.axis('off')

        # Save individual gamma-corrected images
        cv2.imwrite(f"gamma_{gamma:.1f}.jpg", cv2.cvtColor(gamma_image, cv2.COLOR_RGB2BGR))

    # Save the comparison figure
    plt.tight_layout()
    plt.savefig("gamma_comparison.png")
    plt.show()

    print("Gamma comparison saved to gamma_comparison.png")
    print("Individual gamma-corrected images saved as gamma_X.X.jpg")


# Use the provided image path
image_path = r"C:\Users\abhis\PycharmProjects\STPL_MAIN\Face-Recognition-Based-Attendance-System\Backend\dataset\Sarthak\WIN_20250518_13_46_51_Pro.jpg"

# Run the test
try:
    test_gamma_correction(image_path)
except Exception as e:
    print(f"Error: {e}")
