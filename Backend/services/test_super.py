import cv2
import matplotlib.pyplot as plt
from super_resolution import SuperResolution


def test_super_resolution():
    # Load a test image
    img_path = "../dataset/Sarthak/WIN_20250518_13_46_51_Pro.jpg"
    img = cv2.imread(img_path)

    # Initialize super-resolution model
    sr = SuperResolution(model_name="fsrcnn", scale=2)

    # Apply super-resolution
    result = sr.upsample(img)

    # Create standard upscaled version for comparison
    resized = cv2.resize(img, (result.shape[1], result.shape[0]))

    # Display results
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title("Super-Resolution")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 3)
    plt.title("Standard Resize")
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.savefig("super_resolution_comparison.png")
    plt.show()


if __name__ == "__main__":
    test_super_resolution()
