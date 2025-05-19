# train_occlusion_detector.py

import os
import cv2
import numpy as np
from occlusion_detection import OcclusionDetector


def train_occlusion_detector():
    """Train the occlusion detector with labeled images"""
    # Paths to your labeled data
    # You'll need to create these directories and add appropriate images
    glasses_dir = "../training_data/glasses"
    no_glasses_dir = "../training_data/no_glasses"
    beard_dir = "../training_data/beard"
    no_beard_dir = "../training_data/no_beard"

    # Load images with glasses
    glasses_images = []
    for filename in os.listdir(glasses_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(glasses_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                glasses_images.append(img)

    # Load images without glasses
    no_glasses_images = []
    for filename in os.listdir(no_glasses_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(no_glasses_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                no_glasses_images.append(img)

    # Load images with beard
    beard_images = []
    for filename in os.listdir(beard_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(beard_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                beard_images.append(img)

    # Load images without beard
    no_beard_images = []
    for filename in os.listdir(no_beard_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(no_beard_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                no_beard_images.append(img)

    print(f"Loaded {len(glasses_images)} glasses images")
    print(f"Loaded {len(no_glasses_images)} no-glasses images")
    print(f"Loaded {len(beard_images)} beard images")
    print(f"Loaded {len(no_beard_images)} no-beard images")

    # Initialize and train detector
    detector = OcclusionDetector()
    detector.train(
        upper_occluded_images=glasses_images,
        upper_clean_images=no_glasses_images,
        lower_occluded_images=beard_images,
        lower_clean_images=no_beard_images,
        save_path="../assets/occlusion_models.pkl"
    )

    print("Occlusion detector trained successfully!")
    return detector


if __name__ == "__main__":
    train_occlusion_detector()
