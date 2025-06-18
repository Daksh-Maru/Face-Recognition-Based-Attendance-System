import os
import cv2
import numpy as np
import sys

# Include your backend path
sys.path.append("Backend")

from services.detection import detect_face
from services.utils import preprocess_face

DATASET_DIR = "dataset"  # Your actual dataset path

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face
    face = detect_face(img_rgb)
    if face is None:
        print(f"No face detected in: {image_path}")
        return None

    # Preprocess face
    face_processed = preprocess_face(face)

    # Normalize face for saving
    face_normalized = ((face_processed - face_processed.min()) / (face_processed.max() - face_processed.min()) * 255).astype(np.uint8)

    return face_normalized

def main():
    for person_name in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person_name)

        if not os.path.isdir(person_folder):
            continue

        print(f"Processing {person_name}...")

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            processed_img = process_image(img_path)

            if processed_img is not None:
                # Overwrite the original file
                cv2.imwrite(img_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                print(f"Replaced: {img_path}")

    print("\nAll dataset images processed and replaced successfully.")

if __name__ == "__main__":
    main()