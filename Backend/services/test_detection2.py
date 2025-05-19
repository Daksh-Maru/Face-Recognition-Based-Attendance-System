# test_detection.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detection import detect_face
from recognition import predict_face_with_occlusion_handling

# Step 1: Load image
img_path = r"C:\Users\abhis\PycharmProjects\STPL_MAIN\Face-Recognition-Based-Attendance-System\Backend\training_data\beard\10.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"❌ Could not load image from: {img_path}")
    exit()

# Step 2: Convert BGR to RGB for YOLOv8-face
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 3: Detect face with occlusion handling
face, occlusion_mask = detect_face(img_rgb)

# Step 4: Show result
if face is not None:
    print("✅ Face detected! Processing...")

    # Recognize face with occlusion handling
    identity = predict_face_with_occlusion_handling(face, occlusion_mask)

    # Create masked face for visualization
    masked_face = face.copy()
    if occlusion_mask is not None:
        for c in range(3):
            masked_face[:, :, c] = masked_face[:, :, c] * (1 - occlusion_mask)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(face)
    plt.title("Detected Face")
    plt.axis('off')

    if occlusion_mask is not None:
        plt.subplot(132)
        plt.imshow(occlusion_mask, cmap='gray')
        plt.title("Occlusion Mask")
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(masked_face)
        plt.title(f"Non-occluded Regions\nIdentified as: {identity}")
        plt.axis('off')
    else:
        plt.subplot(132)
        plt.imshow(face)
        plt.title(f"Identified as: {identity}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("❌ No face detected.")
