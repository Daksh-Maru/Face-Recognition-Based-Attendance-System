import cv2
from PIL import Image
import numpy as np
import requests
import os
from datetime import datetime
from io import BytesIO

API_URL = "http://localhost:8000/recognize"
DATASET_DIR = "dataset"  # Save processed images here

# Ensure dataset dir exists
os.makedirs(DATASET_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

def preprocess_for_facenet(frame):
    # Convert BGR (OpenCV) to RGB (FaceNet)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (160, 160))
    return Image.fromarray(resized), resized  # PIL image & NumPy array

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press 'r' to recognize, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        pil_img, img_np = preprocess_for_facenet(frame)

        # Send image to FastAPI
        img_byte_arr = BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        response = requests.post(API_URL, files={
            "file": ("frame.jpg", img_byte_arr, "image/jpeg")
        })

        try:
            identity = response.json().get("identity", "Unknown")
            print(f"✅ Recognized: {identity}")

            # Create subfolder if it doesn't exist
            person_folder = os.path.join(DATASET_DIR, identity)
            os.makedirs(person_folder, exist_ok=True)

            # Save processed image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{identity}_{timestamp}.jpg"
            filepath = os.path.join(person_folder, filename)

            # Save as RGB → BGR for OpenCV
            cv2.imwrite(filepath, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            print(f"📁 Image saved to: {filepath}")

        except Exception as e:
            print("❌ Error during recognition:", e)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
