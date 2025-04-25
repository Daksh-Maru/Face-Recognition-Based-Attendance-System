import cv2
import numpy as np
import requests
import os
from datetime import datetime
from io import BytesIO
import sys

# ✅ Include backend path
sys.path.append("Backend")  # Update path to match your structure

# ✅ Import your own detection and preprocessing functions
from services.detection import detect_face
from services.utils import preprocess_face

API_URL = "http://localhost:8000/recognize"
DATASET_DIR = "dataset"

os.makedirs(DATASET_DIR, exist_ok=True)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Press 'r' to recognize, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        # Encode full frame and send to API
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        response = requests.post(API_URL, files={
            "file": ("frame.jpg", img_bytes, "image/jpeg")
        })

        try:
            identity = response.json().get("identity", "Unknown")
            print(f"🧠 Recognized: {identity}")

            # Detect face locally for saving
            face = detect_face(frame)

            if face is None:
                print("⚠️ No face detected locally. Skipping save.")
                continue

            # Ask for new employee if unknown
            if identity == "Unknown":
                register = input("❓ Unknown face detected. Register this person? (y/n): ").strip().lower()
                if register == 'y':
                    identity = input("👤 Enter the new employee name: ").strip()
                else:
                    print("⛔ Skipping image save.")
                    continue

            # Preprocess face using your logic
            processed_np = preprocess_face(face)

            # Normalize float image to 0-255 for saving
            normalized_img = ((processed_np - processed_np.min()) / (processed_np.max() - processed_np.min()) * 255).astype(np.uint8)

            # Create folder if doesn't exist
            person_folder = os.path.join(DATASET_DIR, identity)
            os.makedirs(person_folder, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{identity}_{timestamp}.jpg"
            filepath = os.path.join(person_folder, filename)

            # Save image (convert RGB to BGR for OpenCV)
            cv2.imwrite(filepath, cv2.cvtColor(normalized_img, cv2.COLOR_RGB2BGR))
            print(f"✅ Saved preprocessed image to: {filepath}")

        except Exception as e:
            print("❌ Error:", e)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
