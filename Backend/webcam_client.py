import os
import sys
import cv2
import numpy as np
import requests
from datetime import datetime

# Ensure backend services are on the Python path
sys.path.append("Backend")  # Maintains existing project structure [6]

from services.detection import detect_face
from services.utils import preprocess_face

API_URL = "http://localhost:8000/recognize"
DATASET_DIR = "dataset"

os.makedirs(DATASET_DIR, exist_ok=True)  # Create dataset directory if missing [3]

def main():
    cap = cv2.VideoCapture(0)  # Open default webcam [1]
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    print("Press 'r' to recognize and save, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed")
            break

        cv2.imshow("Webcam – Press 'r' to recognize, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            # Encode frame as JPEG
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()

            # Send to recognition API
            try:
                response = requests.post(API_URL, files={
                    "file": ("frame.jpg", img_bytes, "image/jpeg")
                })  # Multipart upload [2]
                identity = response.json().get("identity", "Unknown")
            except Exception as e:
                print(f"Recognition API error: {e}")
                continue

            print(f"Recognized: {identity}")

            # Local face detection
            face_crop = detect_face(frame)
            if face_crop is None:
                print("No face detected locally. Skipping save.")
                continue

            # Register new identity if unknown
            if identity == "Unknown":
                ans = input("Unknown face. Register? (y/n): ").strip().lower()
                if ans == 'y':
                    identity = input("Enter new name: ").strip()
                else:
                    continue

            # Preprocess and normalize face image
            processed = preprocess_face(face_crop)
            norm = ((processed - processed.min()) /
                    (processed.max() - processed.min()) * 255).astype(np.uint8)

            # Prepare output path
            person_dir = os.path.join(DATASET_DIR, identity)
            os.makedirs(person_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{identity}_{timestamp}.jpg"
            filepath = os.path.join(person_dir, filename)

            # Save BGR image for OpenCV compatibility
            cv2.imwrite(filepath, cv2.cvtColor(norm, cv2.COLOR_RGB2BGR))
            print(f"Saved preprocessed image: {filepath}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
