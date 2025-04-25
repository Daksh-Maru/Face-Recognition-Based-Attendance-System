import cv2
import requests
import os
import time
#import generate_embeddings  # ← Make sure it's importable

API_URL = "http://localhost:8000/recognize"
DATASET_PATH = "dataset"
EMBEDDING_PATH = "assets/embeddings.pkl"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

print("🎥 Press 'r' to recognize. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)

    if key == ord('r'):
        # Encode and send to API
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={
            "file": ("webcam.jpg", img_encoded.tobytes(), "image/jpeg")
        })

        try:
            result = response.json()
            identity = result.get("identity", "Unknown")
            print(f"🔎 Recognized: {identity}")

            timestamp = int(time.time())

            if identity == "Unknown":
                name = input("📝 Enter name of new person: ").strip()
                folder = os.path.join(DATASET_PATH, name)
                os.makedirs(folder, exist_ok=True)

                save_path = os.path.join(folder, f"{name}_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"📁 Saved new face to: {save_path}")

            else:
                folder = os.path.join(DATASET_PATH, identity)
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(folder, f"{identity}_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"📸 Added image for {identity} at: {save_path}")

            # 🔁 Regenerate embeddings
            #print("🧠 Updating embeddings...")
            #generate_embeddings.save_embeddings(DATASET_PATH, EMBEDDING_PATH)
            #print("✅ Embeddings updated successfully.")

        except Exception as e:
            print("❌ Recognition failed:", e)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
