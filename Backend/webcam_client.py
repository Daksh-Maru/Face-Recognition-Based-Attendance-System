import cv2
import requests

API_URL = "http://localhost:8000/recognize"  # Change if deployed elsewhere

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Show the webcam frame
    cv2.imshow("Face Recognition - Press 'r' to recognize", frame)

    key = cv2.waitKey(1)

    if key == ord('r'):  # Press 'r' to send frame to FastAPI
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(API_URL, files={
            "file": ("webcam.jpg", img_encoded.tobytes(), "image/jpeg")
        })

        try:
            result = response.json()
            identity = result.get("identity", "Unknown")
            print(f"✅ Recognized: {identity}")
        except Exception as e:
            print("❌ Error recognizing face:", e)

    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
