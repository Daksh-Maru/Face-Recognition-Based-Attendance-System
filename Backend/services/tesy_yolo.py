import os
import cv2
from detection import detect_face

base_dir = os.path.dirname(__file__)
img_path = os.path.abspath(os.path.join(base_dir, "../dataset/Harshit_Singh/Harshit_Singh_1745601110.jpg"))

img = cv2.imread(img_path)

if img is None:
    print(f"Failed to load image from: {img_path}")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
face = detect_face(img_rgb)

if face is not None:
    print("Face detected, showing window...")
    cv2.imshow("Detected Face", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face detected in the image.")
