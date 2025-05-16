import os
import cv2
from detection import detect_face

# Step 1: Load image
base_dir = os.path.dirname(__file__)
img_path = os.path.abspath(os.path.join(base_dir, "../dataset/Aaron_Guiel/Aaron_Guiel_0001.jpg"))

img = cv2.imread(img_path)

if img is None:
    print(f"❌ Could not load image from: {img_path}")
    exit()

# Step 2: Convert BGR to RGB for YOLOv8-face
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 3: Detect face
face = detect_face(img_rgb)

# Step 4: Show result
if face is not None:
    print("✅ Face detected! Showing cropped face.")
    cv2.imshow("Detected Face", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ No face detected.")
