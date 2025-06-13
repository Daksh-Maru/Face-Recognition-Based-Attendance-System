import os
import cv2
import numpy as np
import pickle
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Define the correct path to the YOLOv8 face model
MODEL_PATH = os.path.join("..", "assets", "yolov8n-face.pt")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLOv8 face model not found at {MODEL_PATH}. Please ensure the model file exists.")

# Initialize YOLOv8 model for face detection
yolo_model = YOLO(MODEL_PATH)
print(f"YOLOv8 face detection model loaded from {MODEL_PATH}")

# Initialize InceptionResnetV1 for embedding extraction
model = InceptionResnetV1(pretrained='vggface2').eval()

# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def detect_face(image):
    """
    Detect the largest face in an image using YOLOv8

    Args:
        image: PIL Image or numpy array

    Returns:
        Cropped face as a PIL Image or None if no face detected
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image

    # Ensure image is RGB
    if len(img_np.shape) == 2:  # Grayscale
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:  # RGBA
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    # Convert to BGR for YOLOv8
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Detect faces
    results = yolo_model.predict(img_bgr, conf=0.2, verbose=False)

    # Extract face bounding boxes
    faces = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    if len(faces) == 0:
        return None

    # Select the largest face
    largest_face = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    x1, y1, x2, y2 = map(int, largest_face)

    # Ensure coordinates are within image bounds
    h, w = img_np.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Extract face region
    if x2 > x1 and y2 > y1:  # Ensure valid box dimensions
        face_img = img_np[y1:y2, x1:x2]
        return Image.fromarray(face_img)

    return None


# Convert image to embedding
def get_embedding(image_path):
    """
    Generate face embedding from an image file

    Args:
        image_path: Path to the image file

    Returns:
        Face embedding as numpy array or None if no face detected
    """
    try:
        # Open image
        img = Image.open(image_path).convert('RGB')

        # Detect face
        face = detect_face(img)

        if face is not None:
            # Transform face for the model
            face_tensor = transform(face).unsqueeze(0)  # Add batch dimension

            # Generate embedding
            with torch.no_grad():
                embedding = model(face_tensor)

            return embedding.detach().cpu().numpy()[0]  # Return as numpy array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

    return None  # Return None if any error occurs or no face detected


# Main function to generate and save embeddings
def save_embeddings(dataset_path: str, pkl_output_path: str = "assets/embeddings.pkl"):
    embeddings_dict = {}

    # Loop over each employee folder
    for emp_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, emp_folder)
        if not os.path.isdir(folder_path):  # Skip if it's not a directory
            continue

        embeddings_list = []
        images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

        print(f"Processing {len(images)} images for {emp_folder}")

        # Loop over images and extract embeddings
        for img_name in images:
            image_path = os.path.join(folder_path, img_name)
            embedding = get_embedding(image_path)

            if embedding is not None:
                embeddings_list.append(embedding)

        if embeddings_list:  # Only save if at least one embedding was found
            # Use the average embedding of all images
            embeddings_dict[emp_folder] = np.mean(embeddings_list, axis=0)
            print(f"✅ Generated embedding for {emp_folder}")
        else:
            print(f"❌ No valid embeddings for {emp_folder}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(pkl_output_path), exist_ok=True)

    # Save embeddings to .pkl file
    with open(pkl_output_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    print(f"Embeddings saved to {pkl_output_path}")


if __name__ == "__main__":
    save_embeddings("../dataset", pkl_output_path="../assets/embeddings.pkl")