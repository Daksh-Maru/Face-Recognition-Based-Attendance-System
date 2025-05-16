import cv2
import torch
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1
from .detection import detect_face  # Your YOLOv8-face detection function
from PIL import Image
from torchvision import transforms

# Initialize FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load embeddings
with open('assets/embeddings.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)

# Transform function to prepare face image for FaceNet
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_embedding(image):
    # Detect face using YOLOv8
    face_img = detect_face(image)
    if face_img is None:
        return None
    # Convert to PIL Image
    face_pil = Image.fromarray(face_img)
    # Apply transforms
    face_tensor = transform(face_pil).unsqueeze(0)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding.detach().cpu().numpy()[0]  # Return flattened array


def predict_face(embedding, stored_embeddings):
    """
    Compare the embedding with stored embeddings to find a match

    Args:
        embedding: Face embedding of the query face
        stored_embeddings: Dictionary of stored embeddings

    Returns:
        Identity of the matched face or "Unknown"
    """
    min_dist = float('inf')
    identity = "Unknown"

    for name, emb in stored_embeddings.items():
        dist = np.linalg.norm(embedding - emb)
        if dist < min_dist and dist < 0.8:  # Threshold can be adjusted
            min_dist = dist
            identity = name

    return identity

