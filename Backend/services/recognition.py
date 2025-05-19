import cv2
import torch
import numpy as np
import pickle
import os
from facenet_pytorch import InceptionResnetV1
from detection import detect_face  # Your YOLOv8-face detection function
from PIL import Image
from torchvision import transforms
from occlusion_detection import extract_selective_lgbphs

# Initialize FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load embeddings
with open('../assets/embeddings.pkl', 'rb') as f:
    stored_embeddings = pickle.load(f)

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


def get_embedding_with_occlusion_handling(face_pixels, occlusion_mask=None):
    """Generate embedding from face image with occlusion handling"""
    # Ensure the input is a NumPy array
    if isinstance(face_pixels, Image.Image):
        face_pixels = np.array(face_pixels)

    # If we have an occlusion mask, apply it to focus on non-occluded regions
    if occlusion_mask is not None:
        # Check if the face is mostly occluded
        if np.mean(occlusion_mask) > 0.7:  # More than 70% occluded
            return None  # Too occluded to process

        # Apply mask to image (invert mask since 0=non-occluded in our mask)
        masked_face = face_pixels.copy()
        for c in range(3):
            masked_face[:, :, c] = masked_face[:, :, c] * (1 - occlusion_mask)

        # Use the masked face
        face_pil = Image.fromarray(masked_face)
    else:
        face_pil = Image.fromarray(face_pixels)

    # Apply transforms
    face_tensor = transform(face_pil).unsqueeze(0)

    # Generate embedding
    with torch.no_grad():
        embedding = model(face_tensor)

    return embedding.detach().cpu().numpy()[0]


def predict_face(embedding, stored_embeddings=stored_embeddings):
    """
    Compare the embedding with stored embeddings to find a match

    Args:
        embedding: Face embedding of the query face
        stored_embeddings: Dictionary of stored embeddings

    Returns:
        Identity of the matched face or "Unknown"
    """
    if embedding is None:
        return "Unknown"

    min_dist = float('inf')
    identity = "Unknown"

    for name, emb in stored_embeddings.items():
        dist = np.linalg.norm(embedding - emb)
        if dist < min_dist and dist < 1.0:  # Threshold can be adjusted
            min_dist = dist
            identity = name

    return identity


def predict_face_with_occlusion_handling(face_pixels, occlusion_mask=None, stored_embeddings=stored_embeddings):
    """Recognize face with occlusion handling"""
    # Get embedding with occlusion handling
    embedding = get_embedding_with_occlusion_handling(face_pixels, occlusion_mask)

    # If embedding generation failed (e.g., too much occlusion), return Unknown
    if embedding is None:
        return "Unknown"

    # Match embedding to known identities
    min_dist = float("inf")
    identity = "Unknown"

    for name, emb in stored_embeddings.items():
        dist = np.linalg.norm(embedding - emb)
        if dist < min_dist and dist < 1.0:  # Threshold can be adjusted
            min_dist = dist
            identity = name

    return identity
