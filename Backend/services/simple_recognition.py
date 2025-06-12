# simple_recognition.py
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

# Simple global model initialization
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = None
transform = None


def initialize_simple_model():
    """Initialize simple FaceNet model"""
    global model, transform
    try:
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        logger.info(f"Simple FaceNet model loaded on {device}")
        return True
    except Exception as e:
        logger.error(f"Failed to load simple model: {e}")
        return False


def get_embedding_simple(face_img):
    """Simple embedding extraction"""
    global model, transform

    if model is None:
        if not initialize_simple_model():
            return None

    try:
        if isinstance(face_img, np.ndarray):
            face_pil = Image.fromarray(face_img)
        else:
            face_pil = face_img

        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(face_tensor).cpu().numpy()[0]

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    except Exception as e:
        logger.error(f"Simple embedding failed: {e}")
        return None


# Initialize on import
initialize_simple_model()
