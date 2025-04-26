import torch
import numpy as np
import pickle
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# Load the facenet-pytorch model (pretrained on VGGFace2)
try:
    model = InceptionResnetV1(pretrained='vggface2').eval()
except Exception as e:
    raise RuntimeError("Could not load FaceNet PyTorch model.") from e

# Load stored embeddings
try:
    with open("assets/embeddings.pkl", "rb") as f:
        stored_embeddings = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Embeddings file not found. Please run generate_embeddings.py")

# # Load class labels (optional)
# try:
#     with open("assets/class_labels.pkl", "rb") as f:
#         class_labels = pickle.load(f)
# except FileNotFoundError:
#     class_labels = {}

# Define the image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Generate embedding from face image
def get_embedding(face_pixels):
    # Ensure the input is a NumPy array
    if isinstance(face_pixels, Image.Image):
        img_tensor = preprocess(np.array(face_pixels))
    else:
        img_tensor = preprocess(face_pixels)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding[0].numpy()

# Match input embedding to known identities
def predict_face(embedding, stored_embeddings):
    min_dist = float("inf")
    identity = "Unknown"

    for label, db_emb in stored_embeddings.items():
        dist = np.linalg.norm(embedding - db_emb)
        if dist < min_dist and dist < 0.8:
            min_dist = dist
            identity = label
    return identity
