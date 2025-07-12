import numpy as np
import faiss
import pickle
import os
import logging
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """Face recognition service using FAISS indexing."""

    def __init__(self, embeddings_path="assets/embeddings.pkl"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        ])
        self.embeddings_path = embeddings_path
        self.index, self.id_map = self._load_embeddings()
        self.threshold = 0.7

    def _load_embeddings(self):
        if not os.path.exists(self.embeddings_path):
            logger.warning(f"Embeddings file not found: {self.embeddings_path}")
            return None, {}
        with open(self.embeddings_path, 'rb') as f:
            data = pickle.load(f)

        embeddings, ids = [], []
        for pid, emb in data.items():
            if isinstance(emb, np.ndarray) and emb.size == 512:
                embeddings.append(emb.astype('float32'))
                ids.append(pid)
        if not embeddings:
            logger.warning("No valid embeddings found.")
            return None, {}

        # Dynamic FAISS index selection
        if len(embeddings) > 1000:
            nlist = min(256, len(embeddings) // 39)
            quantizer = faiss.IndexFlatL2(512)
            index = faiss.IndexIVFFlat(quantizer, 512, nlist)
            index.train(np.array(embeddings))
        else:
            index = faiss.IndexFlatL2(512)
        index.add(np.array(embeddings))
        logger.info("FAISS index built with %d embeddings", len(embeddings))
        return index, {i: pid for i, pid in enumerate(ids)}

    def get_embedding(self, face_image):
        """
        Generates a normalized 512-D embedding from a face crop.
        """
        if face_image is None:
            return None
        if not isinstance(face_image, Image.Image):
            face_image = Image.fromarray(face_image)
        tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor).cpu().numpy()[0]
        return emb / np.linalg.norm(emb)

    def recognize(self, embedding):
        """
        Identifies the person by nearest neighbor search in FAISS.
        Returns (person_id, confidence).
        """
        if self.index is None or embedding is None:
            return "Unknown", 0.0
        arr = np.array([embedding]).astype('float32')
        distances, indices = self.index.search(arr, 1)
        idx = indices[0][0]
        if idx == -1 or distances[0][0] >= self.threshold:
            return "Unknown", 0.0
        pid = self.id_map.get(idx, "Unknown")
        confidence = 1.0 - (distances[0][0] / self.threshold)
        return pid, float(confidence)
