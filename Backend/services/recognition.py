# services/recognition.py
import numpy as np
import faiss
import pickle
import os
import logging
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image  # Added missing import

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Face recognition service using FAISS indexing"""

    def __init__(self, embeddings_path="assets/embeddings.pkl"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.embeddings_path = embeddings_path
        self.index, self.id_map = self._load_embeddings()
        self.threshold = 0.7  # Optimized for LFW dataset

    def _load_embeddings(self):
        try:
            if not os.path.exists(self.embeddings_path):
                logger.warning(f"Embeddings file not found: {self.embeddings_path}")
                return None, {}

            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)

            # Validate and process embeddings
            embeddings = []
            ids = []

            for pid, emb in data.items():
                if isinstance(emb, np.ndarray) and emb.size == 512:
                    embeddings.append(emb.astype('float32'))
                    ids.append(pid)

            if not embeddings:
                logger.warning("No valid embeddings found in file")
                return None, {}

            # Build FAISS index with proper error handling
            try:
                if len(embeddings) > 1000:
                    # For large datasets, use IVF index with proper clustering ratio
                    nlist = min(256, len(embeddings) // 39)  # At least 39 points per centroid
                    quantizer = faiss.IndexFlatL2(512)
                    index = faiss.IndexIVFFlat(quantizer, 512, nlist)
                    index.train(np.array(embeddings))
                else:
                    # For smaller datasets, use flat index
                    index = faiss.IndexFlatL2(512)

                index.add(np.array(embeddings))
                logger.info(f"FAISS index built with {len(embeddings)} embeddings")
                return index, {i: pid for i, pid in enumerate(ids)}
            except Exception as e:
                logger.error(f"FAISS index creation failed: {e}")
                return None, {}

        except Exception as e:
            logger.error(f"Embeddings loading failed: {e}")
            return None, {}

    def get_embedding(self, face_image):
        """Generate normalized face embedding"""
        try:
            if face_image is None:
                return None

            # Convert to PIL Image if necessary
            if not isinstance(face_image, Image.Image):
                face_image = Image.fromarray(face_image)

            face_tensor = self.transform(face_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(face_tensor).cpu().numpy()[0]

            # Normalize embedding for consistent similarity calculation
            embedding = embedding / np.linalg.norm(embedding)
            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def recognize(self, embedding):
        """Identify face using FAISS similarity search"""
        try:
            if self.index is None or embedding is None:
                return "Unknown", 0.0

            # FAISS search with proper type handling
            embedding_array = np.array([embedding]).astype('float32')
            distances, indices = self.index.search(embedding_array, 1)

            # Extract and validate distance
            if indices[0][0] == -1:  # No match found
                return "Unknown", 0.0

            distance = float(distances[0][0])  # Explicit conversion to float

            if distance < self.threshold:
                pid = self.id_map.get(indices[0][0], "Unknown")
                confidence = 1.0 - (distance / self.threshold)
                return pid, float(confidence)  # Ensure float return

            return "Unknown", 0.0

        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return "Unknown", 0.0
