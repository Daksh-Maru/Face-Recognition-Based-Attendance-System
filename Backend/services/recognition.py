import torch
import numpy as np
import pickle
import os
import logging
import time
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import faiss
from typing import Dict, List, Tuple, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognitionSystem:
    """
    Optimized face recognition system for LFW and standard datasets
    """

    def __init__(self, embeddings_path: str = "assets/embeddings.pkl",
                 model_device: str = None, use_faiss: bool = True):
        """
        Initialize the face recognition system

        Args:
            embeddings_path: Path to stored embeddings file
            model_device: Device to run model on ('cuda' or 'cpu')
            use_faiss: Whether to use FAISS for fast similarity search
        """
        self.embeddings_path = embeddings_path
        self.device = torch.device(model_device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.use_faiss = use_faiss

        # Initialize model
        self.model = self._load_model()

        # Load embeddings
        self.stored_embeddings = self._load_embeddings()
        self.identity_map = {}
        self.faiss_index = None

        # Initialize FAISS index if requested
        if self.use_faiss and self.stored_embeddings:
            self._build_faiss_index()

        # Define preprocessing pipeline optimized for LFW
        self.preprocess = transforms.Compose([
            transforms.ToPILImage() if not isinstance(transforms.ToPILImage(), type) else lambda x: x,
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Fixed for RGB
        ])

        logger.info(f"Face recognition system initialized with {len(self.stored_embeddings)} identities")

    def _load_model(self) -> InceptionResnetV1:
        """Load and initialize the FaceNet model"""
        try:
            model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            logger.info(f"FaceNet model loaded successfully on {self.device}")
            return model
        except Exception as e:
            raise RuntimeError(f"Could not load FaceNet PyTorch model: {e}")

    def _load_embeddings(self) -> Dict:
        """Load stored embeddings with enhanced error handling"""
        try:
            with open(self.embeddings_path, "rb") as f:
                embeddings = pickle.load(f)

            # Validate embeddings structure
            if not isinstance(embeddings, dict):
                raise ValueError("Embeddings must be a dictionary")

            # Process embeddings to handle different formats
            processed_embeddings = {}
            for identity, embedding_data in embeddings.items():
                if isinstance(embedding_data, list):
                    # Handle multiple embeddings per identity
                    valid_embeddings = [emb for emb in embedding_data
                                        if isinstance(emb, np.ndarray) and emb.size == 512]
                    if valid_embeddings:
                        # Use average embedding for better representation
                        processed_embeddings[identity] = np.mean(valid_embeddings, axis=0)
                elif isinstance(embedding_data, np.ndarray) and embedding_data.size == 512:
                    processed_embeddings[identity] = embedding_data
                else:
                    logger.warning(f"Invalid embedding format for {identity}")

            logger.info(f"Loaded {len(processed_embeddings)} valid embeddings")
            return processed_embeddings

        except FileNotFoundError:
            logger.warning("Embeddings file not found. System will work with empty embeddings.")
            return {}
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return {}

    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        try:
            embeddings_list = []
            identity_list = []

            for identity, embedding in self.stored_embeddings.items():
                embeddings_list.append(embedding)
                identity_list.append(identity)

            if not embeddings_list:
                return

            # Convert to numpy array
            embeddings_matrix = np.array(embeddings_list).astype('float32')

            # Choose appropriate FAISS index
            if len(embeddings_list) > 1000:
                # Use IVF index for large datasets
                nlist = min(int(np.sqrt(len(embeddings_list))), 256)
                quantizer = faiss.IndexFlatL2(512)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, 512, nlist)
                self.faiss_index.train(embeddings_matrix)
            else:
                # Use flat index for smaller datasets
                self.faiss_index = faiss.IndexFlatL2(512)

            self.faiss_index.add(embeddings_matrix)
            self.identity_map = {i: identity for i, identity in enumerate(identity_list)}

            logger.info(f"FAISS index built with {len(embeddings_list)} embeddings")

        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            self.faiss_index = None

    def get_embedding(self, face_pixels: Union[np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """
        Generate embedding from face image with enhanced preprocessing

        Args:
            face_pixels: Input face image as numpy array or PIL Image

        Returns:
            512-dimensional embedding vector or None if processing fails
        """
        try:
            # Input validation
            if face_pixels is None:
                return None

            # Handle different input types
            if isinstance(face_pixels, Image.Image):
                # Convert PIL to numpy
                face_array = np.array(face_pixels)
            elif isinstance(face_pixels, np.ndarray):
                face_array = face_pixels.copy()
            else:
                logger.error("Invalid input type for face_pixels")
                return None

            # Ensure 3-channel RGB
            if len(face_array.shape) == 2:
                face_array = np.stack([face_array] * 3, axis=-1)
            elif face_array.shape[2] == 4:  # RGBA
                face_array = face_array[:, :, :3]

            # Ensure uint8 format
            if face_array.dtype != np.uint8:
                if face_array.max() <= 1.0:
                    face_array = (face_array * 255).astype(np.uint8)
                else:
                    face_array = np.clip(face_array, 0, 255).astype(np.uint8)

            # Apply preprocessing
            img_tensor = self.preprocess(face_array)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # Generate embedding
            with torch.no_grad():
                embedding = self.model(img_tensor)

            # Convert to numpy and normalize
            embedding_np = embedding[0].cpu().numpy()
            embedding_normalized = embedding_np / np.linalg.norm(embedding_np)

            return embedding_normalized

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def predict_face_faiss(self, embedding: np.ndarray, k: int = 5,
                           threshold: float = 0.6) -> Tuple[str, float]:
        """
        Predict identity using FAISS index for fast search

        Args:
            embedding: Input embedding vector
            k: Number of nearest neighbors to search
            threshold: Distance threshold for recognition

        Returns:
            Tuple of (identity, confidence_score)
        """
        if self.faiss_index is None or embedding is None:
            return "Unknown", 0.0

        try:
            # Prepare embedding for search
            query_embedding = embedding.reshape(1, -1).astype('float32')

            # Search for nearest neighbors
            distances, indices = self.faiss_index.search(query_embedding, k)

            # Process results
            best_distance = distances[0][0]
            best_idx = indices[0][0]

            if best_distance < threshold and best_idx in self.identity_map:
                identity = self.identity_map[best_idx]
                confidence = max(0.0, 1.0 - (best_distance / threshold))
                return identity, confidence

            return "Unknown", 0.0

        except Exception as e:
            logger.error(f"Error in FAISS prediction: {e}")
            return "Unknown", 0.0

    def predict_face_linear(self, embedding: np.ndarray,
                            threshold: float = 0.6) -> Tuple[str, float]:
        """
        Predict identity using linear search (fallback method)

        Args:
            embedding: Input embedding vector
            threshold: Distance threshold for recognition

        Returns:
            Tuple of (identity, confidence_score)
        """
        if not self.stored_embeddings or embedding is None:
            return "Unknown", 0.0

        try:
            min_distance = float("inf")
            best_identity = "Unknown"

            for identity, stored_emb in self.stored_embeddings.items():
                # Calculate Euclidean distance
                distance = np.linalg.norm(embedding - stored_emb)

                if distance < min_distance:
                    min_distance = distance
                    best_identity = identity

            # Check if distance is within threshold
            if min_distance < threshold:
                confidence = max(0.0, 1.0 - (min_distance / threshold))
                return best_identity, confidence

            return "Unknown", 0.0

        except Exception as e:
            logger.error(f"Error in linear prediction: {e}")
            return "Unknown", 0.0

    def predict_face(self, embedding: np.ndarray, threshold: float = 0.6) -> Tuple[str, float]:
        """
        Main prediction function that chooses between FAISS and linear search

        Args:
            embedding: Input embedding vector
            threshold: Distance threshold for recognition

        Returns:
            Tuple of (identity, confidence_score)
        """
        if self.use_faiss and self.faiss_index is not None:
            return self.predict_face_faiss(embedding, threshold=threshold)
        else:
            return self.predict_face_linear(embedding, threshold=threshold)

    def recognize_face(self, face_image: Union[np.ndarray, Image.Image],
                       threshold: float = 0.6) -> Dict:
        """
        Complete face recognition pipeline

        Args:
            face_image: Input face image
            threshold: Recognition threshold

        Returns:
            Dictionary with recognition results
        """
        start_time = time.time()

        try:
            # Generate embedding
            embedding = self.get_embedding(face_image)
            if embedding is None:
                return {
                    "identity": "Unknown",
                    "confidence": 0.0,
                    "error": "Failed to generate embedding",
                    "processing_time": time.time() - start_time
                }

            # Predict identity
            identity, confidence = self.predict_face(embedding, threshold)

            return {
                "identity": identity,
                "confidence": round(confidence, 4),
                "threshold": threshold,
                "processing_time": round(time.time() - start_time, 4),
                "embedding_norm": round(np.linalg.norm(embedding), 4)
            }

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return {
                "identity": "Unknown",
                "confidence": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def add_identity(self, identity: str, face_images: List[Union[np.ndarray, Image.Image]]):
        """
        Add new identity to the system

        Args:
            identity: Name/ID of the person
            face_images: List of face images for the person
        """
        try:
            embeddings = []
            for face_image in face_images:
                embedding = self.get_embedding(face_image)
                if embedding is not None:
                    embeddings.append(embedding)

            if embeddings:
                # Use average embedding
                avg_embedding = np.mean(embeddings, axis=0)
                self.stored_embeddings[identity] = avg_embedding

                # Rebuild FAISS index
                if self.use_faiss:
                    self._build_faiss_index()

                logger.info(f"Added identity '{identity}' with {len(embeddings)} embeddings")
            else:
                logger.warning(f"No valid embeddings generated for '{identity}'")

        except Exception as e:
            logger.error(f"Error adding identity: {e}")

    def save_embeddings(self, output_path: str = None):
        """Save current embeddings to file"""
        try:
            save_path = output_path or self.embeddings_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'wb') as f:
                pickle.dump(self.stored_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Embeddings saved to {save_path}")

        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "total_identities": len(self.stored_embeddings),
            "device": str(self.device),
            "using_faiss": self.faiss_index is not None,
            "model_loaded": self.model is not None,
            "embeddings_path": self.embeddings_path
        }


# Global instance for backward compatibility
try:
    recognition_system = FaceRecognitionSystem()
    model = recognition_system.model
    stored_embeddings = recognition_system.stored_embeddings
    preprocess = recognition_system.preprocess
except Exception as e:
    logger.error(f"Failed to initialize global recognition system: {e}")
    recognition_system = None


# Backward compatibility functions
def get_embedding(face_pixels):
    """Backward compatible embedding generation function"""
    if recognition_system:
        return recognition_system.get_embedding(face_pixels)
    else:
        logger.error("Recognition system not initialized")
        return None


def predict_face(embedding, stored_embeddings_param=None, threshold=0.6):
    """Backward compatible prediction function"""
    if recognition_system:
        identity, confidence = recognition_system.predict_face(embedding, threshold)
        return identity
    else:
        logger.error("Recognition system not initialized")
        return "Unknown"


def predict_face_with_confidence(embedding, threshold=0.6):
    """Enhanced prediction function that returns confidence"""
    if recognition_system:
        return recognition_system.predict_face(embedding, threshold)
    else:
        return "Unknown", 0.0


# LFW evaluation utilities
def evaluate_lfw_pairs(pairs_file: str, images_dir: str,
                       recognition_sys: FaceRecognitionSystem) -> Dict:
    """
    Evaluate system performance on LFW pairs

    Args:
        pairs_file: Path to LFW pairs file
        images_dir: Directory containing LFW images
        recognition_sys: Face recognition system instance

    Returns:
        Dictionary with evaluation results
    """
    try:
        from sklearn.metrics import roc_curve, auc

        # Load pairs data
        pairs_data = []
        with open(pairs_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:  # Same person
                    pairs_data.append((parts[0], parts[0], int(parts[1]), int(parts[2]), 1))
                elif len(parts) == 4:  # Different person
                    pairs_data.append((parts[0], parts[2], int(parts[1]), int(parts[3]), 0))

        # Calculate similarities
        similarities = []
        labels = []

        for person1, person2, img1_num, img2_num, is_same in pairs_data:
            try:
                # Load images
                img1_path = os.path.join(images_dir, person1, f"{person1}_{img1_num:04d}.jpg")
                img2_path = os.path.join(images_dir, person2, f"{person2}_{img2_num:04d}.jpg")

                if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                    continue

                img1 = Image.open(img1_path)
                img2 = Image.open(img2_path)

                # Generate embeddings
                emb1 = recognition_sys.get_embedding(img1)
                emb2 = recognition_sys.get_embedding(img2)

                if emb1 is not None and emb2 is not None:
                    # Calculate similarity (1 - distance)
                    distance = np.linalg.norm(emb1 - emb2)
                    similarity = 1.0 - distance

                    similarities.append(similarity)
                    labels.append(is_same)

            except Exception as e:
                logger.warning(f"Error processing pair: {e}")
                continue

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)

        # Find best threshold
        best_threshold_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_threshold_idx]

        # Calculate accuracy at best threshold
        predictions = [1 if sim > best_threshold else 0 for sim in similarities]
        accuracy = np.mean([pred == label for pred, label in zip(predictions, labels)])

        return {
            "accuracy": accuracy,
            "auc": roc_auc,
            "best_threshold": best_threshold,
            "total_pairs": len(pairs_data),
            "processed_pairs": len(similarities),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }

    except Exception as e:
        logger.error(f"Error in LFW evaluation: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize system
        face_system = FaceRecognitionSystem()

        # Print system stats
        stats = face_system.get_stats()
        print("Face Recognition System Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Example recognition (if you have a test image)
        # test_image = Image.open("test_face.jpg")
        # result = face_system.recognize_face(test_image)
        # print(f"Recognition result: {result}")

    except Exception as e:
        print(f"Error: {e}")
