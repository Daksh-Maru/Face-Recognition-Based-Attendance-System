import cv2
import torch
import numpy as np
import pickle
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import faiss
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
import threading

logger = logging.getLogger(__name__)

# Updated Enterprise Configuration with Lenient Recognition Thresholds
ENTERPRISE_CONFIG = {
    'euclidean_threshold': 0.8,  # INCREASED from 0.6 for more lenient matching
    'min_embeddings_per_id': 1,  # Accept single embeddings
    'max_embeddings_per_id': 8,
    'normalize_embeddings': True,
    'batch_size': 32,
    'faiss_index_type': 'IVF',
    'memory_limit_gb': 4,
    'cache_size': 1000,
    'num_threads': 4,
    'performance_monitoring': True,
    'auto_reload_embeddings': True,
    'confidence_threshold': 0.25  # LOWERED from 0.4 for more acceptance
}


class PerformanceMonitor:
    """Enterprise performance tracking"""

    def __init__(self):
        self.metrics = {
            'total_recognitions': 0,
            'avg_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        self.lock = threading.Lock()

    def record_recognition(self, response_time, cache_hit=False, error=False):
        with self.lock:
            self.metrics['total_recognitions'] += 1
            self.metrics['avg_response_time'] = (
                    (self.metrics['avg_response_time'] * (self.metrics['total_recognitions'] - 1) + response_time)
                    / self.metrics['total_recognitions']
            )
            if cache_hit:
                self.metrics['cache_hits'] += 1
            else:
                self.metrics['cache_misses'] += 1
            if error:
                self.metrics['errors'] += 1


monitor = PerformanceMonitor()


class EnterpriseRecognizer:
    """Enterprise-grade face recognition system with lenient thresholds"""

    def __init__(self, embeddings_path=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.faiss_index = None
        self.identity_map = {}
        self.embeddings_path = embeddings_path or os.path.abspath("./assets/embeddings.pkl")
        self.last_reload = 0
        self.executor = ThreadPoolExecutor(max_workers=ENTERPRISE_CONFIG['num_threads'])

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        model_loaded = self._initialize_model()
        if model_loaded:
            self._load_embeddings()
        else:
            logger.warning("Enterprise recognizer initialized without model")

    def _initialize_model(self):
        """Initialize FaceNet model with error handling"""
        try:
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            logger.info(f"SUCCESS: FaceNet model loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Enterprise model initialization failed: {e}")
            self.model = None
            return False

    def _load_embeddings(self):
        """Load embeddings with FAISS indexing and improved error handling"""
        try:
            start_time = time.time()

            if not os.path.exists(self.embeddings_path):
                logger.warning(f"Embeddings file not found: {self.embeddings_path}")
                os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
                with open(self.embeddings_path, 'wb') as f:
                    pickle.dump({}, f)
                return

            with open(self.embeddings_path, 'rb') as f:
                raw_data = pickle.load(f)

            if not raw_data:
                logger.warning("Empty embeddings file")
                return

            # Process and validate embeddings with lenient requirements
            valid_embeddings = []
            identity_labels = []

            for identity, data in raw_data.items():
                # Handle multiple data formats
                if isinstance(data, dict) and 'embeddings' in data:
                    emb_list = data['embeddings']
                elif isinstance(data, list):
                    emb_list = data
                elif isinstance(data, np.ndarray) and data.size == 512:
                    emb_list = [data]
                else:
                    continue

                # Validate embeddings
                valid = [e for e in emb_list if isinstance(e, np.ndarray) and e.size == 512]

                # More lenient: accept even single embeddings
                if len(valid) >= ENTERPRISE_CONFIG['min_embeddings_per_id']:
                    valid = valid[:ENTERPRISE_CONFIG['max_embeddings_per_id']]
                    for embedding in valid:
                        if ENTERPRISE_CONFIG['normalize_embeddings']:
                            embedding = embedding / np.linalg.norm(embedding)
                        valid_embeddings.append(embedding)
                        identity_labels.append(identity)

            if not valid_embeddings:
                logger.warning("No valid embeddings found")
                return

            embeddings_matrix = np.array(valid_embeddings).astype('float32')

            # CRITICAL FIX: Adjust FAISS configuration to prevent clustering warnings
            if len(valid_embeddings) > 100:
                nlist = max(1, len(valid_embeddings) // 39)  # Ensure proper clustering ratio
                nlist = min(nlist, 256)  # Cap at reasonable maximum
                quantizer = faiss.IndexFlatL2(512)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, 512, nlist)
                self.faiss_index.train(embeddings_matrix)
            else:
                # Use flat index for smaller datasets
                self.faiss_index = faiss.IndexFlatL2(512)

            self.faiss_index.add(embeddings_matrix)
            self.identity_map = {i: identity for i, identity in enumerate(identity_labels)}

            load_time = time.time() - start_time
            logger.info(
                f"✅ Loaded {len(valid_embeddings)} embeddings for {len(set(identity_labels))} identities in {load_time:.2f}s")
            self.last_reload = time.time()

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            self.faiss_index = None

    def get_embedding(self, face_img):
        """Enterprise-grade embedding extraction"""
        start_time = time.time()

        try:
            if self.model is None:
                logger.error("Model not initialized")
                return None

            if face_img is None or face_img.size == 0:
                logger.error("Invalid face image")
                return None

            # Ensure proper image format
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                if face_img.dtype == np.uint8:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            else:
                logger.error("Face image must be 3-channel RGB")
                return None

            pil_image = Image.fromarray(face_img)
            face_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(face_tensor).cpu().numpy()[0]

            if ENTERPRISE_CONFIG['normalize_embeddings']:
                embedding = embedding / np.linalg.norm(embedding)

            monitor.record_recognition(time.time() - start_time, cache_hit=False)
            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            monitor.record_recognition(time.time() - start_time, error=True)
            return None

    def recognize_face(self, embedding, confidence_threshold=None):
        """Fast recognition using FAISS index with lenient thresholds"""
        if self.faiss_index is None or embedding is None:
            return "Unknown", 0.0

        try:
            threshold = confidence_threshold or ENTERPRISE_CONFIG['euclidean_threshold']
            embedding = embedding.reshape(1, -1).astype('float32')
            k = min(10, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(embedding, k=k)

            identity_scores = {}
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx >= len(self.identity_map):
                    continue

                identity = self.identity_map.get(idx, "Unknown")
                if identity == "Unknown":
                    continue

                # CRITICAL FIX: Enhanced type-safe similarity calculation
                if isinstance(distance, dict):
                    logger.error(f"Distance is a dict, skipping: {distance}")
                    continue

                try:
                    # Ensure distance is numeric before division
                    distance_value = float(distance)
                    similarity = max(0, 1 - (distance_value / (threshold * 2)))
                except (TypeError, ValueError) as e:
                    logger.error(f"Similarity calculation error: {e}")
                    continue

                if identity in identity_scores:
                    identity_scores[identity] = max(identity_scores[identity], similarity)
                else:
                    identity_scores[identity] = similarity

            if not identity_scores:
                return "Unknown", 0.0

            best_identity, best_score = max(identity_scores.items(), key=lambda x: x[1])

            if not isinstance(best_score, (float, int)):
                logger.error(f"Best score is not numeric: {best_score}")
                return "Unknown", 0.0

            # Use the lenient confidence threshold
            if best_score < ENTERPRISE_CONFIG['confidence_threshold']:
                return "Unknown", best_score

            return best_identity, best_score

        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return "Unknown", 0.0

    def reload_embeddings_if_needed(self):
        """Hot reload embeddings if file has been updated"""
        if not ENTERPRISE_CONFIG['auto_reload_embeddings']:
            return

        try:
            if os.path.exists(self.embeddings_path):
                file_mtime = os.path.getmtime(self.embeddings_path)
                if file_mtime > self.last_reload:
                    logger.info("Reloading embeddings due to file update")
                    self._load_embeddings()
        except Exception as e:
            logger.error(f"Failed to check for embedding updates: {e}")

    def get_metrics(self):
        """Get performance metrics"""
        return monitor.metrics.copy()

    def add_face_encoding(self, identity, embedding):
        """Add new face encoding to the system"""
        try:
            if os.path.exists(self.embeddings_path):
                with open(self.embeddings_path, 'rb') as f:
                    embeddings_data = pickle.load(f)
            else:
                embeddings_data = {}

            if identity not in embeddings_data:
                embeddings_data[identity] = []
            if isinstance(embeddings_data[identity], list):
                embeddings_data[identity].append(embedding)
            else:
                embeddings_data[identity] = [embeddings_data[identity], embedding]

            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(embeddings_data, f)

            self._load_embeddings()
            logger.info(f"Added new encoding for {identity}")
            return True

        except Exception as e:
            logger.error(f"Failed to add face encoding: {e}")
            return False


# Global recognizer instance
recognizer = EnterpriseRecognizer()


# Legacy API compatibility
def get_embedding(face_img):
    """Legacy API wrapper"""
    recognizer.reload_embeddings_if_needed()
    return recognizer.get_embedding(face_img)


def predict_face_with_occlusion_handling(face_img, occlusion_info=None, stored_embeddings_param=None,
                                         metric='euclidean'):
    """Legacy API wrapper with enterprise backend"""
    try:
        recognizer.reload_embeddings_if_needed()
        embedding = recognizer.get_embedding(face_img)
        if embedding is None:
            return "Unknown", 0.0

        threshold = ENTERPRISE_CONFIG['euclidean_threshold']
        if occlusion_info:
            occlusion_level = occlusion_info.get('total_occlusion_percentage', 0.0)
            threshold += occlusion_level * 0.05  # Reduced penalty for occlusion

        return recognizer.recognize_face(embedding, threshold)

    except Exception as e:
        logger.error(f"Recognition pipeline failed: {e}")
        return "Unknown", 0.0


def predict_face(embedding, stored_embeddings_param=None, threshold=None):
    """Legacy API wrapper"""
    if embedding is None:
        return "Unknown"
    identity, confidence = recognizer.recognize_face(embedding, threshold)
    return identity


def get_recognition_metrics():
    """Get system performance metrics"""
    return recognizer.get_metrics()


def add_new_face(identity, face_img):
    """Add a new face to the recognition system"""
    embedding = recognizer.get_embedding(face_img)
    if embedding is not None:
        return recognizer.add_face_encoding(identity, embedding)
    return False


logger.info("Enterprise recognition system initialized with lenient thresholds")
