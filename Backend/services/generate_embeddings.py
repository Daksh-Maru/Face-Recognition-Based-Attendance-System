import os
import cv2
import numpy as np
import pickle
import torch
import logging
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
import json
import shutil

# Add services to path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Fixed imports
from detection import detect_face
from preprocessing import preprocess_image_with_occlusion_handling

# Setup enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Updated Enterprise Configuration with Lenient Settings
ENTERPRISE_CONFIG = {
    'detection': {
        'min_face_size': (60, 60),  # Reduced minimum size
        'max_attempts': 3,
        'scale_factors': [1.05, 1.1],
        'confidence_threshold': 0.2  # Very lenient detection
    },
    'quality': {
        'min_quality_score': 0.15,  # Very lenient quality
        'min_sharpness': 10,  # Reduced sharpness requirement
        'brightness_range': (15, 255),  # Very wide range
        'min_contrast': 8  # Minimal contrast requirement
    },
    'processing': {
        'min_images': 1,  # Accept single images
        'max_images': 20,  # Increased for better coverage
        'target_embeddings': 5,
        'workers': 4,
        'memory_limit_gb': 4
    },
    'enterprise': {
        'save_backup': True,
        'save_metadata': True,
        'enable_validation': True,
        'progress_reporting': True,
        'auto_cleanup': True,
        'normalize_embeddings': True
    }
}


class EmbeddingGenerationMonitor:
    """Enterprise monitoring for embedding generation"""

    def __init__(self):
        self.stats = {
            'start_time': time.time(),
            'total_identities': 0,
            'successful_identities': 0,
            'total_images': 0,
            'successful_embeddings': 0,
            'failed_detections': 0,
            'failed_quality': 0,
            'processing_times': []
        }
        self.lock = threading.Lock()

    def update_stats(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    if isinstance(self.stats[key], list):
                        self.stats[key].append(value)
                    else:
                        self.stats[key] += value

    def get_summary(self):
        runtime = time.time() - self.stats['start_time']
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        return {
            'runtime_seconds': runtime,
            'success_rate': (self.stats['successful_identities'] / max(1, self.stats['total_identities'])) * 100,
            'embedding_rate': (self.stats['successful_embeddings'] / max(1, self.stats['total_images'])) * 100,
            'avg_processing_time': avg_time,
            'images_per_second': self.stats['total_images'] / runtime if runtime > 0 else 0,
            **self.stats
        }


class EnterpriseEmbeddingGenerator:
    """Enterprise-grade embedding generation with lenient quality requirements"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.monitor = EmbeddingGenerationMonitor()
        self.executor = ThreadPoolExecutor(max_workers=ENTERPRISE_CONFIG['processing']['workers'])
        self._initialize_models()

    def _initialize_models(self):
        """Initialize FaceNet model with enterprise settings"""
        try:
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            logger.info(f"✅ FaceNet model loaded on {self.device}")

            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            return True
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            return False

    def get_embedding(self, face_img):
        """Generate normalized embedding with validation"""
        try:
            if face_img is None or face_img.size == 0:
                return None

            # Resize to standard size
            face_resized = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_CUBIC)

            # Ensure RGB format
            if len(face_resized.shape) == 3:
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)

            # Convert to PIL and apply transform
            pil_image = Image.fromarray(face_rgb)
            face_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(face_tensor).cpu().numpy()[0]

            # Normalize embedding
            if ENTERPRISE_CONFIG['enterprise']['normalize_embeddings']:
                embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def assess_image_quality_lenient(self, face_img):
        """Very lenient quality assessment for maximum acceptance"""
        try:
            if face_img is None or face_img.size == 0:
                return False, 0.0

            # Convert to grayscale for analysis
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_img

            h, w = gray.shape

            # Very lenient size check
            min_size = min(ENTERPRISE_CONFIG['detection']['min_face_size'])
            if min(h, w) < min_size:
                return False, 0.0

            # Lenient quality metrics
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = gray.std()

            # Very lenient thresholds
            sharpness_score = min(1.0, sharpness / 30.0)  # Much lower threshold
            brightness_range = ENTERPRISE_CONFIG['quality']['brightness_range']
            brightness_score = 1.0 if brightness_range[0] <= brightness <= brightness_range[
                1] else 0.7  # Still accept marginal
            contrast_score = min(1.0, contrast / 20.0)  # Lower threshold

            # Calculate overall quality score with lenient weighting
            size_score = min(1.0, (h * w) / (min_size ** 2))
            quality_score = (size_score * 0.2 + sharpness_score * 0.3 +
                             brightness_score * 0.3 + contrast_score * 0.2)

            # Very lenient pass criteria
            passes = (
                    quality_score >= ENTERPRISE_CONFIG['quality']['min_quality_score'] and
                    sharpness >= ENTERPRISE_CONFIG['quality']['min_sharpness'] and
                    brightness_range[0] <= brightness <= brightness_range[1] and
                    contrast >= ENTERPRISE_CONFIG['quality']['min_contrast']
            )

            return passes, quality_score

        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return True, 0.5  # Default to accept on error

    def process_identity(self, identity_path):
        """Process one identity with lenient quality requirements"""
        identity_name = os.path.basename(identity_path)
        embeddings = []

        try:
            # Get all image files
            image_files = [f for f in os.listdir(identity_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

            if not image_files:
                logger.warning(f"No images found for {identity_name}")
                return None

            logger.info(f"Processing {identity_name}: {len(image_files)} images found")
            self.monitor.update_stats(total_images=len(image_files))

            # Limit processing for efficiency but be more inclusive
            max_images = ENTERPRISE_CONFIG['processing']['max_images']
            image_files = image_files[:max_images]

            for img_file in image_files:
                img_path = os.path.join(identity_path, img_file)

                try:
                    # Load image with multiple methods
                    img = cv2.imread(img_path)
                    if img is None:
                        # Try with different color flags
                        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
                        if img is None:
                            logger.debug(f"Failed to load: {img_file}")
                            continue

                    # Convert to RGB
                    if len(img.shape) == 3:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                    # Face detection with multiple attempts
                    face_result = detect_face(img_rgb, is_webcam=False)
                    if isinstance(face_result, tuple):
                        face_img = face_result[0]
                    else:
                        face_img = face_result

                    if face_img is None:
                        self.monitor.update_stats(failed_detections=1)
                        logger.debug(f"No face detected: {img_file}")
                        continue

                    # Lenient quality assessment
                    quality_pass, quality_score = self.assess_image_quality_lenient(face_img)
                    if not quality_pass:
                        self.monitor.update_stats(failed_quality=1)
                        logger.debug(f"Quality failed: {img_file} (score: {quality_score:.2f})")
                        # Continue anyway if quality is borderline
                        if quality_score < 0.1:  # Only reject very poor quality
                            continue

                    # Generate embedding
                    embedding = self.get_embedding(face_img)
                    if embedding is not None:
                        embeddings.append(embedding)
                        self.monitor.update_stats(successful_embeddings=1)
                        logger.debug(f"✅ Generated embedding: {img_file} (quality: {quality_score:.2f})")

                        # Stop if we have enough embeddings
                        if len(embeddings) >= ENTERPRISE_CONFIG['processing']['target_embeddings']:
                            break
                    else:
                        logger.debug(f"❌ Embedding failed: {img_file}")

                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
                    continue

            # Accept identities with at least one good embedding
            if len(embeddings) >= ENTERPRISE_CONFIG['processing']['min_images']:
                logger.info(f"✅ Generated {len(embeddings)} embeddings for {identity_name}")
                return {identity_name: embeddings}
            else:
                logger.warning(f"❌ Insufficient embeddings for {identity_name}: {len(embeddings)}")
                return None

        except Exception as e:
            logger.error(f"Error processing identity {identity_name}: {e}")
            return None

    def generate_embeddings(self, dataset_path, output_path):
        """Main embedding generation workflow with lenient processing"""
        try:
            # Validate paths
            dataset_path = os.path.abspath(dataset_path)
            if not os.path.exists(dataset_path):
                logger.error(f"❌ Dataset path does not exist: {dataset_path}")
                return False

            logger.info(f"🚀 Starting embedding generation from: {dataset_path}")

            # Get all identity folders
            identity_folders = []
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    identity_folders.append(item_path)

            if not identity_folders:
                logger.error("❌ No valid identity folders found")
                return False

            logger.info(f"📊 Found {len(identity_folders)} identity folders")
            self.monitor.update_stats(total_identities=len(identity_folders))

            # Process identities with threading
            embeddings_dict = {}

            with ThreadPoolExecutor(max_workers=ENTERPRISE_CONFIG['processing']['workers']) as executor:
                futures = [executor.submit(self.process_identity, path) for path in identity_folders]

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        embeddings_dict.update(result)
                        self.monitor.update_stats(successful_identities=1)
                        identity_name = list(result.keys())[0]
                        embedding_count = len(result[identity_name])
                        logger.info(f"✅ Processed {identity_name}: {embedding_count} embeddings")

            # Validate results
            if not embeddings_dict:
                logger.error("❌ No embeddings generated - check dataset quality and paths")
                return False

            # Save embeddings with backup
            if ENTERPRISE_CONFIG['enterprise']['save_backup'] and os.path.exists(output_path):
                backup_name = output_path.replace('.pkl', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
                try:
                    shutil.copy2(output_path, backup_name)
                    logger.info(f"📦 Created backup: {backup_name}")
                except Exception as e:
                    logger.warning(f"Backup failed: {e}")

            # Save embeddings
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Verify save operation
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                total_embeddings = sum(len(emb_list) for emb_list in embeddings_dict.values())
                logger.info(
                    f"🎉 SUCCESS: Generated {len(embeddings_dict)} identities with {total_embeddings} embeddings")

                # Print summary
                summary = self.monitor.get_summary()
                logger.info(f"📈 Processing summary:")
                logger.info(f"  Success rate: {summary['success_rate']:.1f}%")
                logger.info(f"  Embedding rate: {summary['embedding_rate']:.1f}%")
                logger.info(f"  Processing time: {summary['runtime_seconds']:.1f}s")

                return True
            else:
                logger.error("❌ Failed to save embeddings or file is empty")
                return False

        except Exception as e:
            logger.error(f"💥 Critical failure in embedding generation: {e}")
            return False


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Enterprise Embedding Generation')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--output', required=True, help='Output embeddings path')
    args = parser.parse_args()

    generator = EnterpriseEmbeddingGenerator()
    success = generator.generate_embeddings(args.dataset, args.output)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
