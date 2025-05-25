# recognition.py

import cv2
import torch
import numpy as np
import pickle
import os
from facenet_pytorch import InceptionResnetV1
from detection import detect_face  # Your YOLOv8-face detection function
from PIL import Image
from torchvision import transforms
#from occlusion_detection import OcclusionDetector

# Initialize FaceNet model with error handling
try:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print(f"✅ FaceNet model loaded on {device}")
except Exception as e:
    print(f"❌ Error loading FaceNet model: {e}")
    model = None

# Load embeddings with error handling
stored_embeddings = {}
embeddings_path = '../assets/embeddings.pkl'

try:
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            stored_embeddings = pickle.load(f)
        print(f"✅ Loaded {len(stored_embeddings)} stored embeddings")
    else:
        print(f"⚠️ Embeddings file not found at {embeddings_path}")
        print("Recognition will work but no identities will be matched.")
except Exception as e:
    print(f"❌ Error loading embeddings: {e}")
    stored_embeddings = {}

# Transform function to prepare face image for FaceNet
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_embedding(image):
    """Get embedding from image using face detection and FaceNet"""
    try:
        if model is None:
            return None

        # Detect face using YOLOv8
        face_result = detect_face(image)

        # Handle different return formats from detect_face
        if isinstance(face_result, tuple):
            face_img = face_result[0]  # First element is the face image
        else:
            face_img = face_result

        if face_img is None:
            return None

        # Ensure face_img is valid
        if face_img.size == 0:
            return None

        # Convert to PIL Image
        if isinstance(face_img, np.ndarray):
            # Ensure the image is in the correct format (0-255, uint8)
            if face_img.dtype != np.uint8:
                face_img = (face_img * 255).astype(np.uint8) if face_img.max() <= 1.0 else face_img.astype(np.uint8)
            face_pil = Image.fromarray(face_img)
        else:
            face_pil = face_img

        # Apply transforms
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Generate embedding
        with torch.no_grad():
            embedding = model(face_tensor)

        return embedding.detach().cpu().numpy()[0]  # Return flattened array

    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None


def get_embedding_with_occlusion_handling(face_pixels, occlusion_mask=None):
    """Generate embedding from face image with occlusion handling"""
    try:
        if model is None:
            return None

        # Ensure the input is a NumPy array
        if isinstance(face_pixels, Image.Image):
            face_pixels = np.array(face_pixels)
        elif not isinstance(face_pixels, np.ndarray):
            print("Invalid face_pixels format")
            return None

        # Validate face_pixels
        if face_pixels.size == 0:
            return None

        # Ensure correct data type
        if face_pixels.dtype != np.uint8:
            if face_pixels.max() <= 1.0:
                face_pixels = (face_pixels * 255).astype(np.uint8)
            else:
                face_pixels = face_pixels.astype(np.uint8)

        # If we have an occlusion mask, apply it to focus on non-occluded regions
        if occlusion_mask is not None:
            # Ensure occlusion_mask is the right shape
            if len(occlusion_mask.shape) == 3:
                occlusion_mask = np.mean(occlusion_mask, axis=2)

            # Resize mask to match face if needed
            if occlusion_mask.shape != face_pixels.shape[:2]:
                occlusion_mask = cv2.resize(occlusion_mask, (face_pixels.shape[1], face_pixels.shape[0]))

            # Check if the face is mostly occluded
            occlusion_percentage = np.mean(occlusion_mask)
            if occlusion_percentage > 0.7:  # More than 70% occluded
                print(f"Face too occluded ({occlusion_percentage:.1%})")
                return None

            # Apply mask to image (multiply by inverse mask to keep non-occluded regions)
            masked_face = face_pixels.copy()

            # Normalize occlusion mask to 0-1 range
            if occlusion_mask.max() > 1.0:
                occlusion_mask = occlusion_mask / 255.0

            # Apply mask to each channel
            for c in range(min(3, face_pixels.shape[2] if len(face_pixels.shape) == 3 else 1)):
                if len(face_pixels.shape) == 3:
                    masked_face[:, :, c] = masked_face[:, :, c] * (1 - occlusion_mask)
                else:
                    masked_face = masked_face * (1 - occlusion_mask)

            # Use the masked face
            face_pil = Image.fromarray(masked_face)
        else:
            face_pil = Image.fromarray(face_pixels)

        # Apply transforms
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Generate embedding
        with torch.no_grad():
            embedding = model(face_tensor)

        return embedding.detach().cpu().numpy()[0]

    except Exception as e:
        print(f"Error in get_embedding_with_occlusion_handling: {e}")
        return None


def predict_face(embedding, stored_embeddings=None, threshold=1.0):
    """
    Compare the embedding with stored embeddings to find a match

    Args:
        embedding: Face embedding of the query face
        stored_embeddings: Dictionary of stored embeddings
        threshold: Distance threshold for recognition

    Returns:
        Identity of the matched face or "Unknown"
    """
    try:
        if embedding is None:
            return "Unknown"

        if stored_embeddings is None:
            stored_embeddings = globals().get('stored_embeddings', {})

        if len(stored_embeddings) == 0:
            return "Unknown"

        min_dist = float('inf')
        identity = "Unknown"

        for name, emb in stored_embeddings.items():
            try:
                # Ensure embeddings are the same shape
                if embedding.shape != emb.shape:
                    print(f"Shape mismatch for {name}: {embedding.shape} vs {emb.shape}")
                    continue

                # Calculate Euclidean distance
                dist = np.linalg.norm(embedding - emb)

                if dist < min_dist and dist < threshold:
                    min_dist = dist
                    identity = name

            except Exception as e:
                print(f"Error comparing with {name}: {e}")
                continue

        return identity

    except Exception as e:
        print(f"Error in predict_face: {e}")
        return "Unknown"


# In recognition.py - Remove or comment out this import:
# from occlusion_detection import OcclusionDetector

# Update the predict_face_with_occlusion_handling function:
def predict_face_with_occlusion_handling(face_pixels, occlusion_mask=None, stored_embeddings=None, threshold=1.0):
    """Recognize face with heuristic occlusion handling"""
    try:
        if stored_embeddings is None:
            stored_embeddings = globals().get('stored_embeddings', {})

        # Get embedding with occlusion handling
        embedding = get_embedding_with_occlusion_handling(face_pixels, occlusion_mask)

        # If embedding generation failed, return Unknown
        if embedding is None:
            return "Unknown"

        # Use adaptive threshold based on occlusion level
        adaptive_threshold = threshold
        if occlusion_mask is not None:
            occlusion_percentage = np.mean(occlusion_mask)
            # Increase threshold for occluded faces to be more lenient
            if occlusion_percentage > 0.3:
                adaptive_threshold = threshold * 1.2
            elif occlusion_percentage > 0.5:
                adaptive_threshold = threshold * 1.5

        # Match embedding to known identities
        min_dist = float("inf")
        identity = "Unknown"

        for name, emb in stored_embeddings.items():
            try:
                if embedding.shape != emb.shape:
                    continue

                dist = np.linalg.norm(embedding - emb)
                if dist < min_dist and dist < adaptive_threshold:
                    min_dist = dist
                    identity = name

            except Exception as e:
                print(f"Error comparing with {name}: {e}")
                continue

        return identity

    except Exception as e:
        print(f"Error in predict_face_with_occlusion_handling: {e}")
        return "Unknown"


def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    try:
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0


def predict_face_cosine(embedding, stored_embeddings=None, threshold=0.7):
    """
    Predict face using cosine similarity instead of Euclidean distance

    Args:
        embedding: Face embedding of the query face
        stored_embeddings: Dictionary of stored embeddings
        threshold: Cosine similarity threshold (higher = more similar)

    Returns:
        Identity of the matched face or "Unknown"
    """
    try:
        if embedding is None:
            return "Unknown"

        if stored_embeddings is None:
            stored_embeddings = globals().get('stored_embeddings', {})

        if len(stored_embeddings) == 0:
            return "Unknown"

        max_similarity = -1
        identity = "Unknown"

        for name, emb in stored_embeddings.items():
            try:
                if embedding.shape != emb.shape:
                    continue

                similarity = cosine_similarity(embedding, emb)

                if similarity > max_similarity and similarity > threshold:
                    max_similarity = similarity
                    identity = name

            except Exception as e:
                print(f"Error comparing with {name}: {e}")
                continue

        return identity

    except Exception as e:
        print(f"Error in predict_face_cosine: {e}")
        return "Unknown"


def test_recognition():
    """Test the recognition pipeline"""
    try:
        print("🧪 Testing face recognition pipeline...")

        # Create a synthetic test image
        test_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)

        # Test embedding generation
        embedding = get_embedding(test_img)

        if embedding is not None:
            print(f"✅ Embedding generated: shape {embedding.shape}")

            # Test prediction
            identity = predict_face(embedding)
            print(f"✅ Prediction: {identity}")
        else:
            print("⚠️ No embedding generated (expected for random image)")

        # Test with occlusion handling
        test_mask = np.random.rand(100, 100) > 0.7  # Random occlusion mask
        test_face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        identity_with_occlusion = predict_face_with_occlusion_handling(test_face, test_mask)
        print(f"✅ Prediction with occlusion: {identity_with_occlusion}")

        print("🎉 Recognition pipeline test completed!")

    except Exception as e:
        print(f"❌ Recognition test failed: {e}")


if __name__ == "__main__":
    test_recognition()
