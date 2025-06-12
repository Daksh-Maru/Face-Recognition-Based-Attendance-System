# routes/recognize.py

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from services.detection import detect_face
from services.recognition import get_embedding, predict_face_with_occlusion_handling, predict_face
from services.utils import load_image_from_bytes
import numpy as np
import pickle
import cv2
import os
import logging
import time

# Setup logging for production
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Recognition"])

# Global variables for caching
stored_embeddings = {}
embeddings_last_loaded = 0
CACHE_TIMEOUT = 300  # 5 minutes


def load_embeddings():
    """Load embeddings with improved error handling and caching"""
    global stored_embeddings, embeddings_last_loaded

    # Check if we need to reload (cache timeout or first load)
    current_time = time.time()
    if current_time - embeddings_last_loaded < CACHE_TIMEOUT and stored_embeddings:
        return stored_embeddings

    embeddings_paths = [
        "assets/embeddings.pkl",
        "../assets/embeddings.pkl",
        os.path.join(os.path.dirname(__file__), "..", "assets", "embeddings.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "..", "assets", "embeddings.pkl")
    ]

    for path in embeddings_paths:
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    embeddings = pickle.load(f)

                # Validate embeddings structure
                if not isinstance(embeddings, dict):
                    logger.warning(f"Invalid embeddings format in {path}")
                    continue

                # Count total embeddings
                total_embeddings = 0
                valid_identities = {}

                for identity, emb_data in embeddings.items():
                    if isinstance(emb_data, list):
                        valid_embs = [e for e in emb_data if isinstance(e, np.ndarray) and e.size == 512]
                        if valid_embs:
                            valid_identities[identity] = valid_embs
                            total_embeddings += len(valid_embs)
                    elif isinstance(emb_data, np.ndarray) and emb_data.size == 512:
                        valid_identities[identity] = [emb_data]
                        total_embeddings += 1

                stored_embeddings = valid_identities
                embeddings_last_loaded = current_time

                logger.info(
                    f"✅ Loaded {total_embeddings} embeddings for {len(valid_identities)} identities from {path}")
                return stored_embeddings

        except Exception as e:
            logger.warning(f"❌ Error loading from {path}: {e}")
            continue

    logger.warning("⚠️ No valid embeddings file found, returning empty dict")
    stored_embeddings = {}
    embeddings_last_loaded = current_time
    return {}


# Load embeddings on startup
stored_embeddings = load_embeddings()


@router.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """Enhanced face recognition endpoint with better error handling"""
    start_time = time.time()

    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "Please upload a valid image file", "confidence": 0.0}
            )

        # Read image bytes
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")

        # Enhanced image loading using utils
        try:
            img_rgb = load_image_from_bytes(contents)
            if img_rgb is None:
                # Fallback to direct OpenCV decoding
                np_img = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if img is None:
                    raise HTTPException(status_code=400, detail="Invalid image format")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Image loading failed: {e}")
            raise HTTPException(status_code=400, detail="Failed to process image")

        # Reload embeddings if needed
        current_embeddings = load_embeddings()
        if not current_embeddings:
            return JSONResponse(
                status_code=200,
                content={
                    "identity": "Unknown",
                    "reason": "No stored embeddings available",
                    "confidence": 0.0
                }
            )

        # Face detection with better error handling
        try:
            detection_result = detect_face(img_rgb, is_webcam=False)

            # Handle different return formats from detection
            if isinstance(detection_result, tuple) and len(detection_result) >= 3:
                face, occlusion_info, detection_info = detection_result
            elif isinstance(detection_result, tuple) and len(detection_result) == 2:
                face, detection_info = detection_result
                occlusion_info = None
            else:
                face = detection_result
                detection_info = "basic_detection"
                occlusion_info = None

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return JSONResponse(
                status_code=200,
                content={
                    "identity": "Unknown",
                    "reason": "Face detection failed",
                    "confidence": 0.0,
                    "error": str(e)
                }
            )

        # Check if face was detected
        if face is None:
            return JSONResponse(
                status_code=200,
                content={
                    "identity": "Unknown",
                    "reason": "No face detected in image",
                    "confidence": 0.0,
                    "detection_info": str(detection_info)
                }
            )

        # Validate face image
        if not isinstance(face, np.ndarray) or face.size == 0:
            return JSONResponse(
                status_code=200,
                content={
                    "identity": "Unknown",
                    "reason": "Invalid face detected",
                    "confidence": 0.0
                }
            )

        # Ensure face is in correct format
        if face.dtype != np.uint8:
            if face.max() <= 1.0:
                face = (face * 255).astype(np.uint8)
            else:
                face = np.clip(face, 0, 255).astype(np.uint8)

        logger.debug(f"🔍 Processing face: shape={face.shape}, dtype={face.dtype}")

        # Enhanced recognition with proper error handling
        try:
            # Use the corrected recognition function with occlusion handling
            identity, confidence = predict_face_with_occlusion_handling(
                face,
                occlusion_info=occlusion_info,
                stored_embeddings_param=current_embeddings
            )

            # Validate recognition results
            if not isinstance(confidence, (int, float)):
                confidence = 0.0

            confidence = max(0.0, min(1.0, float(confidence)))  # Clamp between 0 and 1

            # Prepare response
            response_data = {
                "identity": str(identity),
                "confidence": round(confidence, 3),
                "processing_time": round(time.time() - start_time, 3),
                "detection_method": str(detection_info) if detection_info else "unknown"
            }

            # Add occlusion info if available
            if occlusion_info is not None and isinstance(occlusion_info, dict):
                occlusion_percentage = occlusion_info.get('total_occlusion_percentage', 0.0)
                if isinstance(occlusion_percentage, (int, float)):
                    response_data["occlusion_info"] = {
                        "detected": occlusion_percentage > 0.1,
                        "percentage": round(float(occlusion_percentage) * 100, 1)
                    }

            # Add debug info in development
            if logger.level <= logging.DEBUG:
                response_data["debug"] = {
                    "face_shape": face.shape,
                    "available_identities": len(current_embeddings)
                }

            logger.info(
                f"Recognition result: {identity} (confidence: {confidence:.3f}, time: {time.time() - start_time:.3f}s)")
            return JSONResponse(content=response_data)

        except Exception as e:
            logger.error(f"❌ Recognition failed: {e}")
            return JSONResponse(
                status_code=200,
                content={
                    "identity": "Unknown",
                    "reason": "Recognition processing failed",
                    "confidence": 0.0,
                    "error": str(e)
                }
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in recognize_face: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/embeddings/status")
async def embeddings_status():
    """Get detailed status of loaded embeddings"""
    current_embeddings = load_embeddings()

    # Calculate statistics
    total_embeddings = 0
    identity_stats = {}

    for identity, emb_list in current_embeddings.items():
        count = len(emb_list) if isinstance(emb_list, list) else 1
        total_embeddings += count
        identity_stats[identity] = count

    return JSONResponse(content={
        "status": "loaded" if current_embeddings else "empty",
        "total_identities": len(current_embeddings),
        "total_embeddings": total_embeddings,
        "avg_embeddings_per_identity": round(total_embeddings / max(1, len(current_embeddings)), 2),
        "identities": list(current_embeddings.keys()),
        "identity_stats": identity_stats,
        "last_loaded": embeddings_last_loaded,
        "cache_valid": time.time() - embeddings_last_loaded < CACHE_TIMEOUT
    })


@router.post("/embeddings/reload")
async def reload_embeddings():
    """Force reload embeddings from file"""
    global embeddings_last_loaded
    embeddings_last_loaded = 0  # Force reload

    new_embeddings = load_embeddings()
    total_embeddings = sum(len(emb_list) if isinstance(emb_list, list) else 1
                           for emb_list in new_embeddings.values())

    return JSONResponse(content={
        "status": "reloaded",
        "total_identities": len(new_embeddings),
        "total_embeddings": total_embeddings,
        "timestamp": time.time()
    })


@router.post("/test/recognition")
async def test_recognition_pipeline():
    """Test endpoint to verify the recognition pipeline is working"""
    try:
        current_embeddings = load_embeddings()

        # Test embedding generation
        test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        test_embedding = get_embedding(test_face)

        return JSONResponse(content={
            "pipeline_status": "working",
            "embeddings_loaded": len(current_embeddings) > 0,
            "embedding_generation": test_embedding is not None,
            "available_identities": list(current_embeddings.keys())[:5],  # First 5
            "total_identities": len(current_embeddings)
        })

    except Exception as e:
        return JSONResponse(content={
            "pipeline_status": "error",
            "error": str(e)
        })


@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        current_embeddings = load_embeddings()
        return JSONResponse(content={
            "status": "healthy",
            "embeddings_loaded": len(current_embeddings) > 0,
            "total_identities": len(current_embeddings),
            "timestamp": time.time()
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )
