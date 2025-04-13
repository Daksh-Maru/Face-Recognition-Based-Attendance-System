from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from services.detection import detect_face
from services.recognition import get_embedding, predict_face
import numpy as np
import pickle
import cv2

router = APIRouter(tags=["Recognition"])

# Load stored embeddings
try:
    with open("assets/embeddings.pkl", "rb") as f:
        stored_embeddings = pickle.load(f)
except FileNotFoundError:
    stored_embeddings = {}

@router.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    # Read image bytes
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Detect face
    face = detect_face(img)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected.")

    # Get embedding
    embedding = get_embedding(face)

    # Predict identity
    identity = predict_face(embedding, stored_embeddings)

    return JSONResponse(content={"identity": identity})
