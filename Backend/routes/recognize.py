from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from services.utils import load_image_from_bytes, preprocess_face
from services.detection import detect_face
from services.recognition import FaceRecognizer
from database import get_db
import crud
import pickle, os
from datetime import datetime
import schema
import numpy as np
import cv2
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Recognition"])

# Load stored embeddings
try:
    with open("assets/embeddings.pkl", "rb") as f:
        stored_embeddings = pickle.load(f)
except FileNotFoundError:
    stored_embeddings = {}

@router.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """
    Time Analysis (milliseconds me difference)
    """
    recognizer = FaceRecognizer(embeddings_path="assets/embeddings.pkl")


    # Read image bytes
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Detect face
    face = detect_face(img)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected.")

    # Get embedding
    embedding = recognizer.get_embedding(face)
    if embedding is None:
        raise HTTPException(status_code=400, detail="Failed to generate face embedding.")

    # Predict identity
    identity, confidence = recognizer.recognize(embedding)

    return JSONResponse(content={"identity": identity, "confidence": confidence})


# @router.post("/recognize", response_model=schema.AttendanceResponse)
# async def recognize_face(image: UploadFile = File(...), db: Session = Depends(get_db)):
#     # Step 1: Load image and detect face
#     image_bytes = await image.read()
#     image_np = load_image_from_bytes(image_bytes)
#     face = detect_face(image_np)
#     if face is None:
#         raise HTTPException(status_code=400, detail="No face detected.")

#     # Step 2: Get embedding and predict identity
#     face = preprocess_face(face)
#     embedding = get_embedding(face)
#     identity = predict_face(embedding, stored_embeddings)
#     if identity == "Unknown":
#         raise HTTPException(status_code=404, detail="Face not recognized.")

#     # Step 3: Save image (optional)
#     save_dir = "uploads"
#     os.makedirs(save_dir, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#     image_path = os.path.join(save_dir, f"{identity}_{timestamp}.jpg")
#     with open(image_path, "wb") as f:
#         f.write(image_bytes)

#     # Step 4: Log attendance to DB
#     new_attendance = crud.create_attendance(
#         db,
#         employee_id=identity,      # assumes identity = ID from stored embeddings
#         employee_name=identity,    # or fetch actual name using a lookup if needed
#         image_path=image_path
#     )

#     # Step 5: Return the stored attendance info
#     return new_attendance