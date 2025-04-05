from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import crud, database, schema
from services.utils import load_image_from_bytes, preprocess_face
from services.detection import detect_face
from services.recognition import get_embedding, predict_face
import pickle
import os
from datetime import datetime

router = APIRouter()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Load precomputed embeddings
with open("assets/embeddings.pkl", "rb") as f:
    stored_embeddings = pickle.load(f)

@router.post("/attendance/recognize", response_model=schema.AttendanceResponse)
async def recognize_and_create_attendance(image: UploadFile = File(...), db: Session = Depends(get_db)):
    image_bytes = await image.read()
    image_np = load_image_from_bytes(image_bytes)

    # Detect face
    face = detect_face(image_np)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    # Preprocess & get embedding
    face = preprocess_face(face)
    embedding = get_embedding(face)

    # Predict identity
    identity = predict_face(embedding, stored_embeddings)
    if identity == "Unknown":
        raise HTTPException(status_code=404, detail="Face not recognized.")

    # Save uploaded image (optional for records)
    save_dir = "uploads"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_path = os.path.join(save_dir, f"{identity}_{timestamp}.jpg")
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # Record attendance
    new_attendance = crud.create_attendance(
        db,
        employee_id=identity,  # assuming name == id from training data
        employee_name=identity,
        image_path=image_path
    )

    return new_attendance

# Existing: Create via form data
@router.post("/attendance/", response_model=schema.AttendanceResponse)
def create_attendance(attendance: schema.AttendanceCreate, db: Session = Depends(get_db)):
    db_attendance = crud.create_attendance(db, attendance.employee_id, attendance.employee_name, attendance.image_path)
    return db_attendance

# Existing: Get by date
@router.get("/attendance/{date}", response_model=list[schema.AttendanceResponse])
def get_attendance_by_date(date: str, db: Session = Depends(get_db)):
    record = crud.get_attendance_by_date(db, date)
    if not record:
        raise HTTPException(status_code=404, detail="No attendance record found for this date")
    return record
