from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import crud, database, schema

router = APIRouter()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create a new attendance record
@router.post("/attendance/", response_model=schema.AttendanceResponse)
def create_attendance(attendance: schema.AttendanceCreate, db: Session = Depends(get_db)):
    db_attendance = crud.create_attendance(db, attendance.employee_id, attendance.employee_name, attendance.image_path)
    return db_attendance

# Get attendance records by date
@router.get("/attendance/{date}", response_model=list[schema.AttendanceResponse])
def get_attendance_by_date(date: str, db: Session = Depends(get_db)):
    record = crud.get_attendance_by_date(db, date)
    if not record:
        raise HTTPException(status_code=404, detail="No attendance record found for this date")
    return record