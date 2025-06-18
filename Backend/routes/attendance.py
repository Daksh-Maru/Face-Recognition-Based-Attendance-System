from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import crud, database, schema
from services.utils import load_image_from_bytes, preprocess_face
from services.detection import detect_face
import pickle
import os
from datetime import datetime, date
from database import get_db 


router = APIRouter(
    prefix="/attendance",
    tags=["Attendance"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=schema.AttendanceResponse, status_code=201,
             summary="Create attendance record",
             description="Create a new attendance record for an employee")
def create_attendance(attendance: schema.AttendanceCreate, db: Session = Depends(get_db)):
    """
    Create a new attendance record with the following information:
    
    - **employee_id**: ID of the employee
    - **image_path**: Path to the stored attendance image
    """
    # Check if employee exists
    employee = crud.get_employee(db, attendance.employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail=f"Employee with ID {attendance.employee_id} not found")
    
    db_attendance = crud.create_attendance(db, attendance.employee_id, employee.name, attendance.image_path)
    return db_attendance


@router.get("/by-date/{date}", response_model=List[schema.Attendance],
            summary="Get attendance by date",
            description="Retrieve all attendance records for a specific date")
def get_attendance_by_date(date: str, db: Session = Depends(get_db)):
    """
    Get all attendance records for a specific date.
    
    Date format should be YYYY-MM-DD.
    """
    try:
        # Validate date format
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    record = crud.get_attendance_by_date(db, date)
    if not record:
        return []  # Return empty list instead of 404 error
    return record


@router.get("/today", response_model=List[schema.Attendance],
            summary="Get today's attendance",
            description="Retrieve all attendance records for the current day")
def get_today_attendance(db: Session = Depends(get_db)):
    """
    Get all attendance records for the current date.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    record = crud.get_attendance_by_date(db, today)
    return record or []


@router.get("/employee/{employee_id}", response_model=List[schema.AttendanceResponse],
            summary="Get employee attendance history",
            description="Retrieve attendance history for a specific employee")
def get_employee_attendance_records(
    employee_id: int, 
    start_date: Optional[date] = Query(None, description="Filter by start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="Filter by end date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    db: Session = Depends(get_db)
):
    """
    Get attendance history for a specific employee.
    
    Optional filters:
    - **start_date**: Filter records from this date
    - **end_date**: Filter records until this date
    - **limit**: Maximum number of records to return
    - **skip**: Number of records to skip (for pagination)
    """
    # Check if employee exists
    employee = crud.get_employee(db, employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail=f"Employee with ID {employee_id} not found")
    
    attendance_records = crud.get_employee_attendance_with_filters(
        db, employee_id, start_date, end_date, skip, limit
    )
    
    return attendance_records


@router.get("/stats/monthly/{year}/{month}", response_model=schema.MonthlyAttendanceStats,
            summary="Get monthly attendance statistics",
            description="Retrieve attendance statistics for a specific month")
def get_monthly_attendance_stats(year: int, month: int, db: Session = Depends(get_db)):
    """
    Get attendance statistics for a specific month.
    
    Returns:
    - Total attendance count
    - Unique employee count
    - Daily breakdown
    """
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    
    if not (2000 <= year <= 2100):
        raise HTTPException(status_code=400, detail="Year must be between 2000 and 2100")
    
    stats = crud.get_monthly_attendance_stats(db, year, month)
    if not stats or stats["total_attendance"] == 0:
        return {
            "year": year,
            "month": month,
            "total_attendance": 0,
            "unique_employees": 0,
            "daily_breakdown": []
        }
    
    return stats


@router.delete("/{attendance_id}", status_code=200,
               summary="Delete attendance record",
               description="Delete a specific attendance record by ID")
def delete_attendance(attendance_id: int, db: Session = Depends(get_db)):
    """
    Delete an attendance record by ID.
    
    Returns the deleted record on successful deletion.
    """
    attendance = crud.get_attendance(db, attendance_id)
    if not attendance:
        raise HTTPException(status_code=404, detail=f"Attendance record with ID {attendance_id} not found")
    
    deleted_record = crud.delete_attendance(db, attendance_id)
    
    return {
        "message": "Record successfully deleted",
        "deleted_record": {
            "id": deleted_record.id,
            "employee_id": deleted_record.employee_id,
            "employee_name": deleted_record.employee_name,
            "time_in": deleted_record.time_in,
            "image_path": deleted_record.image_path
        }
    }
