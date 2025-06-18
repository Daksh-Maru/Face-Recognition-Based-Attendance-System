from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any

# Employee Schemas
class EmployeeCreate(BaseModel):
    name: str
    email: str

class EmployeeResponse(EmployeeCreate):
    id: int
    
    class Config:
        from_attributes = True

# Attendance Schemas
class AttendanceCreate(BaseModel):
    employee_id: int
    image_path: Optional[str] = None
    # No need for employee_name since we'll fetch it from the database


class AttendanceResponse(AttendanceCreate):
    id: int
    time_in: datetime 
    
    class Config:
        from_attributes = True

# For the employee attendance endpoint
class Attendance(BaseModel):
    id: int
    employee_id: int
    employee_name: str
    image_path: Optional[str] = None
    time_in: datetime
    
    class Config:
        from_attributes = True

class MonthlyAttendanceStats(BaseModel):
    year: int
    month: int
    total_attendance: int
    unique_employees: int
    daily_breakdown: List[Dict[str, Any]]
    
    class Config:
        from_attributes = True  # Updated from orm_mode which is deprecated
