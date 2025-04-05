from pydantic import BaseModel
from datetime import datetime
from typing import Optional

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
    employee_name: str
    image_path: Optional[str] = None

class AttendanceResponse(AttendanceCreate):
    id: int
    time_in: datetime
    class Config:
        from_attributes = True
