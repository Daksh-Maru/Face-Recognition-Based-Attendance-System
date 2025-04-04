from pydantic import BaseModel
from datetime import datetime
from typing import Optional

#Schema for Employee creation
class EmployeeCreate(BaseModel):
    name: str
    email: str

#Schema for Employee response
class EmployeeResponse(EmployeeCreate):
    id: int

    class Config:
        from_attributes = True  # This allows Pydantic to work with SQLAlchemy models

#Schema for Attendance creation
class AttendanceCreate(BaseModel):
    employee_id: int
    employee_name: str
    image_path: Optional[str] = None

#Schema for Attendance response
class AttendanceResponse(AttendanceCreate):
    id: int
    time_in: datetime

    class Config:
        from_attributes = True  # This allows Pydantic to work with SQLAlchemy models

