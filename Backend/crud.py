from sqlalchemy.orm import Session
from models import Employee, Attendance
import schema
from datetime import datetime

def create_employee(db: Session, employee: schema.EmployeeCreate):
    db_employee = Employee(name=employee.name, email=employee.email)
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)
    return db_employee

def get_employee(db: Session, employee_id: int):
    return db.query(Employee).filter(Employee.id == employee_id).first()

def get_all_employees(db: Session):
    return db.query(Employee).all()

def create_attendance(db: Session, employee_id: int, employee_name: str, image_path: str):
    db_attendance = Attendance(employee_id=employee_id, employee_name=employee_name, image_path=image_path)
    db.add(db_attendance)
    db.commit()
    db.refresh(db_attendance)
    return db_attendance

def get_attendance_by_date(db: Session, date_str: str):
    # Convert the string to a date object; expect "YYYY-MM-DD" format
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return []  # Or raise an exception if appropriate

    start_of_day = datetime.combine(date_obj, datetime.min.time())
    end_of_day = datetime.combine(date_obj, datetime.max.time())
    return db.query(Attendance).filter(Attendance.time_in.between(start_of_day, end_of_day)).all()
