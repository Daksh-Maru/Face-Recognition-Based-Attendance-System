from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from models import Employee, Attendance
import schema
from typing import Optional
from datetime import datetime, date

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

def get_employee_attendance(db: Session, employee_id: int):
    return db.query(Attendance).filter(
        Attendance.employee_id == employee_id
    ).order_by(Attendance.time_in.desc()).all()

def get_employee_attendance_with_filters(
    db: Session, 
    employee_id: int, 
    start_date: Optional[date] = None, 
    end_date: Optional[date] = None,
    skip: int = 0,
    limit: int = 100
):
    query = db.query(Attendance).filter(
        Attendance.employee_id == employee_id
    )
    
    if start_date:
        start_of_day = datetime.combine(start_date, datetime.min.time())
        query = query.filter(Attendance.time_in >= start_of_day)
    
    if end_date:
        end_of_day = datetime.combine(end_date, datetime.max.time())
        query = query.filter(Attendance.time_in <= end_of_day)
    
    return query.order_by(Attendance.time_in.desc()).offset(skip).limit(limit).all()

def get_attendance(db: Session, attendance_id: int):
    return db.query(Attendance).filter(Attendance.id == attendance_id).first()

def delete_attendance(db: Session, attendance_id: int):
    db_attendance = db.query(Attendance).filter(Attendance.id == attendance_id).first()
    if db_attendance:
        db.delete(db_attendance)
        db.commit()
    return db_attendance

def get_monthly_attendance_stats(db: Session, year: int, month: int):
    # Count total attendance for the month
    total_count = db.query(func.count(Attendance.id)).filter(
        extract('year', Attendance.time_in) == year,
        extract('month', Attendance.time_in) == month
    ).scalar()
    
    # Count unique employees
    unique_employees = db.query(func.count(func.distinct(Attendance.employee_id))).filter(
        extract('year', Attendance.time_in) == year,
        extract('month', Attendance.time_in) == month
    ).scalar()
    
    # Get daily breakdown
    daily_counts = db.query(
        func.date(Attendance.time_in).label('date'),
        func.count(Attendance.id).label('count')
    ).filter(
        extract('year', Attendance.time_in) == year,
        extract('month', Attendance.time_in) == month
    ).group_by(
        func.date(Attendance.time_in)
    ).all()
    
    daily_breakdown = [{"date": str(day.date), "count": day.count} for day in daily_counts]
    
    return {
        "year": year,
        "month": month,
        "total_attendance": total_count,
        "unique_employees": unique_employees,
        "daily_breakdown": daily_breakdown
    }
