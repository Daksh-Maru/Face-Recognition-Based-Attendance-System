from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

import crud, schema, database

router = APIRouter(tags=["Employees"])  # This helps group routes in Swagger UI

# Dependency to get DB session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create a new employee
@router.post("/employees/", response_model=schema.EmployeeResponse)
def create_employee(employee: schema.EmployeeCreate, db: Session = Depends(get_db)):
    return crud.create_employee(db=db, employee=employee)

# Get employee by ID
@router.get("/employees/{employee_id}", response_model=schema.EmployeeResponse)
def get_employee(employee_id: int, db: Session = Depends(get_db)):
    db_employee = crud.get_employee(db, employee_id)
    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return db_employee

# Get all employees
@router.get("/employees/", response_model=List[schema.EmployeeResponse])
def get_all_employees(db: Session = Depends(get_db)):
    return crud.get_all_employees(db)
