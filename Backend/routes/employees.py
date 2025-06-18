from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database import get_db 
import crud, schema, database

router = APIRouter(
    prefix="/employees",
    tags=["Employees"],
    responses={404: {"description": "Employee not found"}}
)

@router.post("/", response_model=schema.EmployeeResponse, status_code=status.HTTP_201_CREATED)
def create_employee(employee: schema.EmployeeCreate, db: Session = Depends(get_db)):
    """
    Create a new employee with the following information:
    - name: Employee's full name
    - email: Employee's email address
    """
    # Check if employee with this email already exists
    existing_employees = crud.get_all_employees(db)
    for existing_employee in existing_employees:
        if existing_employee.email == employee.email:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
    
    return crud.create_employee(db=db, employee=employee)

@router.get("/{employee_id}", response_model=schema.EmployeeResponse)
def get_employee(employee_id: int, db: Session = Depends(get_db)):
    """
    Get an employee by their ID.
    """
    db_employee = crud.get_employee(db, employee_id)
    if not db_employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    return db_employee

@router.get("/", response_model=List[schema.EmployeeResponse])
def get_all_employees(db: Session = Depends(get_db)):
    """
    Get a list of all employees.
    """
    return crud.get_all_employees(db)
