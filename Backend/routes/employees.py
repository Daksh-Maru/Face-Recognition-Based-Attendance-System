from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import crud, schema, database

router = APIRouter()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()
        

#Create a new employee
@router.post("/employees/", response_model=schema.EmployeeResponse)
def create_employee(employee: schema.EmployeeCreate, db: Session = Depends(get_db)):
    db_employee = crud.create_employee(db=db, employee=employee)
    return db_employee

#Get employee by ID
@router.get("/employees/{employee_id}", response_model=schema.EmployeeResponse)
def get_employee(employee_id: int, db: Session = Depends(get_db)):
    db_employee = crud.get_employee(db, employee_id)
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return db_employee

