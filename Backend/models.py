from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from database import Base
import datetime


# Employee Table
class Employee(Base):
    __tablename__ = 'employees' # Table name in the database
    
    id = Column(Integer, primary_key=True, index=True)  # Unique ID for each employee
    name = Column(String, nullable=False)  # Name of the employee
    email = Column(String, unique=True, nullable=False) # Unique email for each employee
    attendances = relationship("Attendance", back_populates="employee") # Relationship to Attendance table


# Attendance Table
class Attendance(Base): 
    __tablename__ = 'attendance' # Table name in the database
    
    id = Column(Integer, primary_key=True, index=True) # Unique ID for each attendance record
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False) # Foreign key to Employee table
    employee_name = Column(String, nullable=False) # Name of the employee
    time_in = Column(DateTime, default=func.now()) # Time of check-in
    image_path = Column(String, nullable=True) # Path to captured image
    
    employee = relationship("Employee", back_populates="attendances") # Relationship to Employee table