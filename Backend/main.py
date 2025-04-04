from fastapi import FastAPI
from routes import attendence, employees
import database, models

# Create database tables (if not already created)
database.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

app.include_router(employees.router)
app.include_router(attendence.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Attendance Management System"}