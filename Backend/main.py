from fastapi import FastAPI
from routes import attendance, employees, recognize  # Ensure naming is consistent!
import database, models

# Create database tables (if not already created)
database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="Face Recognition Attendance System",
    description="An API to manage attendance using facial recognition.",
    version="1.0.0"
)

# Include routers (with optional prefixes)
app.include_router(employees.router)
app.include_router(attendance.router)
# app.include_router(recognize.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Attendance Management System"}
