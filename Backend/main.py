from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import attendance, employees, recognize  # Ensure naming is consistent!
import database, models

# Create database tables (if not already created)
database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="Face Recognition Attendance System",
    description="An API to manage attendance using facial recognition.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers (with optional prefixes)
app.include_router(employees.router)
app.include_router(attendance.router)
app.include_router(recognize.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Attendance Management System"}
