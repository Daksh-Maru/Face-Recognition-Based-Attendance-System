from database import engine, Base
from models import Employee, Attendance

# Create tables in the database
Base.metadata.create_all(bind=engine)

print("Done, Tables created successfully!")
# This script creates the database tables defined in the models.py file.