from database import engine, Base
from models import Employee, Attendance  # If you add new models like FaceEmbedding, import them too.

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully!")
    except Exception as e:
        print("Error creating tables:", e)

if __name__ == "__main__":
    init_db()
