# generate_embeddings.py

import os
import cv2
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
from database import SessionLocal
from models import Employee
from sqlalchemy.orm import Session

model = InceptionResnetV1(pretrained='vggface2').eval()
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])


def get_embedding(image_path):
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)
    embedding = model(img_tensor).detach().numpy()
    return embedding


def save_embeddings_to_db(dataset_path: str):
    db: Session = SessionLocal()
    for emp_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, emp_folder)
        images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

        if not images:
            continue

        # Take one image for embedding (or average if multiple)
        image_path = os.path.join(folder_path, images[0])
        embedding = get_embedding(image_path)

        # Save to DB
        db_employee = Employee(name=emp_folder, email=f"{emp_folder}@example.com",
                               face_embedding=pickle.dumps(embedding))
        db.add(db_employee)
    db.commit()
    db.close()
    print("Embeddings saved to database!")

# Run this:
# save_embeddings_to_db("dataset/")
