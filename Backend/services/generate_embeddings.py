import os
import cv2
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms

# Initialize MTCNN (for face detection) and InceptionResnetV1 (for embedding extraction)
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Convert image to embedding
def get_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    faces = mtcnn(img)  # Detect faces
    if faces is not None:
        embeddings = model(faces)  # Get embeddings for detected faces
        return embeddings.detach().cpu().numpy()
    return None  # No face detected

# Main function to generate and save embeddings
def save_embeddings(dataset_path: str, pkl_output_path: str = "assets/embeddings.pkl"):
    embeddings_dict = {}

    # Loop over each employee folder (assuming each folder represents an employee)
    for emp_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, emp_folder)

        if not os.path.isdir(folder_path):  # Skip if it's not a directory
            continue

        embeddings_list = []
        images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Loop over images and extract embeddings
        for img_name in images:
            image_path = os.path.join(folder_path, img_name)
            embedding = get_embedding(image_path)

            if embedding is not None:
                embeddings_list.append(embedding[0])  # Save embedding (remove extra dimension)

        if embeddings_list:  # Only save if at least one embedding was found
            embeddings_dict[emp_folder] = np.mean(embeddings_list, axis=0)  # Use the average embedding of all images

    # Save embeddings to .pkl file
    with open(pkl_output_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)

    print(f"Embeddings saved to {pkl_output_path}")

# Example usage
# save_embeddings("dataset/")  # Replace with your dataset folder path

if __name__ == "__main__":
    save_embeddings('dataset', pkl_output_path="assets/embeddings.pkl")


