import tensorflow as tf
import numpy as np
import pickle

model = tf.keras.models.load_model("assets/facenet_keras.h5")
with open("assets/class_labels.pkl", "rb") as f:
    class_labels = pickle.load(f)

def get_embedding(face_pixels):
    samples = np.expand_dims(face_pixels, axis=0)
    embedding = model.predict(samples)
    return embedding[0]

def predict_face(embedding, stored_embeddings):
    min_dist = float("inf")
    identity = "Unknown"

    for label, db_emb in stored_embeddings.items():
        dist = np.linalg.norm(embedding - db_emb)
        if dist < min_dist and dist < 0.8:  # threshold
            min_dist = dist
            identity = label
    return identity
