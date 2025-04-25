import cv2 as cv
import numpy as np
from .preprocessing import enhance_image
def load_image_from_bytes(img_bytes):
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return enhance_image(img_rgb)
    #return img_rgb

def preprocess_face(face, target_size=(160, 160)):
    face = cv.resize(face, target_size)
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    return (face - mean) / std
