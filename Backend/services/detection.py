from mtcnn import MTCNN
import cv2 as cv

detector = MTCNN()

def detect_face(image):
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = abs(x), abs(y)
        return image[y:y+h, x:x+w]
    return None
