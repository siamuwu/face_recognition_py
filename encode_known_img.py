import face_recognition
import numpy as np
import cv2
import os
import pickle

known_faces = []
known_faces_name = []

path = 'known_faces'

for file_name in os.listdir(path):
    img_file = os.path.join(path, file_name)
    known_img = face_recognition.load_image_file(img_file)
    known_img_rsz = cv2.resize(known_img, (0,0), fx = 0.25, fy = 0.25)
    known_faces_recent = face_recognition.face_encodings(known_img_rsz)
    if len(known_faces_recent) > 0:
        known_faces.append(known_faces_recent[0])
        name = os.path.splitext(file_name)[0]
        known_faces_name.append(name)

data = {
    'known_faces': known_faces,
    'known_faces_name': known_faces_name
}
with open('known_faces.pkl', 'wb') as file:
    pickle.dump(data, file)