import face_recognition
import pickle
import cv2
import numpy as np

with open('known_faces.pkl', 'rb') as file:
    data = pickle.load(file)

known_faces = data['known_faces']
known_faces_name = data['known_faces_name']

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    colored_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(colored_frame)
    face_encodings = face_recognition.face_encodings(colored_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        distance = face_recognition.face_distance(known_faces, face_encoding)
        distance_min = np.argmin (distance)
        nice_match = matches[distance_min]
        if nice_match:
            name = known_faces_name[distance_min]
            frame = cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0,255,0), 2)
            frame = cv2.putText(frame, name, (left*4, (top*4)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            name = "Unknown"
            frame = cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0,255,0), 2)
            frame = cv2.putText(frame, name, (left*4, (top*4)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

