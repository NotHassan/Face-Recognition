import face_recognition
import os
import cv2
import pickle
import numpy
import time
import tkinter as tk
from tkinter import simpledialog

KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.55
MODEL = 'cnn'

FONT_COLOR = (0, 0, 0)
FONT_THICKNESS = 2
FRAME_COLOR = [255, 255, 255]
FRAME_THICKNESS = 3

video = cv2.VideoCapture(0)


def get_encodings(directory_name):
    id_encodings = []
    for file in os.listdir(f'{KNOWN_FACES_DIR}/{directory_name}'):
        id_encodings.append(pickle.load(open(f'{KNOWN_FACES_DIR}/{directory_name}/{file}', 'rb')))
    return id_encodings


def draw_box(image, location, text='Face'):
    top_left = (location[3], location[0])
    bottom_right = (location[1], location[2])
    cv2.rectangle(image, top_left, bottom_right, FRAME_COLOR, FRAME_THICKNESS)

    top_left = (location[3], location[2])
    bottom_right = (location[1], location[2] + 22)
    cv2.rectangle(image, top_left, bottom_right, FRAME_COLOR, cv2.FILLED)

    cv2.putText(image, text, (location[3] + 10, location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FONT_COLOR)


while True:
    ret, frame = video.read()
    if ret is False:
        break

    locations = face_recognition.face_locations(frame, model=MODEL)
    encodings = face_recognition.face_encodings(frame, locations, num_jitters=10)

    for face_encoding, face_location in zip(encodings, locations):
        create_new_identity = True
        name = 'Face'
        directories = [d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]

        for directory in directories:
            identity_encodings = get_encodings(directory)
            distance = face_recognition.face_distance(identity_encodings, face_encoding)
            compare = face_recognition.compare_faces(identity_encodings, face_encoding, TOLERANCE)
            next_id = len([f for f in os.listdir(f'{KNOWN_FACES_DIR}/{directory}') if os.path.isfile(os.path.join(f'{KNOWN_FACES_DIR}/{directory}', f))])

            if (0.55 if next_id <= 15 else 0.70) <= numpy.average(distance) <= 0.8 and True in compare:
                name = directory
                create_new_identity = False
                pickle.dump(face_encoding, open(f'{KNOWN_FACES_DIR}/{directory}/{next_id}-{int(time.time())}.pkl', 'wb'))
            elif True in compare:
                name = directory
                create_new_identity = False

        if create_new_identity:
            new_identity = len(directories)
            application_window = tk.Tk()
            application_window.withdraw()
            draw_box(frame, face_location, 'UNKNOWN')
            cv2.imshow('Face Recognition', frame)
            identity_name = simpledialog.askstring('Input', f'Enter a name for UNKNOWN:', parent=application_window)

            if identity_name is not None:
                name = identity_name
                try:
                    os.mkdir(f'{KNOWN_FACES_DIR}/{identity_name}')
                except FileExistsError:
                    pass
                next_encoding_id = len([f for f in os.listdir(f'{KNOWN_FACES_DIR}/{identity_name}') if os.path.isfile(os.path.join(f'{KNOWN_FACES_DIR}/{identity_name}', f))])
                pickle.dump(face_encoding, open(f'{KNOWN_FACES_DIR}/{identity_name}/{next_encoding_id}-{int(time.time())}.pkl', 'wb'))
                application_window.destroy()
        draw_box(frame, face_location, name)
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        break
