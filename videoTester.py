import cv2
import os
import numpy as np
import faceRecognition  as fr
from faceRecognition import *



face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('/home/sam/Desktop/python/face_reginition/trainingData.yml')

name={0:"Rami",1:"Henry",2:"sam"}
cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
    resized_img=cv2.resize(test_img,(1000,700))
    cv2.imshow('face detection Tutorial ', resized_img)
    cv2.waitKey(10)

    for face in faces_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y + h, x:x + h]
        label, confidence = face_recognizer.predict(roi_gray)
        print("Confidence:", confidence)
        print("label: ", label)
        fr.draw_rect(test_img, face)
        predicted_name = name[label]
        if (confidence <50):  # smaller the value of confidence more right it would be
            fr.put_text(test_img, predicted_name, x, y)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face detection Tutorial ', resized_img)
    if cv2.waitKey(10)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows
