import cv2
import os
import numpy as np
import faceRecognition  as fr
from faceRecognition import *
test_img=cv2.imread('/home/sam/Desktop/python/face_reginition/face_image/henryTest2.jpeg')
faces_detected,gray_img=faceDetection(test_img)
print("face Detected :",faces_detected)



'''for(x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

# noinspection PyUnresolvedReferences
resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection tutorial ",resized_img)
cv2.waitKey(0)
# noinspection PyUnresolvedReferences
cv2.destroyAllWindows_'''


#faces,faceID=fr.lables_for_training_data('/home/sam/Desktop/python/face_reginition/trainingimages')
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.save('trainingData.yml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('/home/sam/Desktop/python/face_reginition/trainingData.yml')
name={0:"Rami",1:"Henry",2:"sam",3:"Shreyash",4:"Chirag"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("Confidence:",confidence)
    print("label: ",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>50):                                #smaller the value of confidence more right it would be
        continue
    fr.put_text(test_img,predicted_name,x,y)


resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection tutorial ",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows