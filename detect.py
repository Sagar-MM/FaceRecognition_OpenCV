import cv2,os
import numpy as np
from PIL import Image
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('train/train.yml')
cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataSet'

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX #Creates a font
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        bb-predict, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        if(bb-predict==1):
             bb-predict='balu'
        elif(bb-predict==2):
             bb-predict='sagar'
        elif(bb-predict==3):
             bb-predict='ramu'

        cv2.putText(img = im, text = (bb-predict)+"----"+str(conf), org = (x,y+h), fontFace = font, fontScale = 1,
                    color = (0, 255, 0))
        cv2.imshow('im',im)

        print(bb-predict+str(conf))
        cv2.waitKey(10)
