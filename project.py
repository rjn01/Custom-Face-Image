# -*- coding: utf-8 -*-
"""
Created on Sat May 30 01:56:31 2020

@author: shiva
"""

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
img = cv2.imread('d.jpg')

#x=np.asarray(x[0:,0:])

while True:
    _,frame = cap.read()
    #x = cv2.imread('a.jpg')
    img=np.asarray(img)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        #y=cv2.resize(x,(w,h), interpolation = cv2.INTER_AREA)
        img=cv2.resize(img,(w,h),interpolation = cv2.INTER_AREA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w] 
        frame[y:y+h,x:x+w] = img
        

    cv2.imshow("frame",frame)
    k= cv2.waitKey(30) & 0xff   #press escape to exit
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
