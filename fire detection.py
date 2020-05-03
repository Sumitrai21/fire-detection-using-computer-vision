###### MAKE A OBJECT DETECTION IN OPENCV USING MASKING IN REAL TIME #######################

import cv2 as cv
import numpy as np

cap = cv.VideoCapture('fire_video.mp4')
_, frame1 = cap.read()
_, frame2 = cap.read()


while True:
    hsv = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)
    #diff = cv.absdiff(frame1,frame2)
    
    lb = np.array([0,255,255])
    ub = np.array([255,255,255])
    mask = cv.inRange(hsv,lb,ub)

    gray = mask
    blur = cv.GaussianBlur(gray,(5,5),0)
    _, thresh = cv.threshold(blur,100,200, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations = 3)
    contours,_ = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        (x,y,w,h) = cv.boundingRect(contour)

        if cv.contourArea(contour) <500:
            continue

        cv.rectangle(frame1, (x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(frame1, 'Fire',(10,20),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0))



    output = cv.bitwise_and(frame1,frame1,mask = mask)

    cv.imshow('frame', frame1)
    cv.imshow('mask', dilated)
    cv.imshow('output', output)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

    frame1 = frame2
    _, frame2 = cap.read()
          
    

   
       

cv.destroyAllWindows()