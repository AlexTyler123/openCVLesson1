import cv2
import numpy as np
import matplotlib.pyplot as plt

img =cv2.imread('../Faces/usrc_cropped.png')
cv2.imshow('croppedUSRC',img)

#convert to greyscale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('croppedUSRC',gray)

#store the haar face database to haarCascade
haarCascade=cv2.CascadeClassifier('haar_face.xml')

#detect a face and return the rectangular coordinates of the face
facesRect=haarCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=7)
#modify minNeighbors to help filter noise


print(f'Number of faces found = {len(facesRect)}')

#get coordinates from facesRect and draw rectangles
for (x,y,w,h) in facesRect:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    
cv2.imshow('Detected Face',img)

cv2.waitKey(0)
