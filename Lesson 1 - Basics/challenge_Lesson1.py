#challenge rescale

import cv2

img = cv2.imread('..\Photos\other_lady.jpg')
cv2.imshow('lady', img)

init_width=int(img.shape[1])
init_height=int(img.shape[0])
init_dimensions = (init_width,init_height)

width=int(img.shape[1]*0.5)
height=int(img.shape[0]*0.5)
dimensions = (width,height)

lady_resized = cv2.resize(img,dimensions,interpolation =cv2.INTER_AREA)

cv2.imshow('lady resize', lady_resized)


greyscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Grey',greyscale)

blur=cv2.GaussianBlur(img,(9,9),cv2.BORDER_DEFAULT)
cv2.imshow('Blur',blur)

canny=cv2.Canny(img,125,200)
cv2.imshow('Canny',canny)

rotPoint=(width//2,height//2)
rotMat = cv2.getRotationMatrix2D(rotPoint,45,scale=1.0)

img_rot = cv2.warpAffine(img,rotMat,init_dimensions)
cv2.imshow('lady rotated', img_rot)
