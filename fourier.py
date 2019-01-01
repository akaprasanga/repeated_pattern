import cv2
import numpy as np
import imutils

img = cv2.imread('cubinia.png')
rotated_180 = imutils.rotate(img, 90)


cv2.imshow('original',img)

cv2.imshow('rotated',rotated_180)
cv2.waitKey()