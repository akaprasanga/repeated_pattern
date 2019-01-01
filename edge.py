import cv2
import numpy as np  
import random  

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


image = cv2.imread('Bhaca Arture.png',0)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

kernel = np.ones((15, 15), np.uint8)

# opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
edges = cv2.Canny(image, 50, 100, apertureSize=3)  # Canny edge detection

cv2.imshow('edges',edges)
cv2.waitKey(0)