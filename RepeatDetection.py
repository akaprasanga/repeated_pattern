import numpy as np
import cv2
from os import walk
import imutils


def main(IMG_NAME):
        
    def get_indexes(image):
        (height, width) = image.shape[:2]
        row_sum = np.zeros((image.shape))
        col_sum = np.zeros((image.shape))
        for i in range(width-1):
            a = image[:,i:i+1]
            row_sum[:,i:i+1] = (i+1)*(a)
        for j in range(height-1):
            b = image[j:j+1,:]
            col_sum[j:j+1,:] = (j+1)*(b)
                
        row_sum = np.sum(row_sum, axis=1)
        col_sum = np.sum(col_sum, axis=0)
        
        unique_rows = np.unique(row_sum)
        unique_cols = np.unique(col_sum)
        
        color_index = 0

        row_indexes = np.where(row_sum==unique_rows[color_index])
        col_indexes = np.where(col_sum==unique_cols[color_index])
        
        return row_indexes, col_indexes

    def get_break_points(sequence):
        if len(sequence)<2:
            return sequence
        breaks = []
        d = sequence[1]-sequence[0]
        for counter, number in enumerate(sequence):
            if (counter < 1 and number==0):
                continue
            elif (counter < 1  and number!=0):
                breaks.append(number)
            elif (number - sequence[counter-1]) != d:
                breaks.append(number)
        return breaks

    def mse(imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        imageC = imageB.copy()
        if imageA.shape != imageB.shape:
            imageD = imageA.copy()
            imageD[0:imageB.shape[0], 0:imageB.shape[1]]= imageB
            imageC = imageD
        err = np.sum((imageA.astype("float") - imageC.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err


    def compare_breaks(image,row_breaks, col_breaks):
        if (len(row_breaks)<2 or len(col_breaks)<2):
            return 0,0
        if (row_breaks==None or col_breaks==None):
            return 0,0
        for i in range(1,len(row_breaks)):
            y = row_breaks[i]-row_breaks[0]
            for j in range(1, len(col_breaks)):
                x = col_breaks[j]-col_breaks[0]
                err1 = mse(image[0:y, 0:x], image[0:y, x:2*(x)])
                err2 = mse(image[0:y, 0:x], image[y:2*(y), 0:x])
                err3 = mse(image[0:y, 0:x], image[y:2*(y), x:2*x])
                if abs(err1)==abs(err2)==abs(err3):
                    break
            if abs(err1)==abs(err2)==abs(err3):
                break
        return(x,y)

    def detect_contour(thres):
        MIN_THRESH = 0
        # MIN_THRESH = 0.00001*(global_img_X*global_img_Y)
        centers = []
        kernel = np.ones((11,11),np.uint8)

        thres = cv2.dilate(thres, None, iterations=1)
        thres = cv2.convertScaleAbs(thres)
        cnts= cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        # print(cnts)
        # loop over the contours

        i = 0
        for c in cnts:
            if cv2.contourArea(c) > MIN_THRESH:
            # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                centers.append((cX,cY))
        return centers

    # import_image = cv2.imread(IMG_NAME, 0)
    # cv2.imshow('gray',import_image)
    IMG_NAME = IMG_NAME.split('/')[1]
    original_image = cv2.imread('fourier/'+IMG_NAME,0)


    centers = detect_contour(original_image)
    col_value=[]
    row_value=[]
    for each in centers:
        (x,y) = each
        if x not in col_value:
            col_value.append(x)
        if y not in row_value:
            row_value.append(y)

    print(col_value,row_value)
    print(original_image[0][0])
    diff_X = [t - s for s, t in zip(col_value, col_value[1:])]
    diff_Y = [t - s for s, t in zip(row_value, row_value[1:])]
    # cv2.imwrite('output/'+IMG_NAME,original_image)
    print(diff_X,diff_Y)
# IMG_NAME = '181203-043641-5284.png'

f = []
for (dirpath, dirnames, filenames) in walk('fourier'):
    f.extend(filenames)
    break

counter =0
for each in f:
    name = 'fourier/'+each
    IMAGE_NAME = name
    IMAGE_NAME = IMAGE_NAME.split('/')[1]
    # print(IMAGE_NAME)

    main(name)
    print("processing image",counter)
    counter+=1
