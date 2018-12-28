import numpy as np
import cv2
from os import walk



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



    import_image = cv2.imread(IMG_NAME, 0)
    # cv2.imshow('gray',import_image)
    IMG_NAME = IMG_NAME.split('/')[1]
    original_image = cv2.imread('Mirror/'+IMG_NAME)
    row_indexes, col_indexes = get_indexes(import_image)
    row_breaks = get_break_points(row_indexes[0])
    col_breaks = get_break_points(col_indexes[0])
    print(row_indexes, col_indexes)
    # print(np.where(y_indexes==71723))
    x,y = compare_breaks(import_image,row_breaks, col_breaks)
    if (x==0 and y==0):
        if (len(col_breaks) < 2):
            x = col_indexes[0][1] - col_indexes[0][0]
        else:
            x = col_breaks[1] - col_breaks[0]
        if (len(row_breaks) < 2):
            y = row_indexes[0][1] - row_indexes[0][0]
        else:
            y = row_breaks[1] - row_breaks[0]
        # output = original_image[0:y, 0:x]
        cv2.rectangle(original_image,(0,0),(x,y),(0, 255, 0), 2)

    else:
        # output = original_image[0:y, 0:x]
        cv2.rectangle(original_image,(0,0),(x,y),(0, 255, 0), 2)

    cv2.imwrite('output/'+IMG_NAME,original_image)

# IMG_NAME = '181203-043641-5284.png'

# main(IMG_NAME)
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
