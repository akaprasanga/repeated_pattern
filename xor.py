import cv2
import numpy as np
import operator
import time
from os import walk
import random

def main(name):
    img_name = name
    img1 = cv2.imread(img_name,0)
    img2 = cv2.imread(img_name,0)

    h,w = img1.shape

    def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        # dim = None
        (h, w) = image.shape[:2]

        random_number = (random.randint(-10,10)/100)
        h = int(h + (random_number*h))
        w = int(h + (random_number*w))
        dim = (w,h)
        print('size of iamge after resize = ', dim)

        # if width is None and height is None:
        #     return image


        # if width is None:

        #     r = height / float(h)
        #     dim = (int(w * r), height)

        # else:
        #     r = width / float(w)
        #     dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation = inter)
        return resized
    def xoring_image(img1,img2):

        # ret2,img1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # ret2,im2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        average_roll_X = {}
        average_roll_Y = {}

        start_X = time.time()

        for i in range(1,global_imgX):
            rolled = np.roll(img1,i,axis=1)
            xor = cv2.bitwise_xor(rolled,img2)
            avg = np.average(xor)
            average_roll_X[i]=avg

        for i in range(1,global_imgY):
            rolled = np.roll(img1,i,axis=0)
            xor = cv2.bitwise_xor(rolled,img2)
            avg = np.average(xor)
            average_roll_Y[i]=avg

        end_x = time.time()
        print("TIME TAKEN TO PROCESS IMAGE  ==",end_x-start_X)
        # print(average_roll_X,average_roll_Y)
        key_min = min(average_roll_X.keys(), key=(lambda k: average_roll_X[k]))
        minimum_value_X = average_roll_X[key_min]
        if minimum_value_X !=0 :
            print("No COMPLETE DARK SPOTS while rolling in X Direction",minimum_value_X)

        key_min = min(average_roll_Y.keys(), key=(lambda k: average_roll_Y[k]))
        minimum_value_Y = average_roll_Y[key_min]
        if minimum_value_Y !=0 :
            print("No COMPLETE DARK SPOTS while rolling in Y Direction",minimum_value_Y)

        dark_position_in_X = [k for k,v in average_roll_X.items() if v == minimum_value_X]
        dark_position_in_Y = [k for k,v in average_roll_Y.items() if v == minimum_value_Y]

        return(dark_position_in_X,dark_position_in_Y)
    def extract_template(intersection_X,intersection_Y,image):
        
        template1 = image[0:intersection_Y,0:intersection_X]
        template2 = image[0:intersection_Y,intersection_X:(intersection_X)+(intersection_X)]
        (y,x) = template1.shape
        template_sizeX = x
        template_sizeY = y

        (y,x) = template2.shape
        template_sizeX = x
        template_sizeY = y
        cv2.rectangle(template1,(0,0),(intersection_X,intersection_Y),(0,255,0),2)
        return template1
    def reconstruct_image_from_template(original_image,template,img_name):
        img1 = original_image
        img2=template
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        global_img_Y,global_img_X = h1,w1
        vis = np.zeros((global_img_Y, global_img_X+10), np.uint8)
        vis[:global_img_Y, :global_img_X] = img1
        vis[:global_img_Y, global_img_X:global_img_X+10] = 0

        repeat_inX = int(global_img_X/w2)
        repeat_inY = int(global_img_Y/h2)

        this = img2
        for j in range(0,repeat_inX-1):
            this = np.concatenate((this, img2), axis=1)
        that = this
        for i in range (0,repeat_inY-1):
            that = np.concatenate((that, this))

        height_max = (max(global_img_Y,that.shape[0]))

        if global_img_Y < height_max:
            black_rows=np.zeros(vis.shape[1], np.uint8)
            diff = height_max-global_img_Y
            for i in range (0,diff):
                vis = np.vstack([vis,black_rows])

        elif that.shape[0] < height_max:
            black_rows = np.zeros(that.shape[1], np.uint8)

            h1 = that.shape[0]
            diff = height_max-h1
            for i in range (0,diff):
                that = np.vstack([that,black_rows])

        img_name = img_name.split('/')[1]
        img_name = 'reconstructed/'+img_name
        new =np.concatenate((vis,that),axis=1)
        cv2.imwrite(img_name,new)
    def centers_analysis(dark_in_X,dark_in_Y,img_name):
        if (len(dark_in_X) != 0) & (len(dark_in_Y) != 0):
            print(" Dark spots in X direction",dark_in_X)
            print(" Dark spots in Y direction",dark_in_Y)

            if (len(dark_in_X)) >= int(global_imgX/2)-2 :
                print("Horizontal Lining Pattern")
                intersection_X = int(global_imgX/2)
                intersection_Y = dark_in_Y[0]
                template = extract_template(intersection_X,intersection_Y,img1)
                reconstruct_image_from_template(img1,template,img_name) 

            elif (len(dark_in_Y)) >= int(global_imgY/2)-2 :
                print("Vertical Lining Pattern")
                intersection_X = dark_in_X[0]
                intersection_Y = int(global_imgY/2)
                template = extract_template(intersection_X,intersection_Y,img1) 
                reconstruct_image_from_template(img1,template,img_name) 

            else:
                #### Normal Pattern
                ######## LOGIC TO DETECT FALSE NEGATIVES PATTERNS
                if ((dark_in_X[0]==1) & (dark_in_Y[0]==1))|((dark_in_X[0]==global_imgX-1) & (dark_in_Y[0]==global_imgY-1)):
                    print("NO any repeated pattern detected 2")
                    img_name = img_name.split('/')[1]
                    cv2.imwrite('norepeat/'+img_name,img1) 
                elif (dark_in_X[0]==1) & (dark_in_Y[0]!= 1)&(dark_in_Y[0]!=global_imgY-1):
                    intersection_X = global_imgX
                    intersection_Y = dark_in_Y[0]
                    template = extract_template(intersection_X,intersection_Y,img1) 
                    reconstruct_image_from_template(img1,template,img_name) 
                elif (dark_in_Y[0]==1) & (dark_in_X[0]!= 1)&(dark_in_X[0]!= global_imgX-1):
                    intersection_X = dark_in_X[0]
                    intersection_Y = global_imgY
                    template = extract_template(intersection_X,intersection_Y,img1) 
                    reconstruct_image_from_template(img1,template,img_name) 
                else:
                    intersection_X = dark_in_X[0]
                    intersection_Y = dark_in_Y[0]
                    template = extract_template(intersection_X,intersection_Y,img1) 
                    reconstruct_image_from_template(img1,template,img_name) 
        


        else:
            print("NO any repeated pattern detected 1")
            cv2.imwrite('norepeat/'+img_name,img1) 


    img1 = image_resize(img1,width=int(w))
    img2 = img1


    global_imgY, global_imgX = img1.shape

    dark_in_X,dark_in_Y = xoring_image(img1,img2)
    centers_analysis(dark_in_X,dark_in_Y,img_name)


f = []
for (dirpath, dirnames, filenames) in walk('Mirror'):
    f.extend(filenames)
    break

counter =0
for each in f:
    name = 'Mirror/'+each
    IMAGE_NAME = name
    IMAGE_NAME = IMAGE_NAME.split('/')[1]
    print(IMAGE_NAME)

    main(name)
    print("processing image",counter)
    counter+=1
