
import cv2
import numpy as np
from imutils import contours
from skimage import measure
import imutils
import statistics
from os import walk


IMAGE_NAME='a'
weight_counter = 3

def reconstruct(image_name):
    image = cv2.imread(image_name)
    global_img_Y,global_img_X,c = image.shape

    def autocorrelation(img,weight):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fft = np.fft.fft2(img)
        cor = fft*fft
        inv = np.fft.ifft2(cor)
        mag = 20*np.log10(abs(inv))
        thres = cv2.threshold(mag, np.mean(mag)+weight*np.std(mag), 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite('fourier/'+weight+IMAGE_NAME,thres)
        return thres
        # cv2.imshow('mag.png',mag)
    def autocorrelation_for_line(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fft = np.fft.fft2(img)
        cor = fft*fft
        inv = np.fft.ifft2(cor)
        mag = 20*np.log10(abs(inv))
        thres = cv2.threshold(mag, np.mean(mag)+2*np.std(mag), 255, cv2.THRESH_BINARY)[1]
        # th3 = cv2.adaptiveThreshold(mag,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        cv2.imwrite('fourier/2'+IMAGE_NAME,thres)
        return thres
    def detect_contour(thres):
        MIN_THRESH = 0
        centers = []

        thres = cv2.dilate(thres, None, iterations=4)
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

                centers.append([cX,cY])
                centers[i] = (cX,cY)
                print(cX,cY)
                # print('from contour')
                i += 1
                # draw the contour and center of the shape on the image
                # cv2.drawContours(thres, [c], -1, (0, 255, 0), 1)

        
            else:
                print('NO ANY BRIGHT SPOTS DETECTED')
        return centers
    def findintersection(center):
        point_of_same_cordinate =[]
        for i, each in enumerate(center):
            if i == 0:
                continue
            elif (each[0]==each[1]):
                out = each
                point_of_same_cordinate.append(each)
                
            elif(each[0]+1==each[1]):
                out = each
                point_of_same_cordinate.append((each[0],each[0]))
                
            elif(each[0]-1==each[1]):
                out = each
                point_of_same_cordinate.append((each[0],each[0]))
                
            elif(each[0]==each[1]-1):
                out = each
                point_of_same_cordinate.append((each[0],each[0]))
                
            elif(each[0]==each[1]+1):
                out = each
                point_of_same_cordinate.append((each[0],each[0]))
            elif(each[0]+2==each[1]):
                out = each
                point_of_same_cordinate.append((each[0],each[0]))
                
            elif(each[0]-2==each[1]):
                out = each
                point_of_same_cordinate.append((each[0],each[0]))
                
            elif(each[0]==each[1]-2):
                out = each
                point_of_same_cordinate.append((each[0],each[0]))
                
            elif(each[0]==each[1]+2):
                out = each
                point_of_same_cordinate.append((each[0],each[0]))
                
        return out,point_of_same_cordinate
    def roation_and_subtraction(template1,template2):
        subtraction_values = []

        rotated_90 = imutils.rotate(template1, 90)
        rotated_180 = imutils.rotate(template1, 180)
        rotated_270 = imutils.rotate(template1, 270)

        subracted_image_r90 = np.subtract(template2,rotated_90)
        subracted_image = cv2.cvtColor(subracted_image_r90,cv2.COLOR_BGR2GRAY)
        ret,thresh_substracted = cv2.threshold(subracted_image,127,255,cv2.THRESH_BINARY)
        avg_hist_subracted = np.average(thresh_substracted)
        subtraction_values.append(avg_hist_subracted)

        # subracted_image_r180 = np.subtract(template2,rotated_180)
        # subracted_image = cv2.cvtColor(subracted_image_r180,cv2.COLOR_BGR2GRAY)
        # ret,thresh_substracted = cv2.threshold(subracted_image,127,255,cv2.THRESH_BINARY)
        # avg_hist_subracted = np.average(thresh_substracted)
        # subtraction_values.append(avg_hist_subracted)

        subracted_image_r270 = np.subtract(template2,rotated_270)
        subracted_image = cv2.cvtColor(subracted_image_r270,cv2.COLOR_BGR2GRAY)
        ret,thresh_substracted = cv2.threshold(subracted_image,127,255,cv2.THRESH_BINARY)
        avg_hist_subracted = np.average(thresh_substracted)
        subtraction_values.append(avg_hist_subracted)


        return subtraction_values
    def detect_lines(img):
        # thres = autocorrelation_for_line(img)
        centers = detect_contour(thres)
        intersection = ()
        cenX=[]
        cenY=[]
        for each in centers:
            (x,y) = each
            cenX.append(x)
            cenY.append(y)
        ## computing the difference of consecutive elements
        diff_X = [t - s for s, t in zip(cenX, cenX[1:])]
        diff_Y = [t - s for s, t in zip(cenY, cenY[1:])]
        try:
            mode_X = abs(statistics.mode(diff_X))
            mode_Y = abs(statistics.mode(diff_Y))
        except statistics.StatisticsError :
            print('cant find mode')
            mode_X = int(abs(np.mean(diff_X)))
            mode_Y = int(abs(np.mean(diff_Y)))
        print(mode_X,mode_Y)
        if mode_X == 0:
            print("Horizontal lining Pattern")
            try:
                mode_center = statistics.mode(cenX)
            except:
                ######### single horizontal line pattern in autocorrelated image
                mode_center = int(global_img_X)
            # cv2.rectangle(image,(0,0),(mode_center,mode_Y),(0, 255, 0), 1)
            print(mode_center,mode_Y)
            intersection = (mode_center,mode_Y)
            # cv2.imshow('lining',image)

        elif mode_Y ==0:
            print("Vertical Lining Pattern")
            try:
                mode_center = statistics.mode(cenY)
                times_Y = round(global_img_Y/mode_center)
                mode_center = int(global_img_Y/times_Y) 
            except:
                y,x,c = image.shape
                mode_center = int(y)
            print(mode_center)
            # cv2.rectangle(image,(0,0),(mode_X,mode_center),(0, 255, 0), 1)
            print(mode_X,mode_center)
            intersection = (mode_X,mode_center)

            # cv2.imshow('lining',image)
        return intersection
    def extract_templates(intersection_X,intersection_Y):
        
        # times_Y = round(global_img_Y/intersection_Y)
        # intersection_Y = round (global_img_Y/times_Y)
        # times_X = round(global_img_X/intersection_X)
        # intersection_X = round(global_img_X/times_X)

        template1 = image[0:intersection_Y,0:intersection_X]
        template2 = image[0:intersection_Y,intersection_X:(intersection_X)+(intersection_X)]
        (y,x,c) = template1.shape
        template_sizeX = x
        template_sizeY = y

        print('template 1 size',x,y)

        (y,x,c) = template2.shape
        template_sizeX = x
        template_sizeY = y
        print('template 2 size',x,y)
        cv2.imwrite('template1.png',template1)
        cv2.imwrite('template2.png',template2)
        return template1,template2


    try:
        #### FOR GENERAL IMAGE
        thres = autocorrelation(image,3)
        centers = detect_contour(thres)
        if len(centers)<1 :
            thres = autocorrelation(image,2)
            centers = detect_contour(thres)
            if len(centers)<1 :
                thres = autocorrelation(image,1)
                centers = detect_contour(thres)
                if len(centers)<1:
                    print("ERRORRR NO CONTOUR")

        intersection,same_points = findintersection(reversed(centers))

        if (len(same_points)>1):
            (intersection_X,intersection_Y)=same_points[1]
            if (global_img_X/intersection_X) > 1.9:
                (intersection_X,intersection_Y)=same_points[1]
            else:
                (intersection_X,intersection_Y)=same_points[0]
        else:
            (intersection_X,intersection_Y) = same_points[0]
        print(intersection)
        print('same intersection',same_points)
        # pattern = cv2.rectangle(image,(0,0),intersection,(0,255,0),1)

    except:
        ######### FOR LINE
        thres = autocorrelation_for_line(image)
        try:
            centers = detect_contour(thres)
            intersection,same_points = findintersection(reversed(centers))

            if (len(same_points)>1):
                (intersection_X,intersection_Y)=same_points[1]
                if (global_img_X/intersection_X) > 1.9:
                    (intersection_X,intersection_Y)=same_points[1]
                else:
                    (intersection_X,intersection_Y)=same_points[0]
            else:
                (intersection_X,intersection_Y) = same_points[0]
        except:
            print('EXCEPTION CAUGHT')
            thres = autocorrelation_for_line(image)

            (intersection_X,intersection_Y) = detect_lines(thres)

        
    print('finally intersenting point are',intersection_X,intersection_Y)

    template1, template2 = extract_templates(intersection_X,intersection_Y)

    # rotation--------------------------------
    # subtraction_values = roation_and_subtraction(template1,template2)
    # subtraction_values.sort(reverse=True)

    # minimum_value = subtraction_values[len(subtraction_values)-1]
    # maximum_value = subtraction_values[0]
    # difference = maximum_value-minimum_value

    # if difference > (maximum_value*0.50):
    #     print('MIRRORED  IMAGE')
    #     print(centers)
    # else:
    #     print('NOT MIRRORED IMAGE')


    template_sizeY,template_sizeX,c =template1.shape 

    repeat_inX = round((global_img_X/(template_sizeX)))
    repeat_inY = round((global_img_Y/template_sizeY))
    print('Repeat in X Direction =',repeat_inX)
    print('Repeat in Y Direction =',repeat_inY)

    # print(subtraction_values)
    # print(np.std(np.array(subtraction_values)))


    # img1 = cv2.imread(image_name,0)
    # img2 = cv2.cvtColor(template1,cv2.COLOR_BGR2GRAY)
    # h1, w1 = img1.shape[:2]
    # h2, w2 = img2.shape[:2]
    # vis = np.zeros((global_img_Y, global_img_X+10), np.uint8)
    # vis[:global_img_Y, :global_img_X] = img1
    # vis[:global_img_Y, global_img_X:global_img_X+10] = 0


    # this = img2
    # for j in range(0,repeat_inX-1):
    #     this = np.concatenate((this, img2), axis=1)
    # that = this
    # for i in range (0,repeat_inY-1):
    #     that = np.concatenate((that, this))

    # height_max = (max(global_img_Y,that.shape[0]))

    # if global_img_Y < height_max:
    #     black_rows=np.zeros(vis.shape[1], np.uint8)
    #     diff = height_max-global_img_Y
    #     for i in range (0,diff):
    #         vis = np.vstack([vis,black_rows])

    # elif that.shape[0] < height_max:
    #     black_rows = np.zeros(that.shape[1], np.uint8)

    #     h1 = that.shape[0]
    #     diff = height_max-h1
    #     for i in range (0,diff):
    #         that = np.vstack([that,black_rows])

    # image_name = image_name.split('/')[1]
    # image_name = 'reconstructed/'+image_name
    # new =np.concatenate((vis,that),axis=1)
    # cv2.imwrite(image_name,new)

# image_name = 'Gide Paragh-160817-113622-9236-160819-105129-1542.png'
# reconstruct(image_name)

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

    reconstruct(name)
    print("processing image",counter)
    counter+=1

