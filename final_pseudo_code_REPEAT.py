import cv2
import numpy as np
import sys
from matplotlib.image import imread

image_path = 'D:\Work\correlation\exampleimages\\reduced\\hexagonal_full.png'
print(image_path)
original_image = cv2.imread(image_path)
global_img_Y, global_img_X = original_image.shape[:2]

def input_image(image_path):
    image = cv2.imread(image_path, 0)
    return image

# retun type == (array_value_X,array_value_Y)
def xoring_images(img1):

    img2 = img1 # making  a copy of image
    # intializing key value pair for the average of xored value
    average_xor_value_x = {}
    average_xor_value_y = {}

    for i in range(1, global_img_X):
        rolled = np.roll(img1, i, axis=1)
        xor = cv2.bitwise_xor(rolled, img2)
        avg = np.average(xor)
        average_xor_value_x[i] = avg

    for i in range(1, global_img_Y):
        rolled = np.roll(img1, i, axis=0)
        xor = cv2.bitwise_xor(rolled, img2)
        avg = np.average(xor)
        average_xor_value_y[i] = avg


    key_min = min(average_xor_value_x.keys(), key=(lambda k: average_xor_value_x[k]))
    minimum_value_X = average_xor_value_x[key_min]
    if minimum_value_X != 0:
        print("No COMPLETE DARK SPOTS while rolling in X Direction", minimum_value_X)

    key_min = min(average_xor_value_y.keys(), key=(lambda k: average_xor_value_y[k]))
    minimum_value_Y = average_xor_value_y[key_min]
    if minimum_value_Y != 0:
        print("No COMPLETE DARK SPOTS while rolling in Y Direction", minimum_value_Y)

    first_dark_position_in_X = [k for k, v in average_xor_value_x.items() if v == minimum_value_X]
    first_dark_position_in_Y = [k for k, v in average_xor_value_y.items() if v == minimum_value_Y]


    avg_value_X = [v for k, v in average_xor_value_x.items() if v == minimum_value_X]
    avg_value_Y = [v for k, v in average_xor_value_y.items() if v == minimum_value_Y]


    return first_dark_position_in_X,first_dark_position_in_Y, average_xor_value_x, average_xor_value_y


#return type == (inter_val_X,integer_val_Y)
def centers_analysis(dark_in_X, dark_in_Y):
    print(dark_in_X, dark_in_Y)
    if (len(dark_in_X) != 0) & (len(dark_in_Y) != 0):

        # Case for Horizontal lining pattern
        if (len(dark_in_X)) >= int(global_img_X / 2) - 2:
            print("Horizontal Lining Pattern")
            intersection_X = int(global_img_X / 2)
            intersection_Y = dark_in_Y[0]
            return (intersection_X, intersection_Y)

        # Case for Vertical lining pattern
        elif (len(dark_in_Y)) >= int(global_img_Y / 2) - 2:
            print("Vertical Lining Pattern")
            intersection_X = dark_in_X[0]
            intersection_Y = int(global_img_Y / 2)
            return (intersection_X, intersection_Y)

        else:
            # LOGIC TO DETECT FALSE NEGATIVES PATTERNS
            if (((dark_in_X[0] == 1) | (dark_in_X[0] == 2)) & ((dark_in_Y[0] == 1) | (dark_in_Y[0] == 2))) | (
                    (dark_in_X[0] == global_img_X - 1) & (dark_in_Y[0] == global_img_Y - 1)):
                print("NOW returning for EDGE DETECTION")
                return None

            # logic for complex horizontal pattern
            elif (dark_in_X[0] == 1) & (dark_in_Y[0] != 1) & (dark_in_Y[0] != global_img_Y - 1):
                intersection_X = global_img_X
                intersection_Y = dark_in_Y[0]
                return (intersection_X, intersection_Y)

            # logic for complex vertical pattern
            elif (dark_in_Y[0] == 1) & (dark_in_X[0] != 1) & (dark_in_X[0] != global_img_X - 1):
                intersection_X = dark_in_X[0]
                intersection_Y = global_img_Y
                return (intersection_X, intersection_Y)

            # logic for simple pattern
            else:
                intersection_X = dark_in_X[0]
                intersection_Y = dark_in_Y[0]
                return (intersection_X, intersection_Y)


    else:
        print("NO any repeated pattern detected LEVEL = 1")

#return type == (float_in_x, float_in_Y)
def compute_repetation_number(intersection_X, intersection_Y):
    repeat_in_x = global_img_X/intersection_X
    repeat_in_y = global_img_Y/intersection_Y
    return (repeat_in_x,repeat_in_y)

def plot_displacement_vector(average_dictionary_X, average_dictionary_Y, cord_x, cord_y):
    import numpy as np
    import matplotlib.pyplot as plt

    x_x = [k for k,v in average_dictionary_X.items()]
    y_x = [v for k,v in average_dictionary_X.items()]

    x_y = [k for k,v in average_dictionary_Y.items()]
    y_y = [v for k,v in average_dictionary_Y.items()]
    # x = [2, 4, 6]
    # y = [1, 3, 5]
    fig = plt.figure()
    img = imread(image_path)
    rect = cv2.rectangle(img, (cord_x, cord_y), (2*cord_x, 2*cord_y), (255, 255, 0), thickness=2)
    lattice = imread('D:\Work\correlation\exampleimages\\reduced\\hexagonal_lattice.png')

    plt.subplot(1, 3, 1)
    plt.imshow(lattice)
    plt.title('Hexagonal Lattice', fontsize=10)

    plt.subplot(1, 3, 3)
    plt.plot(x_x, y_x)
    plt.plot(x_y, y_y, '--')
    plt.xlabel("Width", fontsize=6)
    plt.ylabel("Avg. of XORed Image", fontsize=6)
    plt.title('Horizontal/Vertical distance vector', fontsize=10)

    plt.subplot(1, 3, 2)
    plt.imshow(rect)
    plt.title('Lattice Detection in Tiled Image', fontsize=10)




    # plt.subplot(2, 2, 4)
    # plt.imshow(img)
    # plt.title('Repeated motif selection', fontsize=10)
    # plt.plot(x, y)
    # plt.xlabel('Width')
    # plt.ylabel('Average value of XORed Image')
    plt.show()

if __name__ == '__main__':

    X,Y, average_x, average_y = xoring_images(original_image)

    returned_centers = centers_analysis(X, Y)
    print("returned centers", returned_centers)
    plot_displacement_vector(average_x, average_y, returned_centers[0], returned_centers[1])


    if returned_centers is not None:
        (intersection_X, intersection_Y) = returned_centers
        repeat_in_x, repeat_in_y = compute_repetation_number(intersection_X, intersection_Y)
        print(intersection_X, intersection_Y, repeat_in_x, repeat_in_y)
        rect = cv2.rectangle(original_image, (0,0), (intersection_X, intersection_Y), (0, 255, 0), thickness=2)
        cv2.imwrite('detected.png', rect)
    else:
        edges = cv2.Canny(original_image, 50, 100, apertureSize=3)  # Canny edge detection
        X, Y = xoring_images(edges)
        returned_centers = centers_analysis(X, Y)
        if returned_centers is not None:
            (intersection_X, intersection_Y) = returned_centers
            repeat_in_x, repeat_in_y = compute_repetation_number(intersection_X, intersection_Y)
            print(intersection_X, intersection_Y, repeat_in_x, repeat_in_y)

        else:
            print(" Failed to detect Any Repeated Pattern Level = 2 ")

    # input("Press Enter to continue...")

