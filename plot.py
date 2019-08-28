import cv2
import numpy as np
import sys
from matplotlib.image import imread
i= "D:\Work\correlation\exampleimages\\reduced\\"
img_path = [i+"square_large_img.png", i+"rectangle_lattice_repeat.png", i+"parallelogram_lattice_img.png", i+"repeat_rhombic.png", i+"hexagonal_full.png", i+"color.png"]
lattice_path = []
coordinates = [(86,86), (82, 23), (148,68), (92, 55), (101, 57), (43, 42)]

def plot_displacement_vector(image_path, coordinates):
    import numpy as np
    import matplotlib.pyplot as plt

    # fig = plt.figure()
    # img = imread(image_path[0])
    # rect = cv2.rectangle(img, (coordinates[0][0], 2*coordinates[0][1]), (2*coordinates[0][0], 3*coordinates[0][1]), (0, 255, 0), thickness=2)
    # plt.subplot(1, 3, 1)
    # plt.imshow(rect)
    # plt.title('Square Lattice', fontsize=10)
    # #
    # img = imread(image_path[1])
    # rect = cv2.rectangle(img, (coordinates[1][0], 2*coordinates[1][1]), (2*coordinates[1][0], 3*coordinates[1][1]), (255, 255, 0), thickness=2)
    # plt.subplot(1, 3, 2)
    # plt.imshow(rect)
    # plt.title('Rectangle Lattice', fontsize=10)
    # #
    # img = imread(image_path[2])
    # rect = cv2.rectangle(img, (coordinates[2][0], coordinates[2][1]), (2*coordinates[2][0], 2*coordinates[2][1]), (255, 255, 0), thickness=2)
    # plt.subplot(1, 3, 3)
    # plt.imshow(rect)
    # plt.title('Parallelogram Lattice', fontsize=10)

    img = imread(image_path[3])
    rect = cv2.rectangle(img, (2*coordinates[3][0], 3*coordinates[3][1]), (3*coordinates[3][0], 4*coordinates[3][1]), (255, 255, 0), thickness=2)
    plt.subplot(2, 3, 1)
    plt.imshow(rect)
    plt.title('Rhombic Lattice', fontsize=10)

    img = imread(image_path[4])
    rect = cv2.rectangle(img, (2*coordinates[4][0], 2*coordinates[4][1]), (3*coordinates[4][0], 3*coordinates[4][1]), (255, 255, 0), thickness=2)
    plt.subplot(2, 3, 2)
    plt.imshow(rect)
    plt.title('Hexagonal Lattice', fontsize=10)


    img = imread(image_path[5])
    rect = cv2.rectangle(img, (2*coordinates[5][0], 2*coordinates[5][1]), (4*coordinates[5][0], 3*coordinates[5][1]), (255, 255, 0), thickness=2)
    plt.subplot(2, 3, 3)
    plt.imshow(rect)
    plt.title('False Detection', fontsize=10)


plot_displacement_vector(img_path, coordinates)