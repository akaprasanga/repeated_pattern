import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Mirror_Designs_test/181203-043642-0713.png", 0)

fft = np.fft.fft2(img)
cor = fft * fft
inv = np.fft.ifft2(cor)
mag = 20 * np.log10(abs(inv))
thres = cv2.threshold(mag, np.mean(mag) + 2 * np.std(mag), 255, cv2.THRESH_BINARY)[1]

cv2.imwrite("fou.jpg", thres)