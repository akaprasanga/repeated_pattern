import cv2
import numpy as np

img = cv2.imread('Gide Paragh-160817-113411-0928-160819-104454-4516.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
fft = np.fft.fft2(img)
cor = fft*fft
inv = np.fft.ifft2(cor)
mag = 20*np.log10(abs(inv))
# mag = mag*255
# mag = mag.astype('uint8')
#### threshold 196

thres = cv2.threshold(mag, np.mean(mag)-1*np.std(mag), 255, cv2.THRESH_BINARY)[1]
kernel = np.ones((11,11),np.uint8)
# opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
# dilation = cv2.dilate(thres,kernel,iterations = 5)
closing = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)

cv2.imwrite('fourier test.png',thres)
