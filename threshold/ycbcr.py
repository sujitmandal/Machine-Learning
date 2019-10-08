from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import skimage
import cv2

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal

"""

image_1 = cv2.imread('D:\\Matchin Larning\\all dataset\\LISC Database\\Main Dataset\\mixt\\1.bmp')
#image = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image_1, cv2.COLOR_BGR2YCrCb)

#plt.title("Original")
plt.imshow(image)
#plt.show()

# Compute the Laplacian of the image
lap = cv2.Laplacian(image, cv2.CV_64F)
lap_1 = np.uint8(np.absolute(lap))

#plt.title("Laplacian")
plt.imshow(lap)
#plt.show()
plt.imshow(lap_1)
#plt.show()

# Compute gradients along the X and Y axis, respectively
sobel_X = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_Y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

sobel_X = np.uint8(np.absolute(sobel_X))
sobel_Y = np.uint8(np.absolute(sobel_Y))

#plt.title("Sobel X")
plt.imshow(sobel_X)
#plt.show()

#plt.title("Sobel Y")
plt.imshow(sobel_Y)
#plt.show()

sobelCombined = cv2.bitwise_or(sobel_X, sobel_Y)
plt.title("HSV_FULL")
plt.imshow(sobelCombined)
plt.show()

#canny = cv2.Canny(sobelCombined, 30, 150)
#cv2.imshow("Canny", canny)

cv2.waitKey(0)
cv2.destroyAllWindows()