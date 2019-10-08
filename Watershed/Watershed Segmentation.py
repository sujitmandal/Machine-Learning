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
image = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

plt.title("Original")
plt.imshow(image)
#plt.show()

# Compute the Laplacian of the image
lap = cv2.Laplacian(image, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))

plt.title("Laplacian")
plt.imshow(lap)
#plt.show()

# Compute gradients along the X and Y axis, respectively
sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

plt.title("Sobel X")
plt.imshow(sobelX)
#plt.show()

plt.title("Sobel Y")
plt.imshow(sobelY)
#plt.show()

sobelCombined = cv2.bitwise_or(sobelX, sobelY)
plt.title("Sobel Combined")
plt.imshow(sobelCombined)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()