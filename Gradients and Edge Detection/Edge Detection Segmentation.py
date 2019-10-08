from skimage.color import rgb2gray
from sklearn.cluster import KMeans
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal

This programe is create by Sujit Mandal

"""


image = cv2.imread('D:\\Matchin Larning\\all dataset\\LISC Database\\Main Dataset\\mixt\\1.bmp')
plt.imshow(image)
#plt.show()

# converting to grayscale
gray = rgb2gray(image)

# defining the sobel filters
sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
print(sobel_horizontal, 'is a kernel for detecting horizontal edges')
 
sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
print(sobel_vertical, 'is a kernel for detecting vertical edges')
plt.imshow(sobel_vertical)
plt.imshow(sobel_horizontal)
#plt.show()


out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
# here mode determines how the input array is extended when the filter overlaps a border.
plt.imshow(out_h, cmap='gray')
plt.show()

plt.imshow(out_v, cmap='gray')
plt.show()

kernel_laplace = np.array([np.array([1, 1, 1]), np.array([1, -8, 1]), np.array([1, 1, 1])])
print(kernel_laplace, 'is a laplacian kernel')
plt.imshow(kernel_laplace)
#plt.show()

out_l = ndimage.convolve(gray, kernel_laplace, mode='reflect')
plt.imshow(out_l, cmap='gray')
plt.show()


plt.subplot(2, 2, 1)
plt.imshow(image)

plt.subplot(2, 2 , 2)
plt.imshow(out_h)

plt.subplot(2, 2, 3)
plt.imshow(out_v)

plt.subplot(2, 2 , 4)
plt.imshow(out_l)

plt.show()

key = cv2.waitKey(0)
cv2.destroyAllWindows()