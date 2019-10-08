from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import cv2

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal

"""

img = cv2.imread('D:\\Matchin Larning\\all dataset\\LISC Database\\Main Dataset\\mixt\\1.bmp')
 
rgb2lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
rgb2hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb2ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
rgb2yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

plt.title("LAB")
plt.imshow(rgb2lab)
plt.show()

plt.title("HSV")
plt.imshow(rgb2hsv)
plt.show()

plt.title("YCRCB")
plt.imshow(rgb2ycrcb)
plt.show()

plt.title("YUV")
plt.imshow(rgb2yuv)
plt.show()

plt.subplot(2, 2, 1)
plt.imshow(rgb2lab)

plt.subplot(2, 2, 2)
plt.imshow(rgb2hsv)

plt.subplot(2, 2, 3)
plt.imshow(rgb2ycrcb)

plt.subplot(2, 2, 4)
plt.imshow(rgb2yuv)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()