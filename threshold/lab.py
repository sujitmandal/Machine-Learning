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

img = cv2.imread('D:\\Matchin Larning\\all dataset\\LISC Database\\Main Dataset\\mixt\\1.bmp')

rgb2lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


plt.title("LAB")
plt.imshow(rgb2lab)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
