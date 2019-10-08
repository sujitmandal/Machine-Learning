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
pic = cv2.imread('D:\\Matchin Larning\\all dataset\\LISC Database\\Main Dataset\\mixt\\1.bmp')/255  # dividing by 255 to bring the pixel values between 0 and 1
print(pic.shape)
plt.imshow(pic)
#plt.show()

pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
pic_n.shape



kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)
plt.show()

plt.subplot(2, 2, 1)
plt.imshow(pic)

plt.subplot(2, 2, 2)
plt.imshow(cluster_pic)

plt.show()

key = cv2.waitKey(0)
cv2.destroyAllWindows()