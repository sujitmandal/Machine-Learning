import matplotlib.pyplot as plt
import numpy as np
import cv2

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal

"""

img = cv2.imread('D:\\Matchin Larning\\all dataset\\LISC Database\\Main Dataset\\mixt\\1.bmp')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

 # define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#cv2.imshow('res2',res2)
plt.imshow(res2)
plt.show()

plt.subplot(2, 2, 1)
plt.imshow(img)

plt.subplot(2, 2, 2)
plt.imshow(res2)

plt.show()

key = cv2.waitKey(0)
cv2.destroyAllWindows()