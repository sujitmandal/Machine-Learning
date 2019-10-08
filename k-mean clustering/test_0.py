import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal

"""

img = cv2.imread("D:\\Matchin Larning\\all dataset\\LISC Database\\Main Dataset\\mixt\\1.bmp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(img)
r = r.flatten()
g = g.flatten()
b = b.flatten()
#plotting 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(r, g, b)
plt.show()