import matplotlib.pyplot as plt
import numpy as np
import cv2

#Github: https://github.com/sujitmandal
#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal

This programe is create by Sujit Mandal

"""
image = cv2.imread('D:\\Matchin Larning\\all dataset\\LISC Database\\Main Dataset\\mixt\\1.bmp')
#image = cv2.imread("D:\\Python\\Machine Learning\\k-mean cluster\\rgb2hsv.bmp")
image_0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_1 = cv2.GaussianBlur(image_0, (5, 5), 0)

plt.title("Orginal")
plt.imshow(image)
#plt.show()
plt.title("Gray")
plt.imshow(image_0)
#plt.show()
plt.title('Blurred')
plt.imshow(image_1)
#plt.show()

#plt.subplot(2, 2, 1)
plt.title("Orginal")
plt.imshow(image)

#plt.subplot(2, 2 , 2)
plt.title("Gray")
plt.imshow(image_0)

#plt.subplot(2, 2, 3)
plt.title('Blurred')
plt.imshow(image_1)
#plt.show()

canny = cv2.Canny(image, 70, 200)
cv2.imshow("Canny", canny)

cv2.waitKey(0)
cv2.destroyAllWindows()