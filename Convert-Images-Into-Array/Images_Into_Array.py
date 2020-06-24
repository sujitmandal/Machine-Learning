import os 
import cv2 
import numpy as np 
from tqdm import tqdm
from random import shuffle 

#Github: https://github.com/sujitmandal
#This programe is create by Sujit Mandal
"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
Facebook : https://www.facebook.com/sujit.mandal.33671748
Twitter : https://twitter.com/mandalsujit37
"""

images_path = input('Enter Image Folder Path : ') #Path of the images folder
image_size = int(input('Enter The Image Size [32, 64, 128] : '))

def images(images_path, image_size):
    empty_list = []

    for image in tqdm(os.listdir(images_path)):
        path = os.path.join(images_path, image)

        image = cv2.imread(path)
        image = cv2.resize(image , (image_size, image_size))
        empty_list.append([np.array(image)])
    shuffle(empty_list)

    return(empty_list)

#All the images Stored Into a List
list_images = images(images_path, image_size)

#Convert List Into Array
array_image = np.array(list_images)
print('Array Shape : ', array_image.shape)

#Removed Dimention 
images = array_image[:,0,:,:]
print('Image Shape : ',images.shape)

#OUTPUT :
'''
Enter Image Folder Path : /media/sujit/92EC423BEC4219BD/GitHub Preoject/ALL ML PROJECT/Face Mask Detection/face mask detection  dataset/test/without mask
Enter The Image Size [32, 64, 128] : 32
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 186/186 [00:00<00:00, 278.02it/s]
Array Shape :  (186, 1, 32, 32, 3)
Image Shape :  (186, 32, 32, 3)
'''
