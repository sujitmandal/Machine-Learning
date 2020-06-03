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

#Path oF The Dataset....
dataset_dir = ('/media/sujit/04BDD44B39086607/All ML Dataset/mask dataset/train')
#Set the Value for Image Size.....

image_size = int(input('Enter The Image Size [32, 64, 128] :'))

#Function it Will Convert Image Into List....
def images():
    list_image = []

    for image in tqdm(os.listdir(dataset_dir)): 
        path = os.path.join(dataset_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        list_image.append([np.array(image)])
    shuffle(list_image)

    return(list_image)

#Call The Function And Pass it to a variable....mask_images.npy
list_images = images()
print('\n')
print('Data Info....')
print('Data Type : ', type(list_images))

#Convert List Into Array....
array_image = np.array(list_images)
print('\nAfter Converted List Into Array...')
print('Data Type : ' ,type(array_image))
print('Shape of The Data : ', array_image.shape)
print('Dimention of The Data : ', array_image.ndim)

#Change The Dimention of The Array....
array_images = array_image[:,0,:,:]
print('\nAfter Changed Dimention...')
print('Shape of The Data : ', array_images.shape)
print('Dimention of The Data : ', array_images.ndim)

#Save The Array in The Local Disk....
np.save('masks_images.npy',array_images)
print('\nmass_images.npy is saved on the Current Directory.....')

#OUTPUT: 
'''    Enter The Image Size [32, 64, 128] :32
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1706/1706 [00:02<00:00, 697.09it/s]


    Data Info....
    Data Type :  <class 'list'>

    After Converted List Into Array...
    Data Type :  <class 'numpy.ndarray'>
    Shape of The Data :  (1706, 1, 32, 32, 3)
    Dimention of The Data :  5

    After Changed Dimention...
    Shape of The Data :  (1706, 32, 32, 3)
    Dimention of The Data :  4

    mask_images.npy is saved on the Current Directory.....'''