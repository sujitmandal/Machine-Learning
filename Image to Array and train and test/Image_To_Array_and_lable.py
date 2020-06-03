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

#This Function Will Label The Samle Data......
def label_image(image): 
    word_label = image.split('.')[-3]
    
    if word_label == 'mask':
        return [1] 
    elif word_label == 'nomask':
        return [2] 
    else:
        print('Image not found!')

#This Function Will Convert Image Into List....
def images():
    list_image = []

    for image in tqdm(os.listdir(dataset_dir)): 
        path = os.path.join(dataset_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        list_image.append([np.array(image)])
    shuffle(list_image)

    return(list_image)

#This Function Will Convert Label Image Into List....
def image_label():
    list_label = []

    for image in tqdm(os.listdir(dataset_dir)): 
        label = label_image(image)
        path = os.path.join(dataset_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        list_label.append([np.array(label)])
    shuffle(list_label)

    return(list_label)

#Call The Function And Pass it to a variable....mask_images.npy
list_images = images()
list_labels = image_label()
print('\n')
print('Data Info....')
print('Image Data Type : ', type(list_images))
print('Image Label Data Type : ', type(list_labels))

#Convert List Into Array....
array_image = np.array(list_images)
array_image_label = np.array(list_labels)
print('\nAfter Converted List Into Array...')
print('Images....')
print('Image Data Type : ' ,type(array_image))
print('Image Shape of The Data : ', array_image.shape)
print('Image Dimention of The Data : ', array_image.ndim)
print('Images Labels....')
print('Image Label Data Type : ' ,type(array_image_label))
print('Image Label Shape of The Data : ', array_image_label.shape)
print('Image Label Dimention of The Data : ', array_image_label.ndim)

#Change The Dimention of The Array....
array_images = array_image[:,0,:,:]
array_images_labels = array_image_label[:,0,:]
print('\nAfter Changed Dimention...')
print('Images....')
print('Image Shape of The Data : ', array_images.shape)
print('Image Dimention of The Data : ', array_images.ndim)
print('Images Labels....')
print('Image Shape of The Data : ', array_images_labels.shape)
print('Image Dimention of The Data : ', array_images_labels.ndim)

#Save The Array in The Local Disk....
np.save('mask_images.npy',array_images)
np.save('mask_images_lebels.npy',array_images_labels)
print('\nmask_images.npy is saved on the Current Directory.....')
print('mask_images_lebels.npy is saved on the Current Directory.....')

#OUTPUT: 
'''
    Enter The Image Size [32, 64, 128] :32
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1706/1706 [00:02<00:00, 768.31it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1706/1706 [00:02<00:00, 638.75it/s]


    Data Info....
    Image Data Type :  <class 'list'>
    Image Label Data Type :  <class 'list'>

    After Converted List Into Array...
    Images....
    Image Data Type :  <class 'numpy.ndarray'>
    Image Shape of The Data :  (1706, 1, 32, 32, 3)
    Image Dimention of The Data :  5
    Images Labels....
    Image Label Data Type :  <class 'numpy.ndarray'>
    Image Label Shape of The Data :  (1706, 1, 1)
    Image Label Dimention of The Data :  3

    After Changed Dimention...
    Images....
    Image Shape of The Data :  (1706, 32, 32, 3)
    Image Dimention of The Data :  4
    Images Labels....
    Image Shape of The Data :  (1706, 1)
    Image Dimention of The Data :  2

    mask_images.npy is saved on the Current Directory.....
    mask_images_lebels.npy is saved on the Current Directory..... 
'''