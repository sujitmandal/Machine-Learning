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
train_dir = ('/media/sujit/04BDD44B39086607/All ML Dataset/mask dataset/train')
test_dir = ('/media/sujit/04BDD44B39086607/All ML Dataset/mask dataset/test')

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
def train_image():
    training_image= []

    for image in tqdm(os.listdir(train_dir)): 
        path = os.path.join(train_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        training_image.append([np.array(image)])
    shuffle(training_image)

    return(training_image)

#This Function Will Convert Label Image Into List....
def train_image_label():
    training_label = []

    for image in tqdm(os.listdir(train_dir)): 
        label = label_image(image)
        path = os.path.join(train_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        training_label.append([np.array(label)])
    shuffle(training_label)

    return(training_label)

#This Function Will Convert Image Into List....
def test_image():
    testing_image = []

    for image in tqdm(os.listdir(test_dir)): 
        path = os.path.join(test_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        testing_image.append([np.array(image)])
    shuffle(testing_image)

    return(testing_image)

#This Function Will Convert Label Image Into List....
def test_image_label():
    testing_label = []

    for image in tqdm(os.listdir(test_dir)): 
        label = label_image(image)
        path = os.path.join(test_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        testing_label.append([np.array(label)])
    shuffle(testing_label)

    return(testing_label)

#Call The Function And Pass it to a variable....mask_images.npy
list_train_image = train_image()
list_train_label = train_image_label()
list_test_image = test_image()
list_test_label = test_image_label()
print('\n')
print('Data Info....')
print('Train Image Data Type : ', type(list_train_image))
print('Train Image Label Data Type : ', type(list_train_label))
print('Test Image Data Type : ', type(list_test_image))
print('Test Image Label Data Type : ', type(list_test_label))

#Convert List Into Array....
train_array_image = np.array(list_train_image)
train_image_label = np.array(list_train_label)
test_array_image = np.array(list_test_image)
test_image_label = np.array(list_test_label)
print('\nAfter Converted List Into Array...')
print('Train Images....')
print('Train Image Data Type : ' ,type(train_array_image))
print('Train Image Shape of The Data : ', train_array_image.shape)
print('Train Image Dimention of The Data : ', train_array_image.ndim)
print('\nTrain Images Label....')
print('Train Image Label Data Type : ' ,type(train_image_label))
print('Train Image Label Shape of The Data : ', train_image_label.shape)
print('Train Image Label Dimention of The Data : ', train_image_label.ndim)
print('\nTest Images....')
print('Test Image Data Type : ' ,type(test_array_image))
print('Test Image Shape of The Data : ', test_array_image.shape)
print('Test Image Dimention of The Data : ', test_array_image.ndim)
print('\nTest Images Label....')
print('Test Image Label Data Type : ' ,type(test_image_label))
print('Test Image Label Shape of The Data : ', test_image_label.shape)
print('Test Image Label Dimention of The Data : ', test_image_label.ndim)


#Change The Dimention of The Array....
train_images = train_array_image[:,0,:,:]
train_labels = train_image_label[:,0,:]
test_images = test_array_image[:,0,:,:]
test_labels = test_image_label[:,0,:]

print('\nAfter Changed Dimention...')
print('Train Images....')
print('Image Shape of The Data : ', train_images.shape)
print('Image Dimention of The Data : ', train_images.ndim)
print('\nTrain Images Label....')
print('Image Shape of The Data : ', train_labels.shape)
print('Image Dimention of The Data : ', train_labels.ndim)
print('\nTest Images....')
print('Test Shape of The Data : ', test_images.shape)
print('Test Dimention of The Data : ', test_images.ndim)
print('\nTest Images Label....')
print('Test Shape of The Data : ', test_labels.shape)
print('Test Dimention of The Data : ', test_labels.ndim)


#Save The Array in The Local Disk....
np.save('train_images.npy',train_images)
np.save('train_labels.npy',train_labels)
np.save('test_images.npy',test_images)
np.save('test_labels.npy',test_labels)
print('\ntrain_images.npy is saved on the Current Directory.....')
print('train_labels.npy is saved on the Current Directory.....')
print('test_images.npy is saved on the Current Directory.....')
print('test_labels.npy is saved on the Current Directory.....')

#OUTPUT:
''' 
    Enter The Image Size [32, 64, 128] :32
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1706/1706 [00:04<00:00, 385.54it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1706/1706 [00:04<00:00, 409.14it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1051/1051 [00:02<00:00, 400.94it/s]
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1051/1051 [00:02<00:00, 406.39it/s]


    Data Info....
    Train Image Data Type :  <class 'list'>
    Train Image Label Data Type :  <class 'list'>
    Test Image Data Type :  <class 'list'>
    Test Image Label Data Type :  <class 'list'>

    After Converted List Into Array...
    Train Images....
    Train Image Data Type :  <class 'numpy.ndarray'>
    Train Image Shape of The Data :  (1706, 1, 32, 32, 3)
    Train Image Dimention of The Data :  5

    Train Images Label....
    Train Image Label Data Type :  <class 'numpy.ndarray'>
    Train Image Label Shape of The Data :  (1706, 1, 1)
    Train Image Label Dimention of The Data :  3

    Test Images....
    Test Image Data Type :  <class 'numpy.ndarray'>
    Test Image Shape of The Data :  (1051, 1, 32, 32, 3)
    Test Image Dimention of The Data :  5

    Test Images Label....
    Test Image Label Data Type :  <class 'numpy.ndarray'>
    Test Image Label Shape of The Data :  (1051, 1, 1)
    Test Image Label Dimention of The Data :  3

    After Changed Dimention...
    Train Images....
    Image Shape of The Data :  (1706, 32, 32, 3)
    Image Dimention of The Data :  4

    Train Images Label....
    Image Shape of The Data :  (1706, 1)
    Image Dimention of The Data :  2

    Test Images....
    Test Shape of The Data :  (1051, 32, 32, 3)
    Test Dimention of The Data :  4

    Test Images Label....
    Test Shape of The Data :  (1051, 1)
    Test Dimention of The Data :  2

    train_images.npy is saved on the Current Directory.....
    train_labels.npy is saved on the Current Directory.....
    test_images.npy is saved on the Current Directory.....
    test_labels.npy is saved on the Current Directory.....
'''
