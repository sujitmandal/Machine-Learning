#Import required libraries
import os 
import cv2 
import numpy as np 
from tqdm import tqdm
import tensorflow as tf
from random import shuffle 
from tensorflow import keras
import matplotlib.pyplot as plt

#Github: https://github.com/sujitmandal
#This programe is create by Sujit Mandal
"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
Facebook : https://www.facebook.com/sujit.mandal.33671748
Twitter : https://twitter.com/mandalsujit37
"""

#Read The Dataset
train_pneumonia_dir = ('/media/sujit/2C1EFB771EFB3902/Pneumonia Dataset/train/Pneumonia')
train_normal_dir = ('/media/sujit/2C1EFB771EFB3902/Pneumonia Dataset/train/Normal')
test_pneumonia_dir = ('/media/sujit/2C1EFB771EFB3902/Pneumonia Dataset/test/Pneumonia')
test_normal_dir = ('/media/sujit/2C1EFB771EFB3902/Pneumonia Dataset/test/Normal')


image_size = 1000
#image_size = int(input('Enter The Image Size [32, 64, 128] :'))

#Label The Images
def label_image(image): 
    word_label = image.split('.')[-3]
    
    if word_label == 'Pneumonia':
        return [0] 
    elif word_label == 'Normal':
        return [1] 
    else:
        print('Image not found!')

def train_pneumonia_image():
    train_pneumonia_image = []

    for image in tqdm(os.listdir(train_pneumonia_dir)): 
        path = os.path.join(train_pneumonia_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        train_pneumonia_image.append([np.array(image)])
    shuffle(train_pneumonia_image)

    return(train_pneumonia_image)

def train_pneumonia_label():
    train_pneumonia_label = []

    for image in tqdm(os.listdir(train_pneumonia_dir)): 
        label = label_image(image)
        path = os.path.join(train_pneumonia_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        train_pneumonia_label.append([np.array(label)])
    shuffle(train_pneumonia_label)

    return(train_pneumonia_label)

def train_normal_image():
    train_normal_image = []

    for image in tqdm(os.listdir(train_normal_dir)): 
        path = os.path.join(train_normal_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        train_normal_image.append([np.array(image)])
    shuffle(train_normal_image)

    return(train_normal_image)

def train_normal_label():
    train_normal_label = []

    for image in tqdm(os.listdir(train_normal_dir)): 
        label = label_image(image)
        path = os.path.join(train_normal_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        train_normal_label.append([np.array(label)])
    shuffle(train_normal_label)

    return(train_normal_label)
    
def test_pneumonia_image():
    test_pneumonia_image = []

    for image in tqdm(os.listdir(test_pneumonia_dir)): 
        path = os.path.join(test_pneumonia_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        test_pneumonia_image.append([np.array(image)])
    shuffle(test_pneumonia_image)

    return(test_pneumonia_image)


def test_pneumonia_label():
    test_pneumonia_label = []

    for image in tqdm(os.listdir(test_pneumonia_dir)): 
        label = label_image(image)
        path = os.path.join(test_pneumonia_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        test_pneumonia_label.append([np.array(label)])
    shuffle(test_pneumonia_label)

    return(test_pneumonia_label)

def test_normal_image():
    test_normal_image = []

    for image in tqdm(os.listdir(test_normal_dir)): 
        path = os.path.join(test_normal_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        test_normal_image.append([np.array(image)])
    shuffle(test_normal_image)

    return(test_normal_image)

def test_normal_label():
    test_normal_label = []

    for image in tqdm(os.listdir(test_normal_dir)): 
        label = label_image(image)
        path = os.path.join(test_normal_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        test_normal_label.append([np.array(label)])
    shuffle(test_normal_label)

    return(test_normal_label)

list_train_pneumonia_image = train_pneumonia_image()
list_pneumonia_label = train_pneumonia_label()
list_train_normal_image = train_normal_image()
list_train_normal_label = train_normal_label()
list_test_pneumonia_image = test_pneumonia_image()
list_test_pneumonia_label = test_pneumonia_label()
list_test_normal_image = test_normal_image()
list_test_normal_label = test_normal_label()

array_train_pneumonia_image = np.array(list_train_pneumonia_image)
array_train_pneumonia_label = np.array(list_pneumonia_label)
array_train_normal_image = np.array(list_train_normal_image)
array_train_normal_label = np.array(list_train_normal_label)
array_test_pneumonia_image = np.array(list_test_pneumonia_image)
array_test_pneumonia_label = np.array(list_test_pneumonia_label)
array_test_normal_image = np.array(list_test_normal_image)
array_test_normal_label = np.array(list_test_normal_label)

print(array_train_pneumonia_image.shape) 
print(array_train_pneumonia_label.shape) 
print(array_train_normal_image.shape)  
print(array_train_normal_label.shape)  
print(array_test_pneumonia_image.shape)  
print(array_test_pneumonia_label.shape)  
print(array_test_normal_image.shape)  
print(array_test_normal_label.shape)

train_pneumonia_image = array_train_pneumonia_image[:,0,:,:]
train_pneumonia_label = array_train_pneumonia_label[:,0,:]
train_normal_image = array_train_normal_image[:,0,:,:]
train_normal_label = array_train_normal_label[:,0,:]
test_pneumonia_image = array_test_pneumonia_image[:,0,:,:]
test_pneumonia_label = array_test_pneumonia_label[:,0,:]
test_normal_image = array_test_normal_image[:,0,:,:]
test_normal_label = array_test_normal_label[:,0,:]

print(train_pneumonia_image.shape) 
print(train_pneumonia_label.shape) 
print(train_normal_image.shape)  
print(train_normal_label.shape)  
print(test_pneumonia_image.shape)  
print(test_pneumonia_label.shape)  
print(test_normal_image.shape)  
print(test_normal_label.shape) 

train_pneumonia_image_class = ['Pneumonia', 'Normal']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_pneumonia_image[i], cmap=plt.cm.binary)
    plt.xlabel(train_pneumonia_image_class[train_pneumonia_label[i][0]])
plt.show()

train_normal_image_class = ['Pneumonia', 'Normal']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_normal_image[i], cmap=plt.cm.binary)
    plt.xlabel(train_normal_image_class[train_normal_label[i][0]])
plt.show()

test_pneumonia_image_class = ['Pneumonia', 'Normal']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_pneumonia_image[i], cmap=plt.cm.binary)
    plt.xlabel(test_pneumonia_image_class[test_pneumonia_label[i][0]])
plt.show()

test_normal_image_class = ['Pneumonia', 'Normal']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_normal_image[i], cmap=plt.cm.binary)
    plt.xlabel(test_normal_image_class[test_normal_label[i][0]])
plt.show()


train_images = np.append(train_pneumonia_image, train_normal_image, axis=0)
train_labels = np.append(train_pneumonia_label, train_normal_label, axis=0)
test_images = np.append(test_pneumonia_image, test_normal_image, axis=0)
test_labels = np.append(test_pneumonia_label, test_normal_label, axis=0)

print(train_images.shape) 
print(train_labels.shape) 
print(test_images.shape)  
print(test_labels.shape)

#Save The Array in The Local Disk....
np.save('Dataset/train_images.npy',train_images)
np.save('Dataset/train_labels.npy',train_labels)
np.save('Dataset/test_images.npy',test_images)
np.save('Dataset/test_labels.npy',test_labels)
print('\ntrain_images.npy is saved on the Current Directory.....')
print('train_labels.npy is saved on the Current Directory.....')
print('test_images.npy is saved on the Current Directory.....')
print('test_labels.npy is saved on the Current Directory.....')
