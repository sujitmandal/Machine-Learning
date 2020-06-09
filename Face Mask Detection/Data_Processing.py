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

train_with_mask_dir = ('/media/sujit/3EE27BA5E27B6057/face mask detection  dataset/train/with mask')
train_without_mask_dir = ('/media/sujit/3EE27BA5E27B6057/face mask detection  dataset/train/without mask')
test_with_mask_dir = ('/media/sujit/3EE27BA5E27B6057/face mask detection  dataset/test/with mask')
test_without_mask_dir = ('/media/sujit/3EE27BA5E27B6057/face mask detection  dataset/test/without mask')

image_size = int(input('Enter The Image Size [32, 64, 128] : '))

def label_image(image): 
    word_label = image.split('.')[-3]
    
    if word_label == 'with-mask':
        return [0] 
    elif word_label == 'without-mask':
        return [1] 
    else:
        print('Image not found!')

def train_with_mask_image():
    train_mask_image = []

    for image in tqdm(os.listdir(train_with_mask_dir)): 
        path = os.path.join(train_with_mask_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        train_mask_image.append([np.array(image)])
    shuffle(train_mask_image)

    return(train_mask_image)

def train_with_mask_label():
    train_mask_label = []

    for image in tqdm(os.listdir(train_with_mask_dir)): 
        label = label_image(image)
        path = os.path.join(train_with_mask_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        train_mask_label.append([np.array(label)])
    shuffle(train_mask_label)

    return(train_mask_label)

def train_without_mask_image():
    train_nomask_image = []

    for image in tqdm(os.listdir(train_without_mask_dir)): 
        path = os.path.join(train_without_mask_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        train_nomask_image.append([np.array(image)])
    shuffle(train_nomask_image)

    return(train_nomask_image)

def train_without_mask_label():
    train_nomask_label = []

    for image in tqdm(os.listdir(train_without_mask_dir)): 
        label = label_image(image)
        path = os.path.join(train_without_mask_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        train_nomask_label.append([np.array(label)])
    shuffle(train_nomask_label)

    return(train_nomask_label)

def test_with_mask_image():
    test_mask_image = []

    for image in tqdm(os.listdir(test_with_mask_dir)): 
        path = os.path.join(test_with_mask_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        test_mask_image.append([np.array(image)])
    shuffle(test_mask_image)

    return(test_mask_image)

def test_with_mask_label():
    test_mask_label = []

    for image in tqdm(os.listdir(test_with_mask_dir)): 
        label = label_image(image)
        path = os.path.join(test_with_mask_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        test_mask_label.append([np.array(label)])
    shuffle(test_mask_label)

    return(test_mask_label)

def test_without_mask_image():
    test_nomask_image = []

    for image in tqdm(os.listdir(test_without_mask_dir)): 
        path = os.path.join(test_without_mask_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        test_nomask_image.append([np.array(image)])
    shuffle(test_nomask_image)

    return(test_nomask_image)

def test_without_mask_label():
    test_nomask_label = []

    for image in tqdm(os.listdir(test_without_mask_dir)): 
        label = label_image(image)
        path = os.path.join(test_without_mask_dir, image)
        
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        test_nomask_label.append([np.array(label)])
    shuffle(test_nomask_label)

    return(test_nomask_label)

list_train_mask_image = train_with_mask_image()
list_train_mask_label = train_with_mask_label()
list_train_without_mask_image = train_without_mask_image()
list_train_without_mask_label = train_without_mask_label()
list_test_mask_image = test_with_mask_image()
list_test_mask_label = test_with_mask_label()
list_test_without_mask_image = test_without_mask_image()
list_test_without_mask_label = test_without_mask_label()

array_train_mask_image = np.array(list_train_mask_image)
array_train_mask_label = np.array(list_train_mask_label)
array_train_without_mask_image = np.array(list_train_without_mask_image)
array_train_without_mask_label = np.array(list_train_without_mask_label)
array_test_mask_image = np.array(list_test_mask_image)
array_test_mask_label = np.array(list_test_mask_label)
array_test_without_mask_image = np.array(list_test_without_mask_image)
array_test_without_mask_label = np.array(list_test_without_mask_label)

print(array_train_mask_image.shape) 
print(array_train_mask_label.shape) 
print(array_train_without_mask_image.shape)  
print(array_train_without_mask_label.shape)  
print(array_test_mask_image.shape)  
print(array_test_mask_label.shape)  
print(array_test_without_mask_image.shape)  
print(array_test_without_mask_label.shape)  

train_mask_image = array_train_mask_image[:,0,:,:]
train_mask_label = array_train_mask_label[:,0,:]
train_without_mask_image = array_train_without_mask_image[:,0,:,:]
train_without_mask_label = array_train_without_mask_label[:,0,:]
test_mask_image = array_test_mask_image[:,0,:,:]
test_mask_label = array_test_mask_label[:,0,:]
test_without_mask_image = array_test_without_mask_image[:,0,:,:]
test_without_mask_label = array_test_without_mask_label[:,0,:]

print(train_mask_image.shape) 
print(train_mask_label.shape) 
print(train_without_mask_image.shape)  
print(train_without_mask_label.shape)  
print(test_mask_image.shape)  
print(test_mask_label.shape)  
print(test_without_mask_image.shape)  
print(test_without_mask_label.shape)

train_mask_image_class = ['with-mask', 'without-mask']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_mask_image[i], cmap=plt.cm.binary)
    plt.xlabel(train_mask_image_class[train_mask_label[i][0]])
plt.show()

test_mask_image_class = ['with-mask', 'without-mask']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_mask_image[i], cmap=plt.cm.binary)
    plt.xlabel(test_mask_image_class[test_mask_label[i][0]])
plt.show()


train_without_mask_image_class = ['with-mask', 'without-mask']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_without_mask_image[i], cmap=plt.cm.binary)
    plt.xlabel(train_without_mask_image_class[train_without_mask_label[i][0]])
plt.show()


test_without_mask_image_class = ['with-mask', 'without-mask']

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_without_mask_image[i], cmap=plt.cm.binary)
    plt.xlabel(test_without_mask_image_class[test_without_mask_label[i][0]])
plt.show()


train_images = np.append(train_mask_image, train_without_mask_image, axis=0)
train_labels = np.append(train_mask_label, train_without_mask_label, axis=0)
test_images = np.append(test_mask_image, test_without_mask_image, axis=0)
test_labels = np.append(test_mask_label, test_without_mask_label, axis=0)

print(train_images.shape) 
print(train_labels.shape) 
print(test_images.shape)  
print(test_labels.shape)


#Save The Array in The Local Disk....
np.save('dataset/train_images.npy',train_images)
np.save('dataset/train_labels.npy',train_labels)
np.save('dataset/test_images.npy',test_images)
np.save('dataset/test_labels.npy',test_labels)
print('\ntrain_images.npy is saved on the [dataset] Directory.....')
print('train_labels.npy is saved on the [dataset]  Directory.....')
print('test_images.npy is saved on the [dataset]  Directory.....')
print('test_labels.npy is saved on the [dataset]  Directory.....')

#OUTOUT:
'''
Enter The Image Size [32, 64, 128] : 64
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:01<00:00, 308.86it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:01<00:00, 252.57it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:01<00:00, 349.97it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:02<00:00, 225.16it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:02<00:00, 86.91it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:01<00:00, 105.10it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 186/186 [00:00<00:00, 214.53it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 186/186 [00:01<00:00, 184.15it/s]
(500, 1, 64, 64, 3)
(500, 1, 1)
(500, 1, 64, 64, 3)
(500, 1, 1)
(190, 1, 64, 64, 3)
(190, 1, 1)
(186, 1, 64, 64, 3)
(186, 1, 1)
(500, 64, 64, 3)
(500, 1)
(500, 64, 64, 3)
(500, 1)
(190, 64, 64, 3)
(190, 1)
(186, 64, 64, 3)
(186, 1)
(1000, 64, 64, 3)
(1000, 1)
(376, 64, 64, 3)
(376, 1)

train_images.npy is saved on the [dataset] Directory.....
train_labels.npy is saved on the [dataset]  Directory.....
test_images.npy is saved on the [dataset]  Directory.....
test_labels.npy is saved on the [dataset]  Directory.....
'''