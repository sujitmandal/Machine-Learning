#Import required libraries
import os 
import cv2 
import numpy as np 
from tqdm import tqdm
import tensorflow as tf
from random import shuffle 
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers

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
train_images = np.load('Dataset/64/train_images.npy')
train_labels = np.load('Dataset/64/train_labels.npy')
test_images = np.load('Dataset/64/test_images.npy')
test_labels = np.load('Dataset/64/test_labels.npy')

#Normalized
train_images = train_images / 255.0
test_images = test_images / 255.0


image_size = 64
#image_size = int(input('Enter The Image Size [32, 64, 128] :'))
EPOCHS = 20

#Convolutional Neural Network(CNN) building
def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        
    return model
        
model = build_model()

ch_path = ('save/64/cp.ckpt')
cp_dir = os.path.dirname(ch_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(ch_path,
                                                 save_weights_only = True,
                                                 verbose = 1)

model = build_model()

#Train the model
history = model.fit(train_images, train_labels, epochs=EPOCHS, 
                    validation_data=(test_images, test_labels), callbacks = [cp_callback])


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('Accuracy: {:5.2f}%'.format(100*test_acc))

model = build_model()
loss, acc = model.evaluate(test_images, test_labels)
print('restored model, accuracy: {:5.2f}%'.format(100*acc))

model.load_weights(ch_path)
loss, acc = model.evaluate(test_images, test_labels)
print('restored model, accuracy: {:5.2f}%'.format(100*acc))

ch_path_2 = ('save/64/cp-{epoch:04d}.ckpt')
cp_dir_2 = os.path.dirname(ch_path_2)

cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(ch_path_2, 
                                                   save_weights_only =  True, 
                                                   verbose = 1,
                                                   period = 5)


model = build_model()
#Train the model
history_2 = model.fit(train_images, train_labels, 
                          epochs=EPOCHS, 
                      validation_data=(test_images, test_labels), 
                      callbacks = [cp_callback_2],
                      verbose = 0
                       )

latest_model = tf.train.latest_checkpoint(cp_dir_2)

#save
model.save_weights('./save/64/my_save')
#restore
model = build_model()
model.load_weights('./save/64/my_save')

loss, acc = model.evaluate(test_images, test_labels)
print('restored model, accuracy: {:5.2f}%'.format(100*acc))

model = build_model()
model.fit(train_images, train_labels, epochs=15)

#save entire model to a HDF5 file
model.save('saved model/64/my_model.h5')

new_model = keras.models.load_model('saved model/64/my_model.h5')
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels)
print('restored model, accuracy: {:5.2f}%'.format(100*acc))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('Final Model, accuracy: {:5.2f}%'.format(100*test_acc))