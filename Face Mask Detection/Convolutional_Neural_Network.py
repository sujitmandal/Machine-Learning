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

train_images = np.load('dataset/train_images.npy')
train_labels = np.load('dataset/train_labels.npy')
test_images = np.load('dataset/test_images.npy')
test_labels = np.load('dataset/test_labels.npy')

train_images = train_images / 255.0
test_images = test_images / 255.0

EPOCHS = 20
image_size = int(input('Enter The Image Size [32, 64, 128] : '))

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

ch_path = ('save/cp.ckpt')
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

model = build_model()
loss, acc = model.evaluate(test_images, test_labels)
print('restored model, accuracy: {:5.2f}%'.format(100*acc))

model.load_weights(ch_path)
loss, acc = model.evaluate(test_images, test_labels)
print('restored model, accuracy: {:5.2f}%'.format(100*acc))

ch_path_2 = ('save/cp-{epoch:04d}.ckpt')
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
model.save_weights('./save/my_save')

#restore
model = build_model()
model.load_weights('./save/my_save')

loss, acc = model.evaluate(test_images, test_labels)
print('restored model, accuracy: {:5.2f}%'.format(100*acc))


model = build_model()
model.fit(train_images, train_labels, epochs=15)

#save entire model to a HDF5 file
model.save('saved model/my_model.h5')

new_model = keras.models.load_model('saved model/my_model.h5')
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

#OUTPUT:
'''
Enter The Image Size [32, 64, 128] : 64

Epoch 1/20
31/32 [============================>.] - ETA: 0s - loss: 0.6238 - accuracy: 0.6996
Epoch 00001: saving model to save/cp.ckpt
32/32 [==============================] - 3s 106ms/step - loss: 0.6208 - accuracy: 0.7010 - val_loss: 0.2974 - val_accuracy: 0.8883
Epoch 2/20
31/32 [============================>.] - ETA: 0s - loss: 0.1740 - accuracy: 0.9335
Epoch 00002: saving model to save/cp.ckpt
32/32 [==============================] - 4s 130ms/step - loss: 0.1729 - accuracy: 0.9340 - val_loss: 0.1886 - val_accuracy: 0.9255
Epoch 3/20
31/32 [============================>.] - ETA: 0s - loss: 0.1096 - accuracy: 0.9647
Epoch 00003: saving model to save/cp.ckpt
32/32 [==============================] - 4s 121ms/step - loss: 0.1089 - accuracy: 0.9650 - val_loss: 0.1388 - val_accuracy: 0.9628
Epoch 4/20
31/32 [============================>.] - ETA: 0s - loss: 0.0769 - accuracy: 0.9778
Epoch 00004: saving model to save/cp.ckpt
32/32 [==============================] - 4s 113ms/step - loss: 0.0764 - accuracy: 0.9780 - val_loss: 0.0619 - val_accuracy: 0.9840
Epoch 5/20
31/32 [============================>.] - ETA: 0s - loss: 0.0671 - accuracy: 0.9758
Epoch 00005: saving model to save/cp.ckpt
32/32 [==============================] - 4s 126ms/step - loss: 0.0667 - accuracy: 0.9760 - val_loss: 0.0597 - val_accuracy: 0.9814
Epoch 6/20
31/32 [============================>.] - ETA: 0s - loss: 0.0433 - accuracy: 0.9869
Epoch 00006: saving model to save/cp.ckpt
32/32 [==============================] - 7s 230ms/step - loss: 0.0430 - accuracy: 0.9870 - val_loss: 0.0397 - val_accuracy: 0.9867
Epoch 7/20
31/32 [============================>.] - ETA: 0s - loss: 0.0316 - accuracy: 0.9879
Epoch 00007: saving model to save/cp.ckpt
32/32 [==============================] - 6s 176ms/step - loss: 0.0313 - accuracy: 0.9880 - val_loss: 0.0407 - val_accuracy: 0.9947
Epoch 8/20
31/32 [============================>.] - ETA: 0s - loss: 0.0237 - accuracy: 0.9909
Epoch 00008: saving model to save/cp.ckpt
32/32 [==============================] - 3s 109ms/step - loss: 0.0235 - accuracy: 0.9910 - val_loss: 0.0547 - val_accuracy: 0.9840
Epoch 9/20
31/32 [============================>.] - ETA: 0s - loss: 0.0206 - accuracy: 0.9940
Epoch 00009: saving model to save/cp.ckpt
32/32 [==============================] - 4s 122ms/step - loss: 0.0204 - accuracy: 0.9940 - val_loss: 0.0544 - val_accuracy: 0.9814
Epoch 10/20
31/32 [============================>.] - ETA: 0s - loss: 0.0261 - accuracy: 0.9909
Epoch 00010: saving model to save/cp.ckpt
32/32 [==============================] - 4s 118ms/step - loss: 0.0259 - accuracy: 0.9910 - val_loss: 0.0494 - val_accuracy: 0.9947
Epoch 11/20
31/32 [============================>.] - ETA: 0s - loss: 0.0300 - accuracy: 0.9899
Epoch 00011: saving model to save/cp.ckpt
32/32 [==============================] - 4s 121ms/step - loss: 0.0298 - accuracy: 0.9900 - val_loss: 0.4765 - val_accuracy: 0.8324
Epoch 12/20
31/32 [============================>.] - ETA: 0s - loss: 0.0878 - accuracy: 0.9677
Epoch 00012: saving model to save/cp.ckpt
32/32 [==============================] - 8s 260ms/step - loss: 0.0873 - accuracy: 0.9680 - val_loss: 0.0451 - val_accuracy: 0.9840
Epoch 13/20
31/32 [============================>.] - ETA: 0s - loss: 0.0211 - accuracy: 0.9919
Epoch 00013: saving model to save/cp.ckpt
32/32 [==============================] - 4s 112ms/step - loss: 0.0209 - accuracy: 0.9920 - val_loss: 0.0964 - val_accuracy: 0.9761
Epoch 14/20
31/32 [============================>.] - ETA: 0s - loss: 0.0179 - accuracy: 0.9919
Epoch 00014: saving model to save/cp.ckpt
32/32 [==============================] - 4s 123ms/step - loss: 0.0178 - accuracy: 0.9920 - val_loss: 0.0601 - val_accuracy: 0.9947
Epoch 15/20
31/32 [============================>.] - ETA: 0s - loss: 0.0109 - accuracy: 0.9960
Epoch 00015: saving model to save/cp.ckpt
32/32 [==============================] - 4s 116ms/step - loss: 0.0109 - accuracy: 0.9960 - val_loss: 0.0589 - val_accuracy: 0.9947
Epoch 16/20
31/32 [============================>.] - ETA: 0s - loss: 0.0075 - accuracy: 0.9980    
Epoch 00016: saving model to save/cp.ckpt
32/32 [==============================] - 5s 153ms/step - loss: 0.0075 - accuracy: 0.9980 - val_loss: 0.0569 - val_accuracy: 0.9947
Epoch 17/20
31/32 [============================>.] - ETA: 0s - loss: 0.0035 - accuracy: 0.9990
Epoch 00017: saving model to save/cp.ckpt
32/32 [==============================] - 4s 128ms/step - loss: 0.0035 - accuracy: 0.9990 - val_loss: 0.0561 - val_accuracy: 0.9947
Epoch 18/20
31/32 [============================>.] - ETA: 0s - loss: 0.0102 - accuracy: 0.9950    
Epoch 00018: saving model to save/cp.ckpt
32/32 [==============================] - 4s 124ms/step - loss: 0.0101 - accuracy: 0.9950 - val_loss: 0.0714 - val_accuracy: 0.9947
Epoch 19/20
31/32 [============================>.] - ETA: 0s - loss: 0.0079 - accuracy: 0.9990
Epoch 00019: saving model to save/cp.ckpt
32/32 [==============================] - 4s 137ms/step - loss: 0.0078 - accuracy: 0.9990 - val_loss: 0.0726 - val_accuracy: 0.9920
Epoch 20/20
31/32 [============================>.] - ETA: 0s - loss: 0.0044 - accuracy: 0.9990    
Epoch 00020: saving model to save/cp.ckpt
32/32 [==============================] - 4s 113ms/step - loss: 0.0044 - accuracy: 0.9990 - val_loss: 0.0654 - val_accuracy: 0.9947
12/12 - 0s - loss: 0.0654 - accuracy: 0.9947
12/12 [==============================] - 0s 20ms/step - loss: 2.2295 - accuracy: 0.1835
restored model, accuracy: 18.35%
12/12 [==============================] - 0s 26ms/step - loss: 0.0654 - accuracy: 0.9947
restored model, accuracy: 99.47%
WARNING:tensorflow:`period` argument is deprecated. Please use `save_freq` to specify the frequency in number of batches seen.

Epoch 00005: saving model to save/cp-0005.ckpt

Epoch 00010: saving model to save/cp-0010.ckpt

Epoch 00015: saving model to save/cp-0015.ckpt

Epoch 00020: saving model to save/cp-0020.ckpt
12/12 [==============================] - 0s 24ms/step - loss: 0.0677 - accuracy: 0.9947
restored model, accuracy: 99.47%
Epoch 1/15
32/32 [==============================] - 3s 87ms/step - loss: 0.7427 - accuracy: 0.6330
Epoch 2/15
32/32 [==============================] - 3s 86ms/step - loss: 0.2661 - accuracy: 0.8980
Epoch 3/15
32/32 [==============================] - 3s 85ms/step - loss: 0.1495 - accuracy: 0.9400
Epoch 4/15
32/32 [==============================] - 3s 85ms/step - loss: 0.1038 - accuracy: 0.9630
Epoch 5/15
32/32 [==============================] - 3s 86ms/step - loss: 0.0763 - accuracy: 0.9730
Epoch 6/15
32/32 [==============================] - 3s 86ms/step - loss: 0.0602 - accuracy: 0.9830
Epoch 7/15
32/32 [==============================] - 3s 86ms/step - loss: 0.0666 - accuracy: 0.9790
Epoch 8/15
32/32 [==============================] - 3s 87ms/step - loss: 0.0490 - accuracy: 0.9840
Epoch 9/15
32/32 [==============================] - 3s 86ms/step - loss: 0.0424 - accuracy: 0.9840
Epoch 10/15
32/32 [==============================] - 3s 86ms/step - loss: 0.0364 - accuracy: 0.9850
Epoch 11/15
32/32 [==============================] - 3s 85ms/step - loss: 0.0427 - accuracy: 0.9820
Epoch 12/15
32/32 [==============================] - 3s 86ms/step - loss: 0.0531 - accuracy: 0.9840
Epoch 13/15
32/32 [==============================] - 3s 86ms/step - loss: 0.0246 - accuracy: 0.9940
Epoch 14/15
32/32 [==============================] - 3s 86ms/step - loss: 0.0120 - accuracy: 0.9970
Epoch 15/15
32/32 [==============================] - 3s 86ms/step - loss: 0.0095 - accuracy: 0.9990
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_15 (Conv2D)           (None, 62, 62, 32)        896       
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 31, 31, 32)        0         
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 29, 29, 64)        18496     
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 12, 12, 64)        36928     
_________________________________________________________________
flatten_5 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_10 (Dense)             (None, 64)                589888    
_________________________________________________________________
dense_11 (Dense)             (None, 10)                650       
=================================================================
Total params: 646,858
Trainable params: 646,858
Non-trainable params: 0
_________________________________________________________________
12/12 [==============================] - 0s 24ms/step - loss: 0.0528 - accuracy: 0.9947
restored model, accuracy: 99.47%
12/12 - 0s - loss: 0.0528 - accuracy: 0.9947
'''