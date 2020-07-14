import os 
import re
import cv2 
import glob
import numpy as np 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import models
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

#Github: https://github.com/sujitmandal
#This programe is create by Sujit Mandal
"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
Facebook : https://www.facebook.com/sujit.mandal.33671748
Twitter : https://twitter.com/mandalsujit37
"""

model = tf.keras.models.load_model('saved model/my_model.h5')
image_size = 32

labels_dictionary = {
    0:'zero [0]',
    1:'one [1]',
    2:'two [2]',
    3:'three [3]',
    4:'four [4]',
    5:'five [5]',
    6:'six [6]',
    7:'seven [7]',
    8:'eight [8]',
    9:'nine [9]'
    }

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page        
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image_file = request.files['file']

        basepath = os.path.dirname(__file__)
        image_path = os.path.join(
            basepath, 'uploads', secure_filename(image_file.filename)
            )

        image_file.save(image_path)
    return(image_path)

@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    images = upload()
    image = cv2.imread(images)

    resized = cv2.resize(image,(image_size, image_size))
    normalized = np.resize(resized, (1, image_size, image_size, 3))
        
    prediction = model.predict(normalized)

    label = np.argmax(prediction, axis=1)[0]
    predictions = labels_dictionary[label]
    print(predictions)

    return(predictions)
            
     
if __name__ == '__main__':
    app.run(
       debug=True, port=1500
    )
