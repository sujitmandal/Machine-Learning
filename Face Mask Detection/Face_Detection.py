import os
import cv2
import requests
import numpy as np
import tensorflow as tf
from keras.models import load_model

#Github: https://github.com/sujitmandal
#This programe is create by Sujit Mandal
"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
Facebook : https://www.facebook.com/sujit.mandal.33671748
Twitter : https://twitter.com/mandalsujit37
"""

laptopCam = cv2.VideoCapture(0)

model = tf.keras.models.load_model('saved model/my_model.h5')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
    if not os.path.exists('Save Video'):
        os.makedirs('Save Video')

except OSError:
    print('Error: Creating directory!')

frame_width = int(laptopCam.get(3))
frame_height = int(laptopCam.get(4))
saveVideo = cv2.VideoWriter('./Save Video/output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

image_size = int(input('Enter The Image Size [32, 64, 128] :'))

labels_dictionary = {
    0:'With_Mask',
    1:'Without_Mask'
    }

color_dictionary = {
    0:(0,255,0),
    1:(0,0,255)
    }


def predioction(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray)
    
    for(x, y, w, h) in faces:
        img_face = gray[y:y+w,x:x+w]
        resized = cv2.resize(img_face,(image_size,image_size))
        normalized = np.resize(resized, (1, image_size, image_size, 3))
        
        prediction = model.predict(normalized)
            

        label = np.argmax(prediction, axis=1)[0]

        cv2.rectangle(frame,(x,y),(x+w,y+h), color_dictionary[label],2)
        cv2.rectangle(frame,(x,y-50),(x+w,y), color_dictionary[label],-1)
        cv2.putText(frame, labels_dictionary[label], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 2)
    return(frame)

def Surveillance():
    while(True):
        check, lapCam = laptopCam.read()
     
        key = cv2.waitKey(1)

        if check==True:
            face = predioction(lapCam)

            saveVideo.write(face)
            cv2.imshow("frame",face)  
        else:
            break

        if key == ord('q'):
            break
            
    laptopCam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Surveillance()