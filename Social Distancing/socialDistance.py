import os
import sys
import cv2
import imutils
import requests
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

'''
# Author : Sujit Mandal
Github : https://github.com/sujitmandal
Pypi : https://pypi.org/user/sujitmandal/
LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/
Facebook : https://www.facebook.com/sujit.mandal.33671748
Twitter : https://twitter.com/mandalsujit37
'''

model = cv2.dnn.readNet('model/MobileNetSSD_deploy.prototxt.txt', 'model/MobileNetSSD_deploy.caffemodel')
laptopCam = cv2.VideoCapture('videos/abc.mp4') #privide any video 
#laptopCam = cv2.VideoCapture(0)
portAddress = ('http://192.168.0.100:8080//')

'''laptopCam.set(cv2.CAP_PROP_FPS, 10)
FPS = int(laptopCam.get(6))'''

try:
    if not os.path.exists('Save Video'):
        os.makedirs('Save Video')

except OSError:
    print('Error: Creating directory!')

saveVideo = cv2.VideoWriter('./Save Video/output.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))


def IpCam():
    ipCamRequest = requests.get(portAddress + 'shot.jpg')
    ipCamArray = np.array(bytearray(ipCamRequest.content), dtype=np.uint8)
    ipcam = cv2.imdecode(ipCamArray, -1)
    return(ipcam)

def classLable():
    tempList = []
    with open('model/class_labels.names', 'r') as text:
        tempList = [line.strip() for line in text.readlines()]
    return(tempList)

def color():
    tempList = classLable()
    colors = np.random.uniform(0, 255, size=(len(tempList), 3))
    return(colors)

labels = classLable()
boxColor = color()
FocalLength = 615

def prediction(frame):
    start, end = frame.shape[:2]

    image = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    model.setInput(image)
    detect = model.forward()

    posDictionary = dict()
    coOrdiantes = dict()

    for i in range(detect.shape[2]):
        confidence = detect[0, 0, i, 2]

        if confidence > 0.2:
                    labelID = int(detect[0,0, i, 1])
                    box = detect[0, 0, i, 3:7] * np.array([end, start, end, start])
                    (x, y, h, w) = box.astype('int')

                    if labelID == 15.00:
                            cv2.rectangle(frame, (x, y), (h, w), boxColor[labelID], 2)
                            label = ('{}: {:0.2f}%'.format(labels[labelID], (confidence * 100)))
                            print('{}'.format(label))

                            coOrdiantes[i] = (x, y, h, w)

                            midX = round((x + h) / 2, 4)
                            midY = round((y + w) / 2, 4)
                            height = round(w - y, 4)

                    
                            distance = ((165 * FocalLength) / height) #unit (mm)
                            #distance = ((165 * FocalLength) / height)/ 10 #unit (cm)
                            print('Distance(CM):{dist:0.2f}\n'.format(dist = distance))

                            midXcm = ((midX * distance) / FocalLength)
                            midYcm = ((midY * distance) / FocalLength)
                            posDictionary[i] = (midXcm, midYcm, distance)
                           
    objects = set()
    for j in posDictionary.keys():
        for k in posDictionary.keys():
            if j < k:
                distances = sqrt(
                    pow(posDictionary[j][0] - posDictionary[k][0], 2) + pow(posDictionary[j][1] - posDictionary[k][1], 2) 
                )
                if distances < 200:
                    objects.add(j)
                    objects.add(k)                   
                            
    for l in posDictionary.keys():
        if l in objects:
            color = (0, 0 , 255)
        else:
            color = (0, 255, 0) 
        (x, y , h, w) = coOrdiantes[l]

        cv2.rectangle(frame, (x, y), (h,w), color, 2)
        #temp = y - 15 if y - 15 > 15 else y + 15
        #cv2.putText(frame, 'D: {l} CM.'.format(l = round(posDictionary[l][2] , 2)), (x, temp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return(frame)

def Surveillance():
    while(True):
        #ipcam = IpCam()
        check, lapCam = laptopCam.read()
        #lapCam = imutils.resize(lapCam, width=720)

        key = cv2.waitKey(1)
        if check==True:
            frame = prediction(lapCam)
            #ipcam = prediction(ipcam)
            saveVideo.write(frame)
            #saveVideo.write(ipcam)
            
            cv2.imshow("frame", frame)
            #cv2.imshow('IP_Cam', ipcam)
        else:
            break

        if key == ord('q'):
            break
        
    laptopCam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Surveillance()
