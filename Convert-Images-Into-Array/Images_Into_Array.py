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

#images_path = input('Enter Image Folder Path : ') #Path of the images folder
#image_size = int(input('Enter The Image Size [32, 64, 128] : '))

images_path = ('/media/sujit/92EC423BEC4219BD/GitHub Preoject/ALL ML PROJECT/Face Mask Detection/face mask detection  dataset/test/without mask') #Path of the images folder
image_size = 32

def images(images_path, image_size):
    imges_list = []

    for image in tqdm(os.listdir(images_path)):
        path = os.path.join(images_path, image)

        image = cv2.imread(path)
        image = cv2.resize(image , (image_size, image_size))
        imges_list.append([np.array(image)])
    shuffle(imges_list)

   #Convert List Into Array
    array_image = np.array(imges_list)
  
    #Removed Dimention 
    images = array_image[:,0,:,:]
  
    return(images)

if __name__ == "__main__":
    images(images_path, image_size)


#OUTPUT :
'''
Enter Image Folder Path : /media/sujit/92EC423BEC4219BD/GitHub Preoject/ALL ML PROJECT/Face Mask Detection/face mask detection  dataset/test/without mask
Enter The Image Size [32, 64, 128] : 32
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 186/186 [00:00<00:00, 278.02it/s]
Array Shape :  (186, 1, 32, 32, 3)
Image Shape :  (186, 32, 32, 3)
'''
