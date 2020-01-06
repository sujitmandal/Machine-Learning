import numpy as np #pip install numpy
import pandas as pd #pip install pandas
from sklearn.model_selection import KFold #pip install scikit-learn

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal
"""

data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
#data = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20])

fold = int(input(print("Enter the number of fold: ")))

kfold = KFold(fold, True, 1)

for tarining, testing in kfold.split(data):
    print("train: %s,   test: %s" %(data[tarining], data[testing]))