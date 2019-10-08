from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd 
import numpy as np

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal

This programe is create by Sujit Mandal

"""

lin = LinearRegression()

data = pd.read_csv("Salary_Data.csv")

x = data["YearsExperience"].values.reshape(-1,1)
y = data['Salary'].values.reshape(-1,1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 15)

#print(x)
#print('\n')
#print(y)
print(x_test.shape)
print(x_train.shape)
print(y_test.shape)
print(y_train.shape)

train_and_test= (lin.fit(x,y))
print("Train and Test:", train_and_test)

intercept = (lin.intercept_)
print("Intercept:", intercept)

coefficient = (lin.coef_)
print("Coefficient :", coefficient)

prediction = lin.predict([[25]]) #value of x is 25  then applying we get y 262041.25823505
print("Prediction:", prediction)

prediction_x_test = lin.predict(x_test)
print("X test Prediction: ", prediction_x_test)

print('\n')

prediction_y_test = lin.predict(y_test)
print("Y test Prediction: ", prediction_y_test)

print("\n")

concatenate = np.concatenate((y_test, prediction_x_test), axis = 0)
print("concatenation of y_test and prediction_x_test:", concatenate)


predic  = metrics.mean_absolute_error(y_test, prediction_x_test)
#y_test is the real sales value of x_test
#prediction_x_test is the predicted sales value of x_test

print("overel prediction value:", predic)