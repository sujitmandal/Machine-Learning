from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  
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


#x = data['YearsExperience']
#y = data['Salary']

#print(x)
#print('\n')
#print(y)


x = data["YearsExperience"].values.reshape(-1,1)
y = data['Salary'].values.reshape(-1,1)

#print(x)
#print('\n')
#print(y)
print(x.shape)
print(y.shape)


train_and_test= (lin.fit(x,y))
print("Train and Test:", train_and_test)

intercept = (lin.intercept_)
print("Intercept:", intercept)

coefficient = (lin.coef_)
print("Coefficient :", coefficient)

prediction = lin.predict([[25]]) #value of x is 25  then applying we get y 262041.25823505
print("Prediction:", prediction)

plt.plot(intercept, coefficient)
#plt.show()