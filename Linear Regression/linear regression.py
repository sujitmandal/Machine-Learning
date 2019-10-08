import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal

This programe is create by Sujit Mandal

"""

data = pd.read_csv("Salary_Data.csv")

#print(data.head())

#x = data['YearsExperience']
#y = data['Salary']

#print(type(x))
#print(type(y))
x = data['YearsExperience'].values
y = data['Salary'].values
#plt.plot(x,y)
#plt.scatter(x,y)
#plt.bar(data['YearsExperience'], data['Salary'])
plt.show()

x_mean = x.mean()
y_mean = y.mean()

number = 0
denominator = 0

for i in range(len(x)):
    number += ((x[i] - x_mean)*(y[i] - y_mean))
    denominator += ((x[i] - x_mean)**2)
m = number/denominator

print(m)

c = (y_mean - m**x_mean)

print(c)

plt.scatter(x,y)
plt.show()

a = np.linspace(x.min(), y.max())
b = m*a+c

plt.plot(a, b, color = "red")
plt.scatter(x, y)
plt.show()

