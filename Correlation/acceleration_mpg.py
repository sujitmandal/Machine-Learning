import numpy as np 
import pandas as pd
import seaborn as sns
from numpy import std
from numpy import mean
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

'''
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal
'''

data = pd.read_csv('car.csv')
#print(data.head())

'''
   Acceleration  Cylinders  Displacement  Horsepower  Model_Year  Weight   Origin   MPG
0          12.0          8         307.0         130          70    3504  USA      18.0
1          11.5          8         350.0         165          70    3693  USA      15.0
2          11.0          8         318.0         150          70    3436  USA      18.0
3          12.0          8         304.0         150          70    3433  USA      16.0
4          10.5          8         302.0         140          70    3449  USA      17.0
'''

acceleration = data['Acceleration']
'''
cylinders = data['Cylinders']
displacement = data['Displacement']
horsepower = data['Horsepower']
model_Year = data['Model_Year']
weight = data['Weight']
origin = data['Origin']
'''
mpg = data['MPG']

data_one = pd.DataFrame(data)
data_two = data_one.drop(['Cylinders', 'Displacement', 'Horsepower', 'Model_Year', 'Weight', 'Origin'], axis = 1)
print(data_two.head())
print('\n')

mse = mean_squared_error(acceleration, mpg)
print('Mean Squared Error = %0.3f' % mse)

rmes = mean_squared_error(acceleration, mpg, squared=False)
print('Root Mean Squared = %0.3f' % rmes)

acceleration_mean = mean(acceleration)
acceleration_stdv = std(acceleration)

mpg_mean = mean(mpg)
mpg_stdv = std(mpg)

print('Acceleration Mean = %0.3f' % acceleration_mean)
print('Acceleration Standard Deviation = %0.3f' % acceleration_stdv)
print('MPG Mean = %0.3f' % mpg_mean)
print('MPG Standard Deviation = %0.3f' % mpg_stdv)


#without regression
sns.pairplot(data_two, kind='scatter')
plt.show()

plt.title('Acceleration vs MPG')
plt.xlabel('Acceleration')
plt.ylabel('MPG')
plt.scatter(acceleration, mpg)
plt.show()


#with regression
sns.pairplot(data_two, kind='reg')
plt.show()

#OUTPUT:
'''
   Acceleration   MPG
0          12.0  18.0
1          11.5  15.0
2          11.0  18.0
3          12.0  16.0
4          10.5  17.0


Mean Squared Error = 115.032
Root Mean Squared = 10.725
Acceleration Mean = 15.520
Acceleration Standard Deviation = 2.800
MPG Mean = 23.051
MPG Standard Deviation = 8.391
'''
