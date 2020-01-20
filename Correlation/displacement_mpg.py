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
'''
acceleration = data['Acceleration']
cylinders = data['Cylinders']
'''
displacement = data['Displacement']
'''
horsepower = data['Horsepower']
model_Year = data['Model_Year']
weight = data['Weight']
origin = data['Origin']
'''
mpg = data['MPG']

data_one = pd.DataFrame(data)
data_two = data_one.drop(['Acceleration', 'Cylinders', 'Horsepower', 'Model_Year', 'Weight', 'Origin'], axis = 1)
print(data_two.head())
print('\n')

mse = mean_squared_error(displacement, mpg)
print('Mean Squared Error = %0.3f' % mse)

mes_without_squared = mean_squared_error(displacement, mpg, squared=False)
print('Mean Squared Error Without Squared = %0.3f' % mes_without_squared)

displacement_mean = mean(displacement)
displacement_stdv = std(displacement)

mpg_mean = mean(mpg)
mpg_stdv = std(mpg)

print('Displacement Mean = %0.3f' % displacement_mean)
print('Displacement Standard Deviation = %0.3f' % displacement_stdv)
print('MPG Mean = %0.3f' % mpg_mean)
print('MPG Standard Deviation = %0.3f' % mpg_stdv)


#without regression
sns.pairplot(data_two, kind='scatter')
plt.show()

plt.title('Displacement vs MPG')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.scatter(displacement, mpg)
plt.show()

#with regression
sns.pairplot(data_two, kind='reg')
plt.show()

#OUTPUT:
'''
   Displacement   MPG
0         307.0  18.0
1         350.0  15.0
2         318.0  18.0
3         304.0  16.0
4         302.0  17.0


Mean Squared Error = 41886.796
Mean Squared Error Without Squared = 204.663
Displacement Mean = 194.780
Displacement Standard Deviation = 104.793
MPG Mean = 23.051
MPG Standard Deviation = 8.391
'''
