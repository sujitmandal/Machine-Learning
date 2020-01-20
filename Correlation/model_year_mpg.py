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
displacement = data['Displacement'
horsepower = data['Horsepower']
'''
model_Year = data['Model_Year']
'''
weight = data['Weight']
origin = data['Origin']
'''
mpg = data['MPG']

data_one = pd.DataFrame(data)
data_two = data_one.drop(['Acceleration', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Origin'], axis = 1)
print(data_two.head())
print('\n')

mse = mean_squared_error(model_Year, mpg)
print('Mean Squared Error = %0.3f' % mse)

rmse = mean_squared_error(model_Year, mpg, squared=False)
print('Root Mean Squared Error = %0.3f' % rmse)

model_Year_mean = mean(model_Year)
model_Year_stdv = std(model_Year)

mpg_mean = mean(mpg)
mpg_stdv = std(mpg)

print('Model Year Mean = %0.3f' % model_Year_mean)
print('Model Year Standard Deviation = %0.3f' % model_Year_stdv)
print('MPG Mean = %0.3f' % mpg_mean)
print('MPG Standard Deviation = %0.3f' % mpg_stdv)


#without regression
sns.pairplot(data_two, kind='scatter')
plt.show()

plt.title('Model_Year vs MPG')
plt.xlabel('Model_Year')
plt.ylabel('MPG')
plt.scatter(model_Year, mpg)
plt.show()


#with regression
sns.pairplot(data_two, kind='reg')
plt.show()

#OUTPUT:
'''
   Model_Year   MPG
0          70  18.0
1          70  15.0
2          70  18.0
3          70  16.0
4          70  17.0


Mean Squared Error = 2842.829
Root Mean Squared Error = 53.318
Model Year Mean = 75.921
Model Year Standard Deviation = 3.744
MPG Mean = 23.051
MPG Standard Deviation = 8.391
'''
