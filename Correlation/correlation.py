import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

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


#without regression
sns.pairplot(data, kind='scatter')
plt.show()

#with regression
sns.pairplot(data, kind='reg')
plt.show()