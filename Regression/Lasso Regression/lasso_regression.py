# Import required libraries
import numpy as np
import pandas as pd 
from numpy import std
from numpy import mean
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error


#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal

LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/

Facebook : https://www.facebook.com/sujit.mandal.33671748

Twitter : https://twitter.com/mandalsujit37
"""
#Read Data
data = pd.read_csv('Salary_Data.csv')

#Data Visualition
print(data.head(5))
print('\n')
print(data.tail(5))
print('\n')
print(data.shape)

#Data Processing
x = data['YearsExperience'].values.reshape(-1,1)
y = data['Salary'].values.reshape(-1,1)

xnew = x[20:30]
ynew = y[20:30]
x = x[:20]
y = y[:20]

#Data Visualition After Processing
print('\n')
print('xnew:',xnew.shape)
print('ynew:',ynew.shape)
print('x:',x.shape)
print('y:',y.shape)

#Scatter Plot
plt.title('YearsExperience vs. Salary')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.scatter(x,y)
plt.show()


x_mean = mean(x)
x_stdv = std(x)

y_mean = mean(y)
y_stdv = std(y)

print('\n')
print('X Mean = %0.3f' % x_mean)
print('X Standard Deviation = %0.3f' %x_stdv)
print('\n')
print('Y Mean = %0.3f' % y_mean)
print('Y Standard Deviation = %0.3f' %y_stdv)

#Spearman's Correlation
correlation, _ = spearmanr(x, y)
print('\n')
print('Spearmans correlation: %.5f' % correlation)

#Regression Model
lr = linear_model.Lasso(alpha=1.0).fit(x, y)
print('\n')
print(lr)

intercept = (lr.intercept_)
print('\n')
print('Intercept: %.5f' % intercept)

#Prediction
predict = lr.predict(xnew)
print('\n')
print('Prediction:')
print(predict)

x_true = xnew
y_true = ynew
y_pred = predict

score = lr.score(y_true, y_pred)
print('\n')
print('Score: %.5f' % score)

#Coefficients
coef = (lr.coef_)
print('Coefficients: ', coef)

#R^2 (coefficient of determination)
r2_Score = r2_score(y_true, y_pred)
print('r2 Score : %.5f' % r2_Score)

#Root Mean Squared Error
rmse = sqrt(mean_squared_error(y_true, y_pred))
print('\n')
print('Model Result :')
print('Root Mean Squared Error = %0.3f' % rmse)

#Mean Squared Error
mse = mean_squared_error(y_true, y_pred)
print('Mean Squared Error = %0.3f' % mse)

#Mean Absolute Error
mae = mean_absolute_error(y_true, y_pred)
print('Mean Absolute Error = %0.3f' % mae)

#Median Absolute Error
med_ea = median_absolute_error(y_true, y_pred)
print('Median Absolute Error = %0.3f' % med_ea)

#Mean Squared Log Error
msle = mean_squared_log_error(y_true, y_pred)
print('Mean Squared Log Error = %0.3f' % msle)

#Max Error
me = max_error(y_true, y_pred)
print('Max Error = %0.3f' % me)

#Polt Actual vs. Predicted
plt.title('Actual vs. Predicted')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.scatter(x_true, y_true)
plt.scatter(x_true, y_pred)
plt.show()

#Outputs Plot
plt.title('Actual vs. Predicted')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.scatter(x_true, y_true)
plt.scatter(x_true, y_pred, color='r')
plt.plot(x_true, y_pred, color='y', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

#OUTPUT :
'''
   YearsExperience   Salary
0              1.1  39343.0
1              1.3  46205.0
2              1.5  37731.0
3              2.0  43525.0
4              2.2  39891.0


    YearsExperience    Salary
25              9.0  105582.0
26              9.5  116969.0
27              9.6  112635.0
28             10.3  122391.0
29             10.5  121872.0


(30, 2)


xnew: (10, 1)
ynew: (10, 1)
x: (20, 1)
y: (20, 1)


X Mean = 3.590
X Standard Deviation = 1.432


Y Mean = 59304.250
Y Standard Deviation = 14381.643


Spearmans correlation: 0.87058


Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
      normalize=False, positive=False, precompute=False, random_state=None,
      selection='cyclic', tol=0.0001, warm_start=False)


Intercept: 26579.15132


Prediction:
[ 88565.41065418  91300.09856578  98592.5996634  101327.287575
 105885.10076101 108619.78867262 113177.60185863 114089.16449583
 120470.10295624 122293.22823065]


Score: -8395486220.93957
Coefficients:  [9115.62637202]
r2 Score : 0.71529


Model Result :
Root Mean Squared Error = 5138.616
Mean Squared Error = 26405370.644
Mean Absolute Error = 3951.098
Median Absolute Error = 3105.189
Mean Squared Log Error = 0.002
Max Error = 12484.712
'''
