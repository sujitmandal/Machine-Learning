import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

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
x = x[:20]
y = y[:20]

#Data Visualition After Processing
print('\n')
print('xnew:',xnew.shape)
print('x:',x.shape)
print('y:',y.shape)

#Scatter Plot
plt.title('YearsExperience vs. Salary')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.scatter(x,y)
plt.show()

#Spearman's Correlation
correlation, _ = spearmanr(x, y)
print('\n')
print('Spearmans correlation: %.5f' % correlation)

#Regression Model
br = linear_model.BayesianRidge().fit(x,y)
print('\n')
print(br)

score = br.score(x,y)
print('\n')
print('Score: %.5f' % score)

#R^2 (coefficient of determination)
r2_Score = r2_score(x, y)
print('\n')
print('r2 Score : %.5f' % r2_Score)

intercept = (br.intercept_)
print('\n')
print('Intercept: %.5f' % intercept)

#Prediction
predict = br.predict(xnew)
predicted = predict.reshape(-1,1)
print(predicted)

#OUTPUT:

'''   YearsExperience   Salary
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
x: (20, 1)
y: (20, 1)


BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, alpha_init=None,
              compute_score=False, copy_X=True, fit_intercept=True,
              lambda_1=1e-06, lambda_2=1e-06, lambda_init=None, n_iter=300,
              normalize=False, tol=0.001, verbose=False)


Score: 0.82353


r2 Score : -1816363021.30743


Intercept: 26946.23053
[[ 88237.18662607]
 [ 90941.19939487]
 [ 98151.90011165]
 [100855.91288044]
 [105362.60082842]
 [108066.61359722]
 [112573.3015452 ]
 [113474.6391348 ]
 [119784.00226198]
 [121586.67744118]]'''