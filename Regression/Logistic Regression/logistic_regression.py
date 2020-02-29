import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression

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
print('\n')

#Regression Model
lr = LogisticRegression().fit(x, y)
print('\n')
print(lr)

score = lr.score(x,y)
print('\n')
print('Score: %.5f' % score)

#R^2 (coefficient of determination)
r2_Score = r2_score(x, y)
print('\n')
print('r2 Score : %.5f' % r2_Score)

intercept = (lr.intercept_)
print('\n')
print('Intercept:')
intercepts = intercept.reshape(-1, 1)
print(intercepts)

#Prediction
predict = lr.predict(xnew)
print('\n')
predicted = predict.reshape(-1,1)
print('Prediction:')
print(predicted)

#OUTPUT:

'''       YearsExperience   Salary
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


Spearmans correlation: 0.87058


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)


Score: 0.15000


r2 Score : -1816363021.30743


Intercept:
[[ 2.79530592]
 [ 3.21443717]
 [ 2.03805287]
 [ 2.2589182 ]
 [ 3.00564501]
 [ 0.83487464]
 [-0.30293154]
 [ 1.21722046]
 [-0.30293154]
 [-0.45927974]
 [ 0.14603357]
 [ 1.09211729]
 [-1.12215049]
 [-0.15002328]
 [ 0.83487464]
 [-2.24709468]
 [-1.85290571]
 [-4.04340728]
 [-2.662184  ]
 [-4.2945715 ]]


Prediction:
[[93940.]
 [93940.]
 [93940.]
 [93940.]
 [93940.]
 [93940.]
 [93940.]
 [93940.]
 [93940.]
 [93940.]]'''