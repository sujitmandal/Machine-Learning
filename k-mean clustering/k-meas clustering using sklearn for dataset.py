from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal

"""

data = pd.DataFrame({'x':[12,20,28,18,29,33,24,45,52,51,15,55,57,62,64,68,77,71],
                      'y':[39,39,30,52,54,46,55,59,63,70,66,62,58,23,14,8,19,7]})

plt.scatter(data['x'], data['y'])
plt.show()

kmeans=KMeans(n_clusters=3,max_iter=500)
k_means = (kmeans.fit(data))

print("Data info.: ", k_means)

labels = (kmeans.predict(data))
print("Labels: ", labels)

centroids = (kmeans.cluster_centers_)
print("Centroids: ", centroids)

cost=[]
for i in range(1,11):
    KM=KMeans(n_clusters=i)
    KM.fit(data)
    cost.append(KM.inertia_)

print('\n')

l = pd.DataFrame(labels)
print(l)

ll = pd.DataFrame(centroids)
print(ll)
