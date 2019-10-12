#[2,3,4,10,11,12,20,25,30] find the two cluster
 
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

data_1 = np.array([2,3,4,10,11,12,20,25,30])

#print(data)
data = data_1.reshape(-1,1)
number = int(input("Enter The Number of cluster: "))

kmeans=KMeans(n_clusters = number , max_iter=500)
k_means = (kmeans.fit(data))

#print("Data info.: ", k_means)

labels = (kmeans.predict(data))
centroids = (kmeans.cluster_centers_)
number_of_cluster = (kmeans.n_clusters)

print("\n")
print('OUTPUT:')
print("Labels: ", labels)
print("Centroids: ", centroids)
print("Number of cluster's is", number_of_cluster)

#OUTPUT:
'''
    Enter The Number of cluster: 2


    OUTPUT:
    Labels:  [0 0 0 0 0 0 1 1 1]
    Centroids:  [[ 7.]
    [25.]]
    Number of cluster's is 2

'''
