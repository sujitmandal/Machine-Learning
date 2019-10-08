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


#print(data)

plt.scatter(data['x'], data['y'])
#plt.show()

centroids={i+1:[np.random.randint(0,80),np.random.randint(0,80)] 
          for i in range(3)}

print(centroids)

plt.scatter(data['x'],data['y'])
colmap={1:'r',2:'g',3:'y'}
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.show()

#assignment
def assignment_1(data,centroids):
    #sqrt((x1-x2)^2+(y1-y2)^2)
    for i in centroids.keys():
        data['distance_from_{}'.format(i)]=np.sqrt((data['x']-centroids[i][0])**2)+((data['y']-centroids[i][1])**2)
    return(data)
data = assignment_1(data,centroids)
print(data.head())

def assignment(data,centroids):
    #sqrt((x1-x2)^2+(y1-y2)^2)
    for i in centroids.keys():
         data['distance_from_{}'.format(i)]=np.sqrt((data['x']-centroids[i][0])**2)+((data['y']-centroids[i][1])**2)
    data['closest']=data.loc[:,'distance_from_1':].idxmin(axis=1)
    data['closest']=data['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    data['color']=data['closest'].map(lambda x:colmap[x])
    return(data)
data = assignment(data,centroids)
print(data.head())


plt.scatter(data['x'], data['y'], color = data['color'])
plt.show()

#update stage= updating your centroids value
import copy
old_centroids=copy.deepcopy(centroids)
def update(k):
    for i in centroids.keys():
        centroids[i][0]=np.mean(data[data['closest']==i]['x'])
        centroids[i][1]=np.mean(data[data['closest']==i]['y'])
    return(k)
centroids=update(centroids)

print("Old Centroids: ", old_centroids)
print("Centroids: ", centroids)