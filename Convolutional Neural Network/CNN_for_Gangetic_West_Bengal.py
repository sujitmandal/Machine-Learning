#Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal

LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/

Facebook : https://www.facebook.com/sujit.mandal.33671748

Twitter : https://twitter.com/mandalsujit37
"""

#Read The Data
data = pd.read_csv('Gangetic West Bengal.csv')
print(data.head(2))
print('\n')
print(data.tail(2))

#Data info
print('\n')
print(data.info())

#Inspect the data
sns.pairplot(data[['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANN']], diag_kind="kde")
plt.show()

#Clean The Data
print('\n')
print(data.isna().sum())

#Preprocess The Data
dataset = pd.get_dummies(data, prefix='', prefix_sep='')

#Split the data into train and test
train_dataset = dataset.sample(frac=0.7, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print('\n')
print('Train dataset shape :', train_dataset.shape)
print('\n')
print('Test dataset shape :',test_dataset.shape)

#Inspect the Train Data
sns.pairplot(train_dataset[['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANN']], diag_kind="kde")
plt.show()

#Inspect the Test Data
sns.pairplot(test_dataset[['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANN']], diag_kind="kde")
plt.show()

#Actual Annual Rainfall
actual = test_dataset['ANN']
#Year
year = test_dataset['YEAR']

#The Overall Statistics of Total Data
statistics = data.describe()
statistics = statistics.transpose()
print('\n')
print('Overall Statistics of Total Data:')
print(statistics)

#The Overall Statistics of Training Data
train_stats = data.describe()
train_stats.pop('ANN')
train_stats = train_stats.transpose()
print('\n')
print('Overall Statistics of Training Data:')
print(train_stats)


#The Overall Statistics of Test Data
test_stats = test_dataset.describe()
test_stats.pop('ANN')
test_stats = test_stats.transpose()
print('\n')
print('Overall Statistics of Test Data:')
print(test_stats)

#Labeling The datasets
train_labels = train_dataset.pop('ANN')
test_labels = test_dataset.pop('ANN')

#Convolutional Neural Network(CNN) building 
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
 
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
print('\n')
print('Model Sructure:')
model.summary()


#Test the prediction
example = train_dataset[:10]
result = model.predict(example)
print('\n')
print('Prediction Test:')
print(result)

#Train the model
EPOCHS = 1000

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])

#Visualize the model's
model_hist = pd.DataFrame(history.history)
model_hist['epoch'] = history.epoch
print('\n')
print('Model History:')
print(model_hist.head(3))
print('\n')
print(model_hist.tail(3))

#Plot The Model 
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

#LOSE
plotter.plot({'Basic': history}, metric = 'loss')
plt.show()

#Mean Absolute Error
plotter.plot({'Basic': history}, metric = 'mae')
plt.show()

#Mean Square Error
plotter.plot({'Basic': history}, metric = "mse")
plt.show()

#Calculate Lose, Mae, Mse
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
print('\n')
print("Testing set Mean Abs Error: {:5.2f} ANN".format(mae))

#Prediction
test_predictions = model.predict(test_dataset).flatten()

#Plot True Values vs. Predictions
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('Actual Rainfall')
plt.ylabel('Predicted Rainfall')
lims = [0,2500]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()

#Plot Actuall vs. Prediction
plt.title('Actual Rainfall vs. Predicted Rainfall')
plt.plot(year, actual)
plt.plot(year, test_predictions, 'r--')
plt.xlabel('[__Actual Rainfall],  [----Predicted Rainfall]')
plt.ylabel('Rainfall')
plt.show()

#Plot Prediction Error vs.Count
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [ANN]")
_ = plt.ylabel("Count")
plt.show()


#OUTPUT:

'''   YEAR   JAN   FEB   MAR    APR    MAY    JUN    JUL    AUG    SEP   OCT   NOV  DEC     ANN
0  1901  37.1  58.4   3.9   64.1  121.7  198.0  280.8  275.7  313.5  51.1  83.4  0.0  1487.6
1  1902   0.0   1.2  44.2  103.8  161.6  140.9  347.8  264.8  230.5  32.5  10.4  9.9  1347.7


     YEAR  JAN   FEB   MAR   APR    MAY    JUN    JUL    AUG    SEP    OCT   NOV   DEC     ANN
115  2016  9.9  37.9  14.7   5.8  111.9  172.7  334.7  416.7  233.7   69.8  19.2   0.0  1427.0
116  2017  1.4   0.0  35.2  25.8  106.3  193.7  489.3  264.5  191.4  225.2  21.9  14.0  1568.6


<class 'pandas.core.frame.DataFrame'>
RangeIndex: 117 entries, 0 to 116
Data columns (total 14 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   YEAR    117 non-null    int64  
 1   JAN     117 non-null    float64
 2   FEB     117 non-null    float64
 3   MAR     117 non-null    float64
 4   APR     117 non-null    float64
 5   MAY     117 non-null    float64
 6   JUN     117 non-null    float64
 7   JUL     117 non-null    float64
 8   AUG     117 non-null    float64
 9   SEP     117 non-null    float64
 10  OCT     117 non-null    float64
 11  NOV     117 non-null    float64
 12  DEC     117 non-null    float64
 13  ANN     117 non-null    float64
dtypes: float64(13), int64(1)
memory usage: 12.9 KB
None


YEAR    0
JAN     0
FEB     0
MAR     0
APR     0
MAY     0
JUN     0
JUL     0
AUG     0
SEP     0
OCT     0
NOV     0
DEC     0
ANN     0
dtype: int64


Train dataset shape : (82, 14)


Test dataset shape : (35, 14)


Overall Statistics of Total Data:
      count         mean         std     min     25%     50%     75%     max
YEAR  117.0  1959.000000   33.919021  1901.0  1930.0  1959.0  1988.0  2017.0
JAN   117.0    12.476923   14.652759     0.0     1.3     6.8    18.0    60.0
FEB   117.0    22.392308   24.089875     0.0     5.1    13.6    31.0   123.6
MAR   117.0    29.019658   30.522590     0.1     7.3    18.9    41.7   152.5
APR   117.0    44.388034   31.791132     0.9    20.2    38.7    60.5   174.2
MAY   117.0   107.810256   50.561489    16.4    72.3    99.6   130.3   250.9
JUN   117.0   246.102564   99.173371    69.7   180.7   226.3   303.6   597.1
JUL   117.0   327.841026   92.699865   148.9   269.3   313.4   398.6   633.1
AUG   117.0   311.882051   81.208123   167.1   252.8   292.6   366.8   573.4
SEP   117.0   245.143590   89.336356   101.1   186.8   229.5   283.8   591.8
OCT   117.0   116.288889   76.970762     4.1    59.7    95.9   155.9   358.3
NOV   117.0    21.561538   30.885067     0.0     1.3     8.9    31.2   160.5
DEC   117.0     5.712821   12.125582     0.0     0.0     0.3     5.2    72.8
ANN   117.0  1490.612821  227.694993  1015.1  1315.0  1482.2  1648.5  2099.8


Overall Statistics of Training Data:
      count         mean        std     min     25%     50%     75%     max
YEAR  117.0  1959.000000  33.919021  1901.0  1930.0  1959.0  1988.0  2017.0
JAN   117.0    12.476923  14.652759     0.0     1.3     6.8    18.0    60.0
FEB   117.0    22.392308  24.089875     0.0     5.1    13.6    31.0   123.6
MAR   117.0    29.019658  30.522590     0.1     7.3    18.9    41.7   152.5
APR   117.0    44.388034  31.791132     0.9    20.2    38.7    60.5   174.2
MAY   117.0   107.810256  50.561489    16.4    72.3    99.6   130.3   250.9
JUN   117.0   246.102564  99.173371    69.7   180.7   226.3   303.6   597.1
JUL   117.0   327.841026  92.699865   148.9   269.3   313.4   398.6   633.1
AUG   117.0   311.882051  81.208123   167.1   252.8   292.6   366.8   573.4
SEP   117.0   245.143590  89.336356   101.1   186.8   229.5   283.8   591.8
OCT   117.0   116.288889  76.970762     4.1    59.7    95.9   155.9   358.3
NOV   117.0    21.561538  30.885067     0.0     1.3     8.9    31.2   160.5
DEC   117.0     5.712821  12.125582     0.0     0.0     0.3     5.2    72.8


Overall Statistics of Test Data:
      count         mean         std     min      25%     50%      75%     max
YEAR   35.0  1969.085714   31.360683  1910.0  1946.00  1971.0  1988.50  2016.0
JAN    35.0    12.208571   12.166888     0.0     2.05     9.1    18.55    40.7
FEB    35.0    25.188571   26.910306     1.6     8.25    14.2    34.85   112.8
MAR    35.0    34.125714   30.409387     0.6     8.15    25.9    49.45   111.3
APR    35.0    41.345714   34.735084     1.9    17.55    32.8    55.00   174.2
MAY    35.0   104.500000   48.917608    35.5    72.60   101.7   127.35   250.9
JUN    35.0   280.742857  117.566465   103.7   200.25   241.8   319.80   597.1
JUL    35.0   329.074286  110.437971   168.7   267.00   299.9   361.10   633.1
AUG    35.0   332.131429   80.668475   213.2   283.70   323.6   383.20   573.4
SEP    35.0   253.022857  107.038050   101.1   191.80   229.5   258.55   591.8
OCT    35.0   119.897143   79.135203    11.6    67.00   100.4   161.05   351.7
NOV    35.0    16.831429   27.863108     0.0     0.30     3.8    20.05   115.2
DEC    35.0     6.837143   14.162255     0.0     0.00     0.6     6.35    72.8


Model Sructure:
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                896       
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_2 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 65        
=================================================================
Total params: 9,281
Trainable params: 9,281
Non-trainable params: 0
_________________________________________________________________


Prediction Test:
[[-16.932373]
 [ -6.029644]
 [-20.028202]
 [-13.167252]
 [-12.775799]
 [  8.718449]
 [-16.646774]
 [-21.572594]
 [-10.348297]
 [ 16.329626]]

Epoch: 0, loss:1893711.7981,  mae:1354.8766,  mse:1893711.7500,  val_loss:1024091.8750,  val_mae:990.3823,  val_mse:1024091.8750,  
....................................................................................................
Epoch: 100, loss:8826.6901,  mae:76.3639,  mse:8826.6895,  val_loss:41362.5000,  val_mae:178.2686,  val_mse:41362.5000,  
....................................................................................................
Epoch: 200, loss:10044.5387,  mae:96.4362,  mse:10044.5381,  val_loss:8594.5449,  val_mae:78.8210,  val_mse:8594.5449,  
....................................................................................................
Epoch: 300, loss:3043.2360,  mae:52.4801,  mse:3043.2361,  val_loss:7538.6880,  val_mae:82.3735,  val_mse:7538.6880,  
....................................................................................................
Epoch: 400, loss:2054.9831,  mae:43.4266,  mse:2054.9832,  val_loss:791.0483,  val_mae:23.3213,  val_mse:791.0483,  
....................................................................................................
Epoch: 500, loss:358.1508,  mae:16.0416,  mse:358.1508,  val_loss:1884.4530,  val_mae:36.9236,  val_mse:1884.4530,  
....................................................................................................
Epoch: 600, loss:285.9937,  mae:14.3691,  mse:285.9937,  val_loss:546.0570,  val_mae:19.0476,  val_mse:546.0570,  
....................................................................................................
Epoch: 700, loss:648.7473,  mae:22.2300,  mse:648.7474,  val_loss:6603.3218,  val_mae:79.8742,  val_mse:6603.3218,  
....................................................................................................
Epoch: 800, loss:1051.6413,  mae:30.3672,  mse:1051.6414,  val_loss:1271.4985,  val_mae:29.8480,  val_mse:1271.4985,  
....................................................................................................
Epoch: 900, loss:1226.0002,  mae:31.8293,  mse:1226.0002,  val_loss:9795.1416,  val_mae:98.2708,  val_mse:9795.1416,  
....................................................................................................

Model History:
           loss          mae           mse      val_loss     val_mae       val_mse  epoch
0  1.893712e+06  1354.876587  1.893712e+06  1.024092e+06  990.382324  1.024092e+06      0
1  8.797623e+05   912.457825  8.797623e+05  5.044409e+05  681.406433  5.044409e+05      1
2  4.189172e+05   616.770264  4.189172e+05  2.807153e+05  492.685303  2.807153e+05      2


            loss        mae          mse     val_loss    val_mae      val_mse  epoch
997   241.958825  13.278739   241.958817  1133.564941  29.017952  1133.564941    997
998  1030.181613  30.591215  1030.181763   323.880066  14.638643   323.880066    998
999   271.697635  14.249655   271.697632  2150.535889  41.700928  2150.535889    999

35/35 - 0s - loss: 2763.5642 - mae: 48.1401 - mse: 2763.5642
Testing set Mean Abs Error: 48.14 ANN'''