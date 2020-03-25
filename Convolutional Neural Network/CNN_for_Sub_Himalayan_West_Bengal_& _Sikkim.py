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
data = pd.read_csv('Sub Himalayan West Bengal & Sikkim.csv')
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
lims = [0,4000]
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
0  1901  26.5  14.8  14.1   29.2  195.5  488.4  524.8  501.1  242.7  55.5  17.9  2.6  2113.2
1  1902   1.2   0.7  87.1  126.1  271.3  539.2  671.0  603.8  799.9  74.4   5.6  0.0  3180.4


     YEAR   JAN  FEB   MAR    APR    MAY    JUN    JUL    AUG    SEP    OCT  NOV  DEC     ANN
115  2016  20.7  7.1  98.9  101.4  233.7  570.8  732.1  209.9  494.3  155.6  0.0  0.3  2624.8
116  2017   3.0  5.1  84.6  146.8  226.8  400.6  478.6  777.9  390.1  166.5  3.8  1.0  2684.9


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
JAN   117.0    14.045299   16.960682     0.0     2.3     9.4    19.6   103.0
FEB   117.0    22.686325   19.538556     0.1     8.2    19.6    32.9   109.9
MAR   117.0    43.966667   31.246214     0.0    15.1    42.6    64.8   132.1
APR   117.0   110.911111   55.314528     4.8    71.5   110.9   146.8   281.8
MAY   117.0   268.478632   69.373152   142.0   217.1   268.8   308.9   503.1
JUN   117.0   536.989744  133.600887   261.7   446.5   527.8   609.2   896.0
JUL   117.0   645.700855  164.364898   340.9   518.1   657.7   749.5  1064.6
AUG   117.0   520.304274  166.492961   209.9   408.4   500.9   610.7  1036.6
SEP   117.0   421.698291  112.828724   174.6   341.6   408.4   486.8   799.9
OCT   117.0   143.943590   88.586364    11.7    76.5   128.7   183.8   513.5
NOV   117.0    15.846154   22.702767     0.0     3.2     8.5    17.9   132.1
DEC   117.0     5.967521    9.509350     0.0     0.5     2.4     6.8    56.5
ANN   117.0  2750.552991  335.192332  1988.2  2530.0  2718.0  2935.1  3655.1


Overall Statistics of Training Data:
      count         mean         std     min     25%     50%     75%     max
YEAR  117.0  1959.000000   33.919021  1901.0  1930.0  1959.0  1988.0  2017.0
JAN   117.0    14.045299   16.960682     0.0     2.3     9.4    19.6   103.0
FEB   117.0    22.686325   19.538556     0.1     8.2    19.6    32.9   109.9
MAR   117.0    43.966667   31.246214     0.0    15.1    42.6    64.8   132.1
APR   117.0   110.911111   55.314528     4.8    71.5   110.9   146.8   281.8
MAY   117.0   268.478632   69.373152   142.0   217.1   268.8   308.9   503.1
JUN   117.0   536.989744  133.600887   261.7   446.5   527.8   609.2   896.0
JUL   117.0   645.700855  164.364898   340.9   518.1   657.7   749.5  1064.6
AUG   117.0   520.304274  166.492961   209.9   408.4   500.9   610.7  1036.6
SEP   117.0   421.698291  112.828724   174.6   341.6   408.4   486.8   799.9
OCT   117.0   143.943590   88.586364    11.7    76.5   128.7   183.8   513.5
NOV   117.0    15.846154   22.702767     0.0     3.2     8.5    17.9   132.1
DEC   117.0     5.967521    9.509350     0.0     0.5     2.4     6.8    56.5


Overall Statistics of Test Data:
      count         mean         std     min      25%     50%      75%     max
YEAR   35.0  1969.085714   31.360683  1910.0  1946.00  1971.0  1988.50  2016.0
JAN    35.0    16.591429   19.146515     0.0     1.20    14.7    23.65    99.8
FEB    35.0    22.537143   19.169708     0.8     9.05    17.9    25.60    81.1
MAR    35.0    50.634286   31.646726     2.2    34.80    46.6    68.85   132.1
APR    35.0   107.497143   51.641083    24.7    60.90   101.4   152.25   239.8
MAY    35.0   271.720000   76.122988   162.7   217.90   254.0   307.00   503.1
JUN    35.0   547.502857  145.510744   287.5   468.85   543.2   596.70   889.2
JUL    35.0   678.391429  136.773831   384.6   597.80   699.4   753.95   970.2
AUG    35.0   496.517143  164.493142   209.9   398.60   477.7   579.35   990.5
SEP    35.0   422.628571   82.630220   229.9   355.60   406.9   477.05   593.6
OCT    35.0   145.345714   95.764363    11.7    69.45   125.1   186.90   351.6
NOV    35.0    15.085714   19.384653     0.0     2.95     8.8    19.20    87.3
DEC    35.0     6.277143    9.831854     0.0     0.50     3.1     8.20    53.2
2020-03-25 21:03:47.984522: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-25 21:03:48.010936: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 1800000000 Hz
2020-03-25 21:03:48.012278: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55840d79f580 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-25 21:03:48.012352: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version


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
[[72.90597 ]
 [58.326073]
 [31.144096]
 [24.391273]
 [61.26278 ]
 [41.102005]
 [56.650055]
 [46.05574 ]
 [68.254974]
 [51.370422]]

Epoch: 0, loss:6859214.4462,  mae:2594.3293,  mse:6859214.5000,  val_loss:4630607.0000,  val_mae:2140.0647,  val_mse:4630607.0000,  
....................................................................................................
Epoch: 100, loss:9790.4904,  mae:86.0930,  mse:9790.4912,  val_loss:27698.2852,  val_mae:153.5016,  val_mse:27698.2852,  
....................................................................................................
Epoch: 200, loss:1719.2351,  mae:34.6824,  mse:1719.2352,  val_loss:7004.7197,  val_mae:76.0146,  val_mse:7004.7197,  
....................................................................................................
Epoch: 300, loss:496.4885,  mae:18.3329,  mse:496.4885,  val_loss:4726.5391,  val_mae:62.6010,  val_mse:4726.5391,  
....................................................................................................
Epoch: 400, loss:17338.1871,  mae:125.9001,  mse:17338.1875,  val_loss:1257.0889,  val_mae:31.4103,  val_mse:1257.0889,  
....................................................................................................
Epoch: 500, loss:7796.8458,  mae:84.7986,  mse:7796.8457,  val_loss:2139.6873,  val_mae:43.5338,  val_mse:2139.6873,  
....................................................................................................
Epoch: 600, loss:500.0753,  mae:19.2864,  mse:500.0753,  val_loss:2092.9746,  val_mae:43.1523,  val_mse:2092.9746,  
....................................................................................................
Epoch: 700, loss:336.0206,  mae:15.3443,  mse:336.0206,  val_loss:940.3281,  val_mae:26.1952,  val_mse:940.3281,  
....................................................................................................
Epoch: 800, loss:219.2415,  mae:11.8783,  mse:219.2415,  val_loss:506.4083,  val_mae:20.3351,  val_mse:506.4083,  
....................................................................................................
Epoch: 900, loss:22936.7063,  mae:148.5973,  mse:22936.7051,  val_loss:712.0970,  val_mae:24.0734,  val_mse:712.0970,  
....................................................................................................

Model History:
           loss          mae         mse   val_loss      val_mae    val_mse  epoch
0  6.859214e+06  2594.329346  6859214.50  4630607.0  2140.064697  4630607.0      0
1  4.469705e+06  2084.944092  4469705.00  3182168.5  1771.112305  3182168.5      1
2  3.012690e+06  1708.499023  3012690.25  1941758.5  1380.128784  1941758.5      2


            loss        mae          mse     val_loss    val_mae      val_mse  epoch
997  3507.480236  57.411232  3507.480225  5384.900879  71.687920  5384.900879    997
998  7528.155589  85.596367  7528.155762  5130.127441  70.205261  5130.127441    998
999  3828.951936  60.771648  3828.951904   398.189728  14.955782   398.189728    999

35/35 - 0s - loss: 386.8268 - mae: 18.3477 - mse: 386.8268
Testing set Mean Abs Error: 18.35 ANN'''