#Import required libraries
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import tensorflow_docs.plots
import tensorflow_docs.modeling
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error


#Read The Dataset
Gangetic_West_Bengal_dataset = pd.read_csv('dataset/Gangetic West Bengal Rainfall Dataset.csv')

st.title('Rainfall Prediction!')
st.write('\n')

st.write("""
# Author : Sujit Mandal

Github: https://github.com/sujitmandal

My Package   : https://pypi.org/project/images-into-array/

LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/

Facebook : https://www.facebook.com/sujit.mandal.33671748

Twitter : https://twitter.com/mandalsujit37
""")

EPOCHS = 10000

#Drop Unnecessary attributes
temp_data = pd.DataFrame(Gangetic_West_Bengal_dataset)
temp_dataset = temp_data.drop(['JF', 'MAM', 'JJAS', 'OND'], axis = 1)

#Preprocess The Data
dataset = pd.get_dummies(temp_dataset, prefix='', prefix_sep='')

#Split the data into train and test
train_dataset = dataset.sample(frac=0.7, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

st.title('Train Data')
st.dataframe(train_dataset)
st.write('\n')
st.title('Test Data')
st.dataframe(test_dataset)


#Plot The Dataset
#Boxplot
st.write('\n')
plt.title('Boxplot')
temp_dataset.boxplot()
st.pyplot()

#histogram 
st.title('Histogram')
temp_dataset.hist()
st.pyplot()

#Kernel Density Estimation
st.title('Kernel Density Estimation')
temp_dataset.plot.kde()
plt.grid(True)
st.pyplot()

#Correlation
st.title('Correlation')
sns.heatmap(temp_dataset.corr(),annot=True)
st.pyplot()

#Actual Price
actual_rainfall  = test_dataset['ANNUAL']
year = test_dataset['YEAR']

#The Overall Statistics of Training Data
train_stats = train_dataset.describe()
train_stats.pop('ANNUAL')
train_stats = train_stats.transpose()
st.title('Overall Statistics of Training Data:')
st.dataframe(train_stats)

#The Overall Statistics of Test Data
test_stats = test_dataset.describe()
test_stats.pop('ANNUAL')
test_stats = test_stats.transpose()
st.title('Overall Statistics of Test Data:')
st.dataframe(test_stats)

#Labeling The datasets
train_labels = train_dataset.pop('ANNUAL')
test_labels = test_dataset.pop('ANNUAL')


try:
    if not os.path.exists('saved model/my_model.h5'):
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


        ch_path = ('save/cp.ckpt')
        cp_dir = os.path.dirname(ch_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(ch_path,
                                                        save_weights_only = True,
                                                        verbose = 1)
        model = build_model()

        #Train the model
        history = model.fit(
          train_dataset, train_labels,
          epochs=EPOCHS, validation_split = 0.2, verbose=0,
          callbacks=[tfdocs.modeling.EpochDots(), cp_callback])

        #Visualize the model's
        st.title('Model  One')
        model_hist = pd.DataFrame(history.history)
        model_hist['epoch'] = history.epoch
        st.write('\n')
        st.title('Model One History')
        st.dataframe(model_hist)


        #LOSE
        st.write('\n')
        st.title('Model  One')
        st.title('Model  Loss')
        plt.plot(model_hist.epoch, model_hist.loss)
        plt.plot(model_hist.epoch, model_hist.val_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Loss', 'Val_Loss'], loc='upper right')
        plt.grid(True)
        st.pyplot()

        #Mean Absolute Error
        st.write('\n')
        st.title('Mean Absolute Error')
        plt.plot(model_hist.epoch, model_hist.mae)
        plt.plot(model_hist.epoch, model_hist.val_mae)
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend(['Mae', 'Val_Mae'],loc='upper right')
        plt.grid(True)
        st.pyplot()

        #Mean Square Error
        st.write('\n')
        st.title('Mean Square Error')
        plt.plot(model_hist.epoch, model_hist.mse)
        plt.plot(model_hist.epoch, model_hist.val_mse)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend(['Mse', 'Val_Mse'], loc='upper right')
        plt.grid(True)
        st.pyplot()


        #Calculate Lose, Mae, Mse
        loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

        st.title('Model  One')
        st.title('Mean Absolute Error')
        st.text('MAE :')
        st.write(mae)
        st.title('Mean Square Error')
        st.text('MSE :')
        st.write(mse)

        model.load_weights(ch_path)
        loss, mae, mse = model.evaluate(test_dataset, test_labels)
        #print('restored model, accuracy: {:5.2f} ANN'.format(mae))
        ch_path_2 = ('save/cp-{epoch:04d}.ckpt')
        cp_dir_2 = os.path.dirname(ch_path_2)

        cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(ch_path_2, 
                                                          save_weights_only =  True, 
                                                          verbose = 1,
                                                          period = 50)
        model = build_model()

        #Train the model
        history2 = model.fit(train_dataset, train_labels, 
                              epochs=EPOCHS,  batch_size=32 ,verbose=0,
                              validation_data=(test_dataset, test_dataset), 
                              callbacks = [cp_callback_2])

        latest_model = tf.train.latest_checkpoint(cp_dir_2)
        #save
        model.save_weights('./save/my_save')

        #restore
        model = build_model()
        model.load_weights('./save/my_save')

        #Visualize the model's
        st.title('Model  Two')
        model_hist2 = pd.DataFrame(history2.history)
        model_hist2['epoch'] = history2.epoch
        st.write('\n')
        st.title('Model Two History')
        st.dataframe(model_hist2)

        #LOSE
        st.write('\n')
        st.title('Model  Two')
        st.title('Model  Loss')
        plt.plot(model_hist2.epoch, model_hist2.loss)
        plt.plot(model_hist2.epoch, model_hist2.val_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Loss', 'Val_Loss'], loc='upper right')
        plt.grid(True)
        st.pyplot()

        #Mean Absolute Error
        st.write('\n')
        st.title('Mean Absolute Error')
        plt.plot(model_hist2.epoch, model_hist2.mae)
        plt.plot(model_hist2.epoch, model_hist2.val_mae)
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend(['Mae', 'Val_Mae'], loc='upper right')
        plt.grid(True)
        st.pyplot()

        #Mean Square Error
        st.write('\n')
        st.title('Mean Square Error')
        plt.plot(model_hist2.epoch, model_hist2.mse)
        plt.plot(model_hist2.epoch, model_hist2.val_mse)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend(['Mse', 'Val_Mse'], loc='upper right')
        plt.grid(True)
        st.pyplot()


        loss2, mae2, mse2 = model.evaluate(test_dataset, test_labels)
        #print('restored model, accuracy: {:5.2f} ANN'.format(mae2))

        st.title('Model  Two')
        st.title('Mean Absolute Error')
        st.text('MAE :')
        st.write(mae2)
        st.title('Mean Square Error')
        st.text('MSE :')
        st.write(mse2)

        model = build_model()
        history3 = model.fit(train_dataset, train_labels, epochs=EPOCHS,
                            validation_split = 0.2, batch_size=32 ,verbose=0,
                            callbacks=[tfdocs.modeling.EpochDots()])

        #Visualize the model's
        st.title('Model  Three')
        model_hist3 = pd.DataFrame(history3.history)
        model_hist3['epoch'] = history3.epoch
        st.write('\n')
        st.title('Model Three History')
        st.dataframe(model_hist3)

        #LOSE
        st.write('\n')
        st.title('Model  Three')
        st.title('Model  Loss')
        plt.plot(model_hist3.epoch, model_hist3.loss)
        plt.plot(model_hist3.epoch, model_hist3.val_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Loss', 'Val_Loss'], loc='upper right')
        plt.grid(True)
        st.pyplot()

        #Mean Absolute Error
        st.write('\n')
        st.title('Mean Absolute Error')
        plt.plot(model_hist3.epoch, model_hist3.mae)
        plt.plot(model_hist3.epoch, model_hist3.val_mae)
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend(['Mae', 'Val_Mae'], loc='upper right')
        plt.grid(True)
        st.pyplot()

        #Mean Square Error
        st.write('\n')
        st.title('Mean Square Error')
        plt.plot(model_hist3.epoch, model_hist3.mse)
        plt.plot(model_hist3.epoch, model_hist3.val_mse)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend(['Mse', 'Val_Mse'], loc='upper right')
        plt.grid(True)
        st.pyplot()


        loss3, mae3, mse3 = model.evaluate(test_dataset, test_labels)
        #print('restored model, accuracy: {:5.2f} ANN'.format(mae3))

        st.title('Model  Three')
        st.title('Mean Absolute Error')
        st.text('MAE :')
        st.write(mae3)
        st.title('Mean Square Error')
        st.text('MSE :')
        st.write(mse3)

        #save entire model to a HDF5 file
        model.save('saved model/my_model.h5')
        st.text('Model is saved...')

        #Load model
        new_model = keras.models.load_model('saved model/my_model.h5')
        new_model.summary()


        f_loss, f_mae, f_mse = new_model.evaluate(test_dataset, test_labels)
        #print('restored model, accuracy: {:5.2f} ANN'.format(f_mae))

        st.title('Final Model')
        st.title('Model Loss')
        st.text('LOSS :')
        st.write(f_loss)
        st.title('Mean Absolute Error')
        st.text('MAE :')
        st.write(f_mae)
        st.title('Mean Square Error')
        st.text('MSE :')
        st.write(f_mse)

        #Prediction
        predictions = new_model.predict(test_dataset).flatten()
        st.title('Prediction')
        st.dataframe(predictions)

        r2_Score = r2_score(actual_rainfall, predictions)
        st.write('\n')
        st.title('R Squared :')
        st.text('Score :')
        st.write(r2_Score)


        #Root Mean Squared Error
        rmse = sqrt(mean_squared_error(actual_rainfall, predictions))
        st.write('\n')
        st.title('Root Mean Squared Error :')
        st.text('RMSE :')
        st.write(rmse)

        #Mean Squared Error
        mse = mean_squared_error(actual_rainfall, predictions)
        st.write('\n')
        st.title('Mean Squared Error :')
        st.text('MSE :')
        st.write(mse)

        #Mean Absolute Error
        mae = mean_absolute_error(actual_rainfall, predictions)
        st.write('\n')
        st.title('Mean Absolute Error :')
        st.text('MAE :')
        st.write(mae)

        #Mean Squared Log Error
        msle = mean_squared_log_error(actual_rainfall, predictions)
        st.write('\n')
        st.title('Mean Squared Log Error :')
        st.text('MSLE :')
        st.write(msle)

        #Max Error
        me = max_error(actual_rainfall, predictions)
        st.write('\n')
        st.title('Max Error :')
        st.text('ME :')
        st.write(me)


        #Plot True Values vs. Predictions
        st.write('\n')
        st.title('Plot True Values vs. Predictions')
        a = plt.axes(aspect='equal')
        plt.scatter(test_labels, predictions)
        plt.xlabel('Actual Rainfall')
        plt.ylabel('Predicted Rainfall')
        lims = [0,2500]
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims)
        plt.grid(True)
        st.pyplot()

    

        #Plot Actuall vs. Prediction
        st.write('\n')
        st.title('Actual Rainfall vs. Predicted Rainfall')
        plt.plot(year, actual_rainfall)
        plt.plot(year, predictions, 'r--')
        plt.xlabel('[__Actual Rainfall],  [----Predicted Rainfall]')
        plt.ylabel('Price')
        plt.grid(True)
        st.pyplot()


        #Plot Prediction Error vs.Count
        error = predictions - test_labels
        plt.hist(error, bins = 25)
        plt.xlabel("Prediction Error [ANNUAL]")
        _ = plt.ylabel("Count")
        plt.grid(True)
        st.pyplot()


        def Average(lst):
          avg = sum(lst) / len(lst)
          return(avg)

        loss = model_hist3['loss']
        mae = model_hist3['mae']
        mse = model_hist3['mse']

        avg_loss = Average(loss)
        avg_mae = Average(mae)
        avg_mse = Average(mse)

        rmse = math.sqrt(avg_mse)
        rmae = math.sqrt(avg_mae)

        st.write('\n')
        st.title('Average Loss :')
        st.text('Loss :')
        st.write(avg_loss)

        st.write('\n')
        st.title('Average Mean Absolute Error :')
        st.text('AMAE :')
        st.write(avg_mae)

        st.write('\n')
        st.title('Average Mean Absolute Error :')
        st.text('AMAE :')
        st.write(avg_mse)

        st.write('\n')
        st.title('Average Root Mean Absolute Error :')
        st.text('ARMAE :')
        st.write(rmse)

        st.write('\n')
        st.title('Average Root Mean Absolute Error :')
        st.text('ARMAE :')
        st.write(rmae)

    else:
        model = tf.keras.models.load_model('saved model/my_model.h5')
        #print(model.summary())

        #Prediction
        predictions = model.predict(test_dataset).flatten()
        st.title('Prediction')
        st.dataframe(predictions)

          #Plot True Values vs. Predictions
        st.write('\n')
        st.title('Plot True Values vs. Predictions')
        a = plt.axes(aspect='equal')
        plt.scatter(test_labels, predictions)
        plt.xlabel('Actual Rainfall')
        plt.ylabel('Predicted Rainfall')
        lims = [0,2500]
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims)
        plt.grid(True)
        st.pyplot()

    

        #Plot Actuall vs. Prediction
        st.write('\n')
        st.title('Actual Rainfall vs. Predicted Rainfall')
        plt.plot(year, actual_rainfall)
        plt.plot(year, predictions, 'r--')
        plt.xlabel('[__Actual Rainfall],  [----Predicted Rainfall]')
        plt.ylabel('Price')
        plt.grid(True)
        st.pyplot()


        #Plot Prediction Error vs.Count
        st.title('Prediction Error vs.Count')
        error = predictions - test_labels
        plt.hist(error, bins = 25)
        plt.xlabel("Prediction Error [ANNUAL]")
        _ = plt.ylabel("Count")
        plt.grid(True)
        st.pyplot()

except OSError:
    print('Error: Creating directory!')
