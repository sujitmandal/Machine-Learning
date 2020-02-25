# Import required libraries
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

#Github: https://github.com/sujitmandal

#This programe is create by Sujit Mandal

"""
Github: https://github.com/sujitmandal
This programe is create by Sujit Mandal

LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/

Facebook : https://www.facebook.com/sujit.mandal.33671748

Twitter : https://twitter.com/mandalsujit37
"""

data = pd.read_csv('wine vintge.csv')
print(data.head(5))
print('\n')
print(data.tail(5))
print('\n')
print(data.shape)

class_names = ['Vinyard #1' , 'Vinyard #2', 'Vinyard #3']

transpose = data.describe().transpose()
print(transpose)

target_column = ['variety']
predictors = list(set(list(data.columns)) - set(target_column)) 
data[predictors] = data[predictors] / data[predictors].max()
predictors_transpose = data.describe().transpose()
print(predictors_transpose)

#Creating the Training and Test Datasets

x = data[predictors].values
y = data[target_column].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=40)

print(x_train.shape)
print(x_test.shape)


neighbors = int(input('Enter The Number of Neighbors : '))
knn = KNeighborsClassifier(n_neighbors= neighbors)
knn.fit(x_train, y_train)

predict_train = knn.predict(x_train)
predict_test = knn.predict(x_test)

predicted = knn.predict(x_test)
accuracy = accuracy_score(y_test, predicted)

accuracyInpercentage = (accuracy * 100)
print('\n')
print('Accuracy :', accuracyInpercentage)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(knn, x_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print('\n')
    print(title)
    print(disp.confusion_matrix)

plt.show()

print('\n')
print('Classification Report:')
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))