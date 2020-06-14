# Machine-Learning
Linear Regression , k-mean clustering , Watershed , Gradients and Edge Detection , Threshold , Corelation , Neural Network
---------------------------------------------------------------------------------------------------------------------------

 Introduction:
---------------
Computer Vision is one of the hottest topics in artificial intelligence. It is making tremendous advances in self-driving cars, robotics as well as in various photo correction apps. Steady progress in object detection is being made every day. GANs is also a thing researchers are putting their eyes on these days. Vision is showing us the future of technology and we can’t even imagine what will be the end of its possibilities. So do you want to take your first step in Computer Vision and participate in this latest movement? Welcome you are at the right place. From this article, we’re going to have a series of tutorials on the basics of image processing and object detection.

 What is machine learning?
----------------------------
• Machine learning teaches computers to do what comes naturally to humans and animals learn from experience. Machine learning algorithms use computational methods to “learn” information directly from data without relying on a predetermined equation as a model.

 How machine learning works?
-----------------------------
• Machine learning are two types of techniques:

• Supervised.

• Unsupervised.

 Machine learning techniques:
------------------------------

 What is supervised learning:
------------------------------
• Supervised learning which trains a model on known input and output data so that it can predict future outputs. Supervised uses classification and regression techniques to develop predictive models.

 Classification:
-----------------
• Classification techniques predict discrete responses for example, whether an e-mail is genuine or spam. Classification models classify input data into categories. Typical application include medical imaging, speech recognition and credit scoring.

 Regression:
-------------
• Regression techniques predict continuous responses for example, changes in temperature or fluctuation in power demand. Typical application include electricity load forecasting and algorithmic trading.

 Output:
-------
 Linear Regression:
--------------------
![](https://github.com/sujitmandal/Machine-Learning/blob/master/Output/Regression/Linear%20Regression/Scatter%20Plot.png)

![](https://github.com/sujitmandal/Machine-Learning/blob/master/Output/Regression/Linear%20Regression/Outputs%20plot.png)

![](https://github.com/sujitmandal/Machine-Learning/blob/master/Output/Regression/Linear%20Regression/Actual%20vs.%20Predicted.png)

 Unsupervised learning:
------------------------
• Unsupervised learning which finds hidden patterns in data. It is used to draw inferences from datasets consisting of input data without labelled responses. Unsupervised uses clustering and association techniques to develop predictive models.

 Clustering:
-------------
• Clustering is the most common unsupervised learning technique. It is used for exploratory data analysis to find hidden patterns. It is used to grope the data for similar type data. For example gene sequence analysis, market research, and object recognition.

 Output:
---------
 k-mean clustering:
--------------------
![](https://github.com/sujitmandal/Machine-Learning/blob/master/Output/k-mean%20clustering/Figure_1.jpg.png)

![](https://github.com/sujitmandal/Machine-Learning/blob/master/Output/k-mean%20clustering/Figure_2.jpg.png)

K-means cluster:
---------------

	K = { 2, 3, 4, 10, 11, 12, 20, 25, 30 } 

	Let  k = 2 , where we are going to find two clusters.

	Let mean M1, M2.
	Therefore M1 = 4, M2 = 12  [ note: (K - M1) = X1, (K - M2) = X2 ,min value will be taken ]

Iteration 1

	k1 = { 2, 3, 4 }
	K2 = { 10, 11, 12, 20, 25, 30 }

	M1 = (2+3+4 / 3 )      M2 = (10+11+12+20+25+30 / 6)
	M1 = (9 / 3)  	       M2 = (108 / 6)
	M1 = 3                 M2 = 18

Iteration 2 :

	K1 = { 2, 3, 4, 10 }
	K2 = { 11, 12, 20, 25, 30 }
	
	M1 = (2+3+4+10 / 4)    M2 = (11+12+20+25+30 / 5)
	M1 = (19 / 4)          M2 = (98 / 5)
	M1 = 4.75	       M2 = 19.5
	M1 = 5		       M2 = 20


Iteration 3 :

	K1 = { 2, 3, 4, 10, 12 }
	K2 = { 20, 25, 30 }
	
	M1 = (2+3+4+10+11+12 / 6)     M2 = (20+25+30 / 3)
	M1 = (42 / 6)		      M2 = (75 / 3)
	M1 = 7			      M2 = 25
	
Iteration 4 : 

	K1 = { 2, 3, 4, 10, 12 }
	K2 = { 20, 25, 30 }
	
	M1 = (2+3+4+10+11+12 / 6)      M2 = (20+25+30 / 3)
	M1 = (42 / 6)		       M2 = (75 / 3)
	M1 = 7			       M2 = 25

	Thus we are getting the same meaning hence we need to stop.

	Therefore the new cluster is…..

		K1 = { 2, 3, 4, 10, 12 }
		K2 = { 20, 25, 30 }

		And the centroids is [ 7, 25 ]
		
 Neural networks :
-------------------

• Neural networks are composed of simple elements operating in parallel. These elements are inspired by biological nervous systems. As in nature, the network function is determined largely by the connections between elements. We can train a neural network to perform a particular function by adjusting the values of the connections (weights) between elements.

 Multi Layer Neural Networks Architecture:
------------------------------------------
 ![](https://github.com/sujitmandal/Machine-Learning/blob/master/data/Data/nnet.jpg)
 
  Multi Layer Feed Forward Neural Networks Architecture :
 ---------------------------------------------------------
 ![](https://github.com/sujitmandal/Machine-Learning/blob/master/data/Data/OH3gI.png)
 
 Output:
---------
1.Multi-layer Perceptron (MLP):
-------------------------------
![](https://github.com/sujitmandal/Machine-Learning/blob/master/Output/Neural%20Network/Multi-layer%20Perceptron%20(MLP)/irisWithOutNormalization.png)

2.KNeighborsClassifier(KNN):
----------------------------
![](https://github.com/sujitmandal/Machine-Learning/blob/master/Output/Neural%20Network/KNeighborsClassifier/irisWithoutNormalization.png)

 Convolutional Neural Network:
-------------------------------
• Convolutional Neural Network has Artificial Neural Network design architecture, which has proven its effectiveness in areas such as image recognition and classification. The Basic Principle behind the working of CNN is the idea of Convolution, producing filtered Feature Maps stacked over each other. A convolutional neural network consists of several layers. Implicit explanation about each of these layers is given below.

 One:
------
![](https://github.com/sujitmandal/Machine-Learning/blob/master/data/Data/cnn1.png)

 Two:
------
![](https://github.com/sujitmandal/Machine-Learning/blob/master/data/Data/cnn2.png)

 Three:
--------
![](https://github.com/sujitmandal/Machine-Learning/blob/master/data/Data/cnn3.jpeg)


 Association:
---------------
• Association is used to find the patterns.

 What is true positive(TP):
------------------------------
• Number of example predicted positive that are actually positive is called true positive (TP) in machine learning.

 What is false positive(FP):
-----------------------------
• Number of example predicted positive that are actually negative is called false positive (FP) in machine learning.

 What is true negative (TN):
-----------------------------
• Number of example predicted negative that are actually negative is called true negative (TN) in machine learning.

 What is false negative (FN):
------------------------------
• Number of example predicted negative that are actually positive is called false negative (FN) in machine learning.

 Precision:
------------
• In pattern recognition, information retrieval and binary classification, precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances. 

 What is recall:
-----------------
• While recall (also known as sensitive) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances.

  What is F-Measure:
---------------------
• A measure that combines precision and recall is the harmonic mean of precision and recall that traditionally called F-Measure or balance F-score. 

 Face Mask Detection using Tensorflow, Keras, Opencv, Python :
-------------------------------------------------------------

 [![Face Mask Detection Using | Convolutional Neural Networks | Tensorflow | Python |](https://yt-embed.herokuapp.com/embed?v=wfK5N9Qq1uk)](https://www.youtube.com/watch?v=wfK5N9Qq1uk "Face Mask Detection Using | Convolutional Neural Networks | Tensorflow | Python |")

 Plot Output:
--------------

 train mask image:
-------------------
![](https://github.com/sujitmandal/Face-Mask-Detection/blob/master/plot/train_mask_image.png)

 train without mask image:
---------------------------
![](https://github.com/sujitmandal/Face-Mask-Detection/blob/master/plot/train_without_mask_image.png)

 test mask image:
-------------------
![](https://github.com/sujitmandal/Face-Mask-Detection/blob/master/plot/test_mask_image.png)

 test without mask image:
---------------------------
![](https://github.com/sujitmandal/Face-Mask-Detection/blob/master/plot/test_without_mask_image.png)

 accuracy:
-------------------
![](https://github.com/sujitmandal/Face-Mask-Detection/blob/master/plot/Figure_1.png)

 accuracy:
---------------------------
![](https://github.com/sujitmandal/Face-Mask-Detection/blob/master/plot/Figure_1.png)

 Method:
---------

1. Linear Regression
2. k-mean clustering 
3. Watershed 
4. Gradients and Edge Detection 
5. Threshold
6. Cross Validation
7. Correlation
8. Neural Network
9. Convolutional Neural Network
10. Face Mask Detection

 Requirement’s:
----------------

• Python 3.7

• Anaconda

• Visual Studio Code

 LINK’S:
---------

• Python : 
----------
Download https://www.python.org/downloads/

• Anaconda : 
------------
Windows:
-------
• Download https://www.anaconda.com/downloads

Linux:
------
Command:
-------
• " wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh "

• " bash Anaconda3-5.3.1-Linux-x86_64.sh "

• " conda update anaconda "

• Visual Studio Code :
----------------------
Download https://code.visualstudio.com/Download

• How to install | Python | | Anaconda | | Opencv library |
------------------------------------------------------------
 [![How to install | Python | | Anaconda | | Opencv library |](https://yt-embed.herokuapp.com/embed?v=eVV3byQlYvA)](https://www.youtube.com/watch?v=eVV3byQlYvA "How to install | Python | | Anaconda | | Opencv library |")


 Installing the required package’s:
-------------------------------------
• pip install -q git+https://github.com/tensorflow/docs 

• conda install -c conda-forge opencv=4.2.0

• pip install opencv-python

• pip install scikit-learn

• pip install scikit-image

• pip install matplotlib

• pip install tensorflow

• conda install opencv

• pip install shuffle

• pip install pandas

• pip install seaborn

• pip install keras

• pip install numpy

• pip install scipy

• pip install tqdm

License:
--------
MIT Licensed

Author:
-------
Sujit Mandal

LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/

Facebook : https://www.facebook.com/sujit.mandal.33671748

Twitter : https://twitter.com/mandalsujit37

