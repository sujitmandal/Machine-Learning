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

 Unsupervised learning:
------------------------
• Unsupervised learning which finds hidden patterns in data. It is used to draw inferences from datasets consisting of input data without labelled responses. Unsupervised uses clustering and association techniques to develop predictive models.

 Clustering:
-------------
• Clustering is the most common unsupervised learning technique. It is used for exploratory data analysis to find hidden patterns. It is used to grope the data for similar type data. For example gene sequence analysis, market research, and object recognition.


K-means cluster:
---------------

	K = { 2, 3, 4, 10, 11, 12, 20, 25, 30 } 

	Let  k = 2 , where we are going to find two clusters.

	Let mean m1, m2.
	Therefore M1 = 4, M2 = 12  [ note: (K - M1) = X1, (K - M2) = X2 ,min value will be taken ]

# Iteration 1
	k1 = { 2, 3, 4 }
	K2 = { 10, 11, 12, 20, 25, 30 }

	M1 = (2+3+4 / 3 )      M2 = (10+11+12+20+25+30 / 6)
	M1 = (9 / 3)  	       M2 = (108 / 6)
	M1 = 3                 M2 = 18

# Iteration 2 :
	K1 = { 2, 3, 4, 10 }
	K2 = { 11, 12, 20, 25, 30 }
	
	M1 = (2+3+4+10 / 4)    M2 = (11+12+20+25+30 / 5)
	M1 = (19 / 4)          M2 = (98 / 5)
	M1 = 4.75	       M2 = 19.5
	M1 = 5		       M2 = 20


# Iteration 3 :	
	K1 = { 2, 3, 4, 10, 12 }
	K2 = { 20, 25, 30 }
	
	M1 = (2+3+4+10+11+12 / 6)     M2 = (20+25+30 / 3)
	M1 = (42 / 6)		      M2 = (75 / 3)
	M1 = 7			      M2 = 25
	
# Iteration 4 : 
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

 Installing the required package’s:
-------------------------------------
• pip install scikit-learn

• pip install scikit-image

• pip install matplotlib

• conda install opencv

• pip install pandas

• pip install seaborn

• pip install numpy

• pip install scipy

License:
--------
MIT Licensed

Author:
-------
Sujit Mandal

LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/

Facebook : https://www.facebook.com/sujit.mandal.33671748

Twitter : https://twitter.com/mandalsujit37
