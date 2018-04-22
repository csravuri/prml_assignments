#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:32:16 2018

@author: Chandra Sekhar Ravuri
"""

# This code is for ES 647 Assignment 2
# Developed in Python 3.6 with Spyder
# Requred packages: numpy, sklearn, matplotlib, pandas
# Please see all the plots in maximised window, the may look overlapping in samll windows
# Details about credit card dataset was found at
# https://www.kaggle.com/lucabasa/credit-card-default-a-very-pedagogical-notebook

########################### Importing libraries ##############################
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree, metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

############################  Loading datasets  ###############################
mnist = fetch_mldata('MNIST original')

mnist_X = mnist.data
mnist_Y = mnist.target

#credit_card = np.array(pd.read_csv('/home/hadoop1/Documents/prml/assignment2/data/UCI_Credit_Card.csv',sep=','))
### credit_card = pd.read_csv('/home/hadoop1/Documents/prml/assignment2/data/UCI_Credit_Card.csv',sep=',')


#########################################################################################
##########################           Question 1           ###############################
#########################################################################################

######################## Train and test data split #############################
indx1 = np.random.permutation(len(mnist_Y)) # to shuffle randomly
tstsz = int(len(indx1)*0.8) # testset size here it is 80%

mnist_X_train, mnist_X_test = mnist_X[indx1[:tstsz],:], mnist_X[indx1[tstsz:],:]
mnist_Y_train, mnist_Y_test = mnist_Y[indx1[:tstsz]], mnist_Y[indx1[tstsz:]]
'''
# 1a)===================== K-nearest neighbors   ========================================

print('======================================================')
tim = time.time()

knn = KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1)
knn.fit(mnist_X_train, mnist_Y_train)

mnist_Y_pred_knn = knn.predict(mnist_X_test)
mnist_score_knn = knn.score(mnist_X_test, mnist_Y_test)



print('KNN took',int(time.time()-tim),'Sec and Score is ',round(mnist_score_knn,3),sep=' ')
'''
#================ 1b) Decision Tree ===================================================

print('======================================================')
tim = time.time()

Dt = tree.DecisionTreeClassifier()
Dt.fit(mnist_X_train, mnist_Y_train)

mnist_Y_pred_Dt = Dt.predict(mnist_X_test)
mnist_score_Dt = Dt.score(mnist_X_test, mnist_Y_test)
#%%----

mnist_Y_train_pred_Dt = Dt.predict(mnist_X_train)
Dt_train_confussion = metrics.confusion_matrix(mnist_Y_train, mnist_Y_train_pred_Dt)
Dt_test_confussion = metrics.confusion_matrix(mnist_Y_test, mnist_Y_pred_Dt)


Dt_1_accuracy = metrics.accuracy_score(mnist_Y_test, mnist_Y_pred_Dt, normalize=True)
# this top 1 accuracy

#Dt_accuacy_report = metrics.classification_report(mnist_Y_test, mnist_Y_pred_Dt)

mnist_Y_pred_Dt_proba = Dt.predict_proba(mnist_X_test)





print('Train data Confussion matrix for Decision tree for MNIST dataset \n')
Dt_conmat_train = pd.DataFrame(Dt_train_confussion) # train confussion matrix
print(Dt_conmat_train)

print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n')
print('Test data Confussion matrix for Decision tree for MNIST dataset \n')
Dt_conmat_test = pd.DataFrame(Dt_test_confussion) # test confussion matrix
print(Dt_conmat_test)


print('Decision tree took',int(time.time()-tim),'Sec and Score is ',round(mnist_score_Dt,3),sep=' ')

#===================  1c) SVM  ===================================================
'''
print('======================================================')
tim = time.time()

SVM = svm.LinearSVC()
SVM.fit(mnist_X_train, mnist_Y_train)

mnist_Y_pred_SVM = SVM.predict(mnist_X_test)
mnist_score_SVM = SVM.score(mnist_X_test, mnist_Y_test)

print('SVM took',int(time.time()-tim),'Sec and Score is ',round(mnist_score_SVM,3),sep=' ')

#================= 1d) Logistic Regression  ===================================================

print('======================================================')
tim = time.time()

LoR = LogisticRegression()
LoR.fit(mnist_X_train, mnist_Y_train)

mnist_Y_pred_LoR = LoR.predict(mnist_X_test)
mnist_score_LoR = LoR.score(mnist_X_test, mnist_Y_test)

print('Logistic regression took',int(time.time()-tim),'Sec and Score is ',round(mnist_score_LoR,3),sep=' ')

'''
print('======================================================')
print('======================================================')






#%% R & D

a = np.random.randint(1,10,(10,7))

b = np.zeros(a.shape[0])
for i in range(len(a)):
    a1 = np.argmax(a[i,:])
    a[i,a1]=0
    a2 = np.argmax(a[i,:])
    a[i,a2]=0
    a3 = np.argmax(a[i,:])
    
    if (i == a1 or i == a2 or i == a3):
        b[i] = 1
    
    
    
    








#%%

#########################################################################################
##########################           Question 2           ###############################
#########################################################################################

# 2a) K-nearest neighbors
# 2b) Decision Tree
# 2c) SVM
# 2d) Logistic Regression



