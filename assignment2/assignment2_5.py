#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:32:16 2018

@author: Chandra Sekhar Ravuri
"""

# This code is for ES 647 Assignment 2
# Developed in Python 3.6 with Spyder
# Requred packages: numpy, sklearn, matplotlib, pandas, time, prettytable, tabulate

# I found details about credit card dataset at
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
from prettytable import PrettyTable
from tabulate import tabulate

############################  Loading datasets  ###############################
'''
mnist = fetch_mldata('MNIST original')

mnist_X = mnist.data
mnist_Y = mnist.target
'''
#---------------------------------------------------------------------------#
credit_card = pd.read_csv('/home/hadoop1/Documents/prml/assignment2/data/UCI_Credit_Card.csv',sep=',')
credit = credit_card.values

credit_X = credit[:,1:-1] # discarding col 'ID'
credit_Y = credit[:,-1]

print('Data loaded!')


#########################################################################################
##########################           Question 1           ###############################
#########################################################################################




#########################################################################################
##########################           Question 2           ###############################
#########################################################################################

######################## Train and test data split #############################
indx2 = np.random.permutation(len(credit_Y)) # to shuffle randomly
tstsz = int(len(indx2)*0.8) # trainset size here it is 80%

credit_X_train, credit_X_test = credit_X[indx2[:tstsz],:], credit_X[indx2[tstsz:],:]
credit_Y_train, credit_Y_test = credit_Y[indx2[:tstsz]], credit_Y[indx2[tstsz:]]

#### Normalizing credit card data ####
credit_X_train_mean = credit_X_train.mean(axis=0)
credit_X_train_std = credit_X_train.std(axis=0)

credit_X_train = (credit_X_train - credit_X_train_mean)/credit_X_train_std 
credit_X_test = (credit_X_test - credit_X_train_mean)/credit_X_train_std

print('Feature vectors were normalized!')

# 2a)===================== K-nearest neighbors   ========================================

print('=======================================================================')
tim = time.time()

knn = KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1)
knn.fit(credit_X_train, credit_Y_train)

credit_Y_test_pred_knn = knn.predict(credit_X_test)
credit_score_knn = knn.score(credit_X_test, credit_Y_test)


credit_Y_train_pred_knn = knn.predict(credit_X_train)

credit_confu_train_knn = metrics.confusion_matrix(credit_Y_train, credit_Y_train_pred_knn)
credit_confu_test_knn = metrics.confusion_matrix(credit_Y_test, credit_Y_test_pred_knn)

credit_acc_knn = metrics.accuracy_score(credit_Y_test, credit_Y_test_pred_knn)

credit_f1_knn = metrics.f1_score(credit_Y_test, credit_Y_test_pred_knn)

credit_prec_knn = metrics.precision_score(credit_Y_test, credit_Y_test_pred_knn)

credit_recal_knn = metrics.recall_score(credit_Y_test, credit_Y_test_pred_knn)

credit_roc_knn = metrics.roc_curve(credit_Y_test, credit_Y_test_pred_knn,drop_intermediate=False)

print('#####   Credit card data with K-nearest neighbors  #####')
print('')
print('Training confussion matrix')
print(tabulate(credit_confu_train_knn, tablefmt='fancy_grid', showindex=True,headers=[0,1]))

print(' ')
print('Testing confussion matrix')
print(tabulate(credit_confu_test_knn, tablefmt='fancy_grid', showindex=True,headers=[0,1]))


tb_knn = PrettyTable(['Parameter', 'Value'])
tb_knn.add_row(['Accuracy', round(credit_acc_knn,4)])
tb_knn.add_row(['F1 Score', round(credit_f1_knn,4)])
tb_knn.add_row(['Precision', round(credit_prec_knn,4)])
tb_knn.add_row(['Recall Score', round(credit_recal_knn,4)])
tb_knn.align['Parameter'] = 'l'

print('')
print(tb_knn)

plt.figure()
plt.title('ROC Curve for K-nearest neighbors')
plt.plot(credit_roc_knn[0],credit_roc_knn[1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print('K-nearest neighbors Running time is',int(time.time()-tim),'Sec.')


#================ 2b) Decision Tree ===================================================

print('=======================================================================')
tim = time.time()

Dt = tree.DecisionTreeClassifier()
Dt.fit(credit_X_train, credit_Y_train)

credit_Y_test_pred_Dt = Dt.predict(credit_X_test)
credit_score_Dt = Dt.score(credit_X_test, credit_Y_test)


credit_Y_train_pred_Dt = Dt.predict(credit_X_train)

credit_confu_train_Dt = metrics.confusion_matrix(credit_Y_train, credit_Y_train_pred_Dt)
credit_confu_test_Dt = metrics.confusion_matrix(credit_Y_test, credit_Y_test_pred_Dt)

credit_acc_Dt = metrics.accuracy_score(credit_Y_test, credit_Y_test_pred_Dt)

credit_f1_Dt = metrics.f1_score(credit_Y_test, credit_Y_test_pred_Dt)

credit_prec_Dt = metrics.precision_score(credit_Y_test, credit_Y_test_pred_Dt)

credit_recal_Dt = metrics.recall_score(credit_Y_test, credit_Y_test_pred_Dt)

credit_roc_Dt = metrics.roc_curve(credit_Y_test, credit_Y_test_pred_Dt,drop_intermediate=False)

print('#####   Credit card data with Decision Tree  #####')
print('')
print('Training confussion matrix')
print(tabulate(credit_confu_train_Dt, tablefmt='fancy_grid', showindex=True,headers=[0,1]))

print(' ')
print('Testing confussion matrix')
print(tabulate(credit_confu_test_Dt, tablefmt='fancy_grid', showindex=True,headers=[0,1]))


tb_Dt = PrettyTable(['Parameter', 'Value'])
tb_Dt.add_row(['Accuracy', round(credit_acc_Dt,4)])
tb_Dt.add_row(['F1 Score', round(credit_f1_Dt,4)])
tb_Dt.add_row(['Precision', round(credit_prec_Dt,4)])
tb_Dt.add_row(['Recall Score', round(credit_recal_Dt,4)])
tb_Dt.align['Parameter'] = 'l'

print('')
print(tb_Dt)

plt.figure()
plt.title('ROC Curve for Decision tree')
plt.plot(credit_roc_Dt[0],credit_roc_Dt[1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print('Decision Tree Running time is',int(time.time()-tim),'Sec.')

#===================  2c) SVM  ===================================================

print('=======================================================================')
tim = time.time()

SVM = svm.LinearSVC()
SVM.fit(credit_X_train, credit_Y_train)

credit_Y_test_pred_SVM = SVM.predict(credit_X_test)
credit_score_SVM = SVM.score(credit_X_test, credit_Y_test)

credit_Y_train_pred_SVM = SVM.predict(credit_X_train)

credit_confu_train_SVM = metrics.confusion_matrix(credit_Y_train, credit_Y_train_pred_SVM)
credit_confu_test_SVM = metrics.confusion_matrix(credit_Y_test, credit_Y_test_pred_SVM)

credit_acc_SVM = metrics.accuracy_score(credit_Y_test, credit_Y_test_pred_SVM)

credit_f1_SVM = metrics.f1_score(credit_Y_test, credit_Y_test_pred_SVM)

credit_prec_SVM = metrics.precision_score(credit_Y_test, credit_Y_test_pred_SVM)

credit_recal_SVM = metrics.recall_score(credit_Y_test, credit_Y_test_pred_SVM)

credit_roc_SVM = metrics.roc_curve(credit_Y_test, credit_Y_test_pred_SVM,drop_intermediate=False)

print('#####   Credit card data with SVM  #####')
print('')
print('Training confussion matrix')
print(tabulate(credit_confu_train_SVM, tablefmt='fancy_grid', showindex=True,headers=[0,1]))

print(' ')
print('Testing confussion matrix')
print(tabulate(credit_confu_test_SVM, tablefmt='fancy_grid', showindex=True,headers=[0,1]))


tb_SVM = PrettyTable(['Parameter', 'Value'])
tb_SVM.add_row(['Accuracy', round(credit_acc_SVM,4)])
tb_SVM.add_row(['F1 Score', round(credit_f1_SVM,4)])
tb_SVM.add_row(['Precision', round(credit_prec_SVM,4)])
tb_SVM.add_row(['Recall Score', round(credit_recal_SVM,4)])
tb_SVM.align['Parameter'] = 'l'

print('')
print(tb_SVM)

plt.figure()
plt.title('ROC Curve for SVM')
plt.plot(credit_roc_SVM[0],credit_roc_SVM[1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print('SVM Running time is',int(time.time()-tim),'Sec.')


#================= 2d) Logistic Regression  ===================================================

print('=======================================================================')
tim = time.time()

LoR = LogisticRegression(C=0.85, dual=True)
LoR.fit(credit_X_train, credit_Y_train)

credit_Y_test_pred_LoR = LoR.predict(credit_X_test)
credit_score_LoR = LoR.score(credit_X_test, credit_Y_test)

credit_Y_train_pred_LoR = LoR.predict(credit_X_train)

credit_confu_train_LoR = metrics.confusion_matrix(credit_Y_train, credit_Y_train_pred_LoR)
credit_confu_test_LoR = metrics.confusion_matrix(credit_Y_test, credit_Y_test_pred_LoR)

credit_acc_LoR = metrics.accuracy_score(credit_Y_test, credit_Y_test_pred_LoR)

credit_f1_LoR = metrics.f1_score(credit_Y_test, credit_Y_test_pred_LoR)

credit_prec_LoR = metrics.precision_score(credit_Y_test, credit_Y_test_pred_LoR)

credit_recal_LoR = metrics.recall_score(credit_Y_test, credit_Y_test_pred_LoR)

credit_roc_LoR = metrics.roc_curve(credit_Y_test, credit_Y_test_pred_LoR,drop_intermediate=False)

print('#####   Credit card data with Logistic regression  #####')
print('')
print('Training confussion matrix')
print(tabulate(credit_confu_train_LoR, tablefmt='fancy_grid', showindex=True,headers=[0,1]))

print(' ')
print('Testing confussion matrix')
print(tabulate(credit_confu_test_LoR, tablefmt='fancy_grid', showindex=True,headers=[0,1]))


tb_LoR = PrettyTable(['Parameter', 'Value'])
tb_LoR.add_row(['Accuracy', round(credit_acc_LoR,4)])
tb_LoR.add_row(['F1 Score', round(credit_f1_LoR,4)])
tb_LoR.add_row(['Precision', round(credit_prec_LoR,4)])
tb_LoR.add_row(['Recall Score', round(credit_recal_LoR,4)])
tb_LoR.align['Parameter'] = 'l'

print('')
print(tb_LoR)

plt.figure()
plt.title('ROC Curve for Logistic regression')
plt.plot(credit_roc_LoR[0],credit_roc_LoR[1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print('Logistic regression Running time is',int(time.time()-tim),'Sec.')

print('=======================================================================')

