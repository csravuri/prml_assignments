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
from prettytable import PrettyTable
from tabulate import tabulate

############################  Loading datasets  ###############################
'''
mnist = fetch_mldata('MNIST original')

mnist_X = mnist.data
mnist_Y = mnist.target
'''

#credit_card = np.array(pd.read_csv('/home/hadoop1/Documents/prml/assignment2/data/UCI_Credit_Card.csv',sep=','))
credit_card = pd.read_csv('/home/hadoop1/Documents/prml/assignment2/data/UCI_Credit_Card.csv',sep=',')
credit = credit_card.values


credit_X = credit[:,1:-1] # discarding ID
credit_Y = credit[:,-1]


#########################################################################################
##########################           Question 1           ###############################
#########################################################################################




#########################################################################################
##########################           Question 2           ###############################
#########################################################################################

######################## Train and test data split #############################
indx2 = np.random.permutation(len(credit_Y)) # to shuffle randomly
tstsz = int(len(indx2)*0.8) # testset size here it is 80%

credit_X_train, credit_X_test = credit_X[indx2[:tstsz],:], credit_X[indx2[tstsz:],:]
credit_Y_train, credit_Y_test = credit_Y[indx2[:tstsz]], credit_Y[indx2[tstsz:]]

# 2a)===================== K-nearest neighbors   ========================================

tim = time.time()

knn = KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1)
knn.fit(credit_X_train, credit_Y_train)

credit_Y_test_pred_knn = knn.predict(credit_X_test)
credit_score_knn = knn.score(credit_X_test, credit_Y_test)

#--
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

#print(pd.DataFrame(credit_confu_train_knn))
print(' ')
print('Testing confussion matrix')
print(tabulate(credit_confu_test_knn, tablefmt='fancy_grid', showindex=True,headers=[0,1]))

#print(pd.DataFrame(credit_confu_test_knn))






tb_knn = PrettyTable(['Parameter', 'Value'])
tb_knn.add_row(['Accuracy', round(credit_acc_knn,4)])
tb_knn.add_row(['F1 Score', round(credit_f1_knn,4)])
tb_knn.add_row(['Precision', round(credit_prec_knn,4)])
tb_knn.add_row(['Recall Score', round(credit_recal_knn,4)])
tb_knn.align['Parameter'] = 'l'

print('')
print(tb_knn)

#print('Credit card data -- Accuracy with KNN:',credit_acc_knn)
#print('Credit card data -- F1 score with KNN:',credit_f1_knn)
#print('Credit card data -- Precision with KNN:',credit_prec_knn)
#print('Credit card data -- Recall score with KNN:',credit_recal_knn)
#print('Credit card data -- ROC curve with KNN:',credit_roc_knn)

plt.figure()
plt.title('ROC Curve for K-nearest neighbors')
plt.plot(credit_roc_knn[0],credit_roc_knn[1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

#--
print('K-nearest neighbors Running time is',int(time.time()-tim),'Sec.')

'''
#================ 2b) Decision Tree ===================================================

tim = time.time()

Dt = tree.DecisionTreeClassifier()
Dt.fit(credit_X_train, credit_Y_train)

credit_Y_test_pred_Dt = Dt.predict(credit_X_test)
credit_score_Dt = Dt.score(credit_X_test, credit_Y_test)
#%%----

credit_Y_train_pred_Dt = Dt.predict(credit_X_train)
Dt_train_confussion = metrics.confusion_matrix(credit_Y_train, credit_Y_train_pred_Dt)
Dt_test_confussion = metrics.confusion_matrix(credit_Y_test, credit_Y_pred_Dt)


Dt_1_accuracy = metrics.accuracy_score(credit_Y_test, credit_Y_pred_Dt, normalize=True)
# this top 1 accuracy

#Dt_accuacy_report = metrics.classification_report(credit_Y_test, credit_Y_pred_Dt)

credit_Y_pred_Dt_proba = Dt.predict_proba(credit_X_test)





print('Train data Confussion matrix for Decision tree for Credit card dataset \n')
Dt_conmat_train = pd.DataFrame(Dt_train_confussion) # train confussion matrix
print(Dt_conmat_train)

print('Test data Confussion matrix for Decision tree for Credit card dataset \n')
Dt_conmat_test = pd.DataFrame(Dt_test_confussion) # test confussion matrix
print(Dt_conmat_test)


print('Decision tree took',int(time.time()-tim),'Sec and Score is ',round(credit_score_Dt,3),sep=' ')

#===================  2c) SVM  ===================================================

tim = time.time()

SVM = svm.LinearSVC()
SVM.fit(credit_X_train, credit_Y_train)

credit_Y_test_pred_SVM = SVM.predict(credit_X_test)
credit_score_SVM = SVM.score(credit_X_test, credit_Y_test)

print('SVM took',int(time.time()-tim),'Sec and Score is ',round(credit_score_SVM,3),sep=' ')

#================= 2d) Logistic Regression  ===================================================

tim = time.time()

LoR = LogisticRegression()
LoR.fit(credit_X_train, credit_Y_train)

credit_Y_test_pred_LoR = LoR.predict(credit_X_test)
credit_score_LoR = LoR.score(credit_X_test, credit_Y_test)

print('Logistic regression took',int(time.time()-tim),'Sec and Score is ',round(credit_score_LoR,3),sep=' ')

'''


