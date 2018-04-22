#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 23:46:48 2018

@author: Chandra Sekhar Ravuri
"""

import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

plt.close('all')
#################### Loading Data    #################### 
boston = datasets.load_boston()     #(Boston)
boston_X_data = boston['data']
boston_Y_data = boston['target']

percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(boston_X_data)/100, np.int)

lamda = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]) # possible to increase number of lambda values

boston_test_error_ridge_global = []
boston_train_error_ridge_global = []
boston_r2_score_ridge_global = []
boston_cv_global = []

for m in percent_data:
    
    boston_test_error_ridge = []
    boston_train_error_ridge = []
    boston_r2_score_ridge = []
    boston_cv = []
    
    for la in lamda:
        ########### Shuffling the data everytime ###############
        boston_shuffle = np.concatenate((boston_X_data, np.expand_dims(boston_Y_data,1)),axis=1)
        np.random.shuffle(boston_shuffle) # shuffel along first axis only
        
        ########### Spliting X and Y again ##################
        boston_X_data_shffle = boston_shuffle[:,:-1]
        boston_Y_data_shffle = boston_shuffle[:,-1]
        
        ########### Data spliting train & test according to 'percent_data' #####
        boston_X_train = boston_X_data_shffle[:m,:]
        boston_X_test = boston_X_data_shffle[m:,:]
        
        boston_Y_train = boston_Y_data_shffle[:m]
        boston_Y_test = boston_Y_data_shffle[m:]
        
        ########## Ridge regresion fitting #############
        boston_ridge_reg = linear_model.Ridge(alpha=la, fit_intercept=True, normalize=True, solver='auto')
        boston_ridge_reg.fit(boston_X_train, boston_Y_train)
        
        ########## Predection and Reporting ################
        boston_Y_pred = boston_ridge_reg.predict(boston_X_test)
        boston_Y_train_pred = boston_ridge_reg.predict(boston_X_train)
        
        
        test_error_boston = mean_squared_error(boston_Y_test, boston_Y_pred)
        train_error_boston = mean_squared_error(boston_Y_train, boston_Y_train_pred)
        r2_score_boston = r2_score(boston_Y_train, boston_Y_train_pred)
        
        ########## Cross validation K = 5 ##################
        cross_val_boston = np.abs(np.mean(cross_validate(boston_ridge_reg, boston_X_train, boston_Y_train, cv=5, scoring='neg_mean_squared_error')['test_score']))
        
        boston_test_error_ridge.append(test_error_boston)
        boston_train_error_ridge.append(train_error_boston)
        boston_r2_score_ridge.append(r2_score_boston)
        boston_cv.append(cross_val_boston)
    
    
    boston_test_error_ridge_global.append(boston_test_error_ridge)
    boston_train_error_ridge_global.append(boston_train_error_ridge)
    boston_r2_score_ridge_global.append(boston_r2_score_ridge)
    boston_cv_global.append(boston_cv)

########### Ploting the reports ####################
ymin_error = np.min(np.array(boston_test_error_ridge_global+boston_train_error_ridge_global+boston_cv_global))
ymax_error = np.max(np.array(boston_test_error_ridge_global+boston_train_error_ridge_global+boston_cv_global))

ymin_r2 = np.min(np.array(boston_r2_score_ridge_global))
ymax_r2 = np.max(np.array(boston_r2_score_ridge_global))

plt.figure(facecolor='0.85')
plt.suptitle('Boston data Ridge Regression (with $ L_{2} $ regularization)', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(boston_test_error_ridge_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, boston_train_error_ridge_global[le], 'b')
    plt.plot(lamda, boston_test_error_ridge_global[le], 'r')
    plt.plot(lamda, boston_cv_global[le], '--r')
    plt.legend(['Training Error', 'Test Error', 'Cross val error'])
    plt.xlabel('$ \lambda $'), plt.ylabel('Mean square error')
    plt.ylim(ymin_error - min(ymin_error, 20), ymax_error + min(20, ymax_error*0.1))
    plt.title('Training and test error\n for '+str(percent_data[le])+' train examples')
    plt.grid(axis='both')
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, boston_r2_score_ridge_global[le], 'k')
    plt.xlabel('$ \lambda $'), plt.ylabel('$R^{2}$ Score')
    plt.ylim(ymin_r2 - ymin_r2*0.1, ymax_r2 + ymax_r2*0.1)
    plt.title('$R^{2}$ score \n for '+str(percent_data[le])+' train examples')
    plt.grid(axis='both')
    

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()


"""
##%%
percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(boston_X_data)/100, np.int)

lamda = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]) # possible to increase number of lambda values

boston_test_error_ridge_global = []
boston_train_error_ridge_global = []
boston_r2_score_ridge_global = []
boston_cv_global = []

for m in percent_data:
    
    boston_test_error_ridge = []
    boston_train_error_ridge = []
    boston_r2_score_ridge = []
    boston_cv = []
    
    for la in lamda:
        ########### Shuffling the data everytime ###############
        boston_shuffle = np.concatenate((boston_X_data, np.expand_dims(boston_Y_data,1)),axis=1)
        np.random.shuffle(boston_shuffle) # shuffel along first axis only
        
        ########### Spliting X and Y again ##################
        boston_X_data_shffle = boston_shuffle[:,:-1]
        boston_Y_data_shffle = boston_shuffle[:,-1]
        
        ########### Data spliting train & test according to 'percent_data' #####
        boston_X_train = boston_X_data_shffle[:m,:]
        boston_X_test = boston_X_data_shffle[m:,:]
        
        boston_Y_train = boston_Y_data_shffle[:m]
        boston_Y_test = boston_Y_data_shffle[m:]
        
        ########## Ridge regresion fitting #############
        boston_ridge_reg = linear_model.Ridge(alpha=la, fit_intercept=True, normalize=True, solver='auto')
        boston_ridge_reg.fit(boston_X_train, boston_Y_train)
        
        ########## Predection and Reporting ################
        boston_Y_pred = boston_ridge_reg.predict(boston_X_test)
        boston_Y_train_pred = boston_ridge_reg.predict(boston_X_train)
        
        
        test_error_boston = mean_squared_error(boston_Y_test, boston_Y_pred)
        train_error_boston = mean_squared_error(boston_Y_train, boston_Y_train_pred)
        r2_score_boston = r2_score(boston_Y_train, boston_Y_train_pred)
        
        ########## Cross validation K = 5 ##################
        cross_val_boston = np.abs(np.mean(cross_validate(boston_ridge_reg, boston_X_train, boston_Y_train, cv=5, scoring='neg_mean_squared_error')['test_score']))
        
        boston_test_error_ridge.append(test_error_boston)
        boston_train_error_ridge.append(train_error_boston)
        boston_r2_score_ridge.append(r2_score_boston)
        boston_cv.append(cross_val_boston)
    
    
    boston_test_error_ridge_global.append(boston_test_error_ridge)
    boston_train_error_ridge_global.append(boston_train_error_ridge)
    boston_r2_score_ridge_global.append(boston_r2_score_ridge)
    boston_cv_global.append(boston_cv)

########### Ploting the reports ####################
ymin_error = np.min(np.array(boston_test_error_ridge_global+boston_train_error_ridge_global+boston_cv_global))
ymax_error = np.max(np.array(boston_test_error_ridge_global+boston_train_error_ridge_global+boston_cv_global))

ymin_r2 = np.min(np.array(boston_r2_score_ridge_global))
ymax_r2 = np.max(np.array(boston_r2_score_ridge_global))

plt.figure(facecolor='0.85')
plt.suptitle('Boston data Ridge Regression (with $ L_{2} $ regularization)', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(boston_test_error_ridge_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, boston_train_error_ridge_global[le], 'b')
    plt.plot(lamda, boston_test_error_ridge_global[le], 'r')
    plt.plot(lamda, boston_cv_global[le], '--r')
    plt.legend(['Training Error', 'Test Error', 'Cross val error'])
    plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
    plt.ylim(ymin_error - min(ymin_error, 20), ymax_error + min(20, ymax_error*0.1))
    plt.title('Training and test error\n for '+str(percent_data[le])+' train example')
    plt.grid(axis='both')
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, boston_r2_score_ridge_global[le], 'k')
    plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
    plt.ylim(ymin_r2 - ymin_r2*0.1, ymax_r2 + ymax_r2*0.1)
    plt.title('$R^{2}$ score \n for '+str(percent_data[le])+' train example')
    plt.grid(axis='both')
    

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()

"""