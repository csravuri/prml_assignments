#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 14:37:24 2018

@author: Chandra Sekhar Ravuri
"""
# This code is for ES 647 Assignment 1
# Developed in Python 3.6 with Spyder
# Requred packages: numpy, sklearn, matplotlib
# Please see all the plots in maximised window, the may look overlapping in samll windows

####################  Required Libraries  ###################
import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

#################### Loading Data    #################### 
boston = datasets.load_boston()     #(Boston)
boston_X_data = boston['data']
boston_Y_data = boston['target']

diabetes = datasets.load_diabetes()     #(Diabetes)
diabetes_X_data = diabetes['data']
diabetes_Y_data = diabetes['target']

plt.close('all') # closes previous figure windows if any
#########################################################################################
##########################           Question 1           ###############################
#########################################################################################

percent_data = np.array([50, 60, 70, 80, 90, 95, 99])
percent_data = np.array(percent_data*len(boston_X_data)/100, np.int)

boston_test_error = []
boston_train_error = []
boston_r2_score = []

for m in percent_data:
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
    
    ########## Least square regresion fitting #############
    boston_least_reg = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    boston_least_reg.fit(boston_X_train, boston_Y_train)
    
    ########## Predection and Reporting ################
    boston_Y_pred = boston_least_reg.predict(boston_X_test)
    boston_Y_train_pred = boston_least_reg.predict(boston_X_train)
    
    test_error_boston = mean_squared_error(boston_Y_test, boston_Y_pred)
    train_error_boston = mean_squared_error(boston_Y_train, boston_Y_train_pred)
    r2_score_boston = r2_score(boston_Y_train, boston_Y_train_pred)
    
    boston_test_error.append(test_error_boston)
    boston_train_error.append(train_error_boston)
    boston_r2_score.append(r2_score_boston)
    
########### Ploting the reports ####################
plt.figure(facecolor='0.85')
plt.suptitle('Q1) Boston data Least Square Regression (without regularization)', fontsize=16, fontweight='bold')
plt.subplot(121)
plt.plot(percent_data, boston_train_error, 'b')
plt.plot(percent_data, boston_test_error,'r')
plt.legend(['Training Error', 'Test Error'])
plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
plt.title('Training and test error')
plt.grid(axis='both')

plt.subplot(122)
plt.plot(percent_data, boston_r2_score,'k')
plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
plt.title('$R^{2}$ score')
plt.grid(axis='both')
plt.tight_layout()
plt.show()

#%% l2 regression

percent_data = np.array([50, 60, 70, 80, 90, 95, 99])
percent_data = np.array(percent_data*len(boston_X_data)/100, np.int)

lamda = np.array([0, 0.01, 0.1, 1]) # possible to increase number of lambda values

boston_test_error_ridge_global = []
boston_train_error_ridge_global = []
boston_r2_score_ridge_global = []

for la in lamda:
    
    boston_test_error_ridge = []
    boston_train_error_ridge = []
    boston_r2_score_ridge = []
    
    for m in percent_data:
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
        
        boston_test_error_ridge.append(test_error_boston)
        boston_train_error_ridge.append(train_error_boston)
        boston_r2_score_ridge.append(r2_score_boston)
        
    boston_test_error_ridge_global.append(boston_test_error_ridge)
    boston_train_error_ridge_global.append(boston_train_error_ridge)
    boston_r2_score_ridge_global.append(boston_r2_score_ridge)

########### Ploting the reports ####################
ymin_error = np.min(np.array(boston_test_error_ridge_global+boston_train_error_ridge_global))
ymax_error = np.max(np.array(boston_test_error_ridge_global+boston_train_error_ridge_global))

ymin_r2 = np.min(np.array(boston_r2_score_ridge_global))
ymax_r2 = np.max(np.array(boston_r2_score_ridge_global))

plt.figure(facecolor='0.85')
plt.suptitle('Q1) Boston data Ridge Regression (with $ L_{2} $ regularization)', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(boston_test_error_ridge_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(percent_data, boston_train_error_ridge_global[le], 'b')
    plt.plot(percent_data, boston_test_error_ridge_global[le], 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
    plt.ylim(ymin_error - min(ymin_error, 20), ymax_error + min(20, ymax_error*0.1))
    plt.title('Training and test error\n for $ \lambda = $'+str(lamda[le]))
    plt.grid(axis='both')
    
    plt.subplot(fig_count+4)
    plt.plot(percent_data, boston_r2_score_ridge_global[le], 'k')
    plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
    plt.ylim(ymin_r2 - ymin_r2*0.1, ymax_r2 + ymax_r2*0.1)
    plt.title('$R^{2}$ score \n for $ \lambda = $'+str(lamda[le]))
    plt.grid(axis='both')
    

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()

#%%
#########################################################################################
##########################           Question 2           ###############################
#########################################################################################

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
plt.suptitle('Q2) Boston data Ridge Regression (with $ L_{2} $ regularization)', fontsize=16, fontweight='bold')
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


#%%
#########################################################################################
##########################           Question 3           ###############################
#########################################################################################

percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(boston_X_data)/100, np.int)

lamda = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]) # possible to increase number of lambda values

boston_test_error_lasso_global = []
boston_train_error_lasso_global = []
boston_r2_score_lasso_global = []
boston_cv_global = []

for m in percent_data:
    
    boston_test_error_lasso = []
    boston_train_error_lasso = []
    boston_r2_score_lasso = []
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
        
        ########## lasso regresion fitting #############
        boston_lasso_reg = linear_model.Lasso(alpha=la, fit_intercept=True, normalize=True, selection='cyclic')
        boston_lasso_reg.fit(boston_X_train, boston_Y_train)
        
        ########## Predection and Reporting ################
        boston_Y_pred = boston_lasso_reg.predict(boston_X_test)
        boston_Y_train_pred = boston_lasso_reg.predict(boston_X_train)
        
        
        test_error_boston = mean_squared_error(boston_Y_test, boston_Y_pred)
        train_error_boston = mean_squared_error(boston_Y_train, boston_Y_train_pred)
        r2_score_boston = r2_score(boston_Y_train, boston_Y_train_pred)
        
        ########## Cross validation K = 5 ##################
        cross_val_boston = np.abs(np.mean(cross_validate(boston_lasso_reg, boston_X_train, boston_Y_train, cv=5, scoring='neg_mean_squared_error')['test_score']))
        
        boston_test_error_lasso.append(test_error_boston)
        boston_train_error_lasso.append(train_error_boston)
        boston_r2_score_lasso.append(r2_score_boston)
        boston_cv.append(cross_val_boston)
    
    
    boston_test_error_lasso_global.append(boston_test_error_lasso)
    boston_train_error_lasso_global.append(boston_train_error_lasso)
    boston_r2_score_lasso_global.append(boston_r2_score_lasso)
    boston_cv_global.append(boston_cv)

########### Ploting the reports ####################
ymin_error = np.min(np.array(boston_test_error_lasso_global+boston_train_error_lasso_global+boston_cv_global))
ymax_error = np.max(np.array(boston_test_error_lasso_global+boston_train_error_lasso_global+boston_cv_global))

ymin_r2 = np.min(np.array(boston_r2_score_lasso_global))
ymax_r2 = np.max(np.array(boston_r2_score_lasso_global))

plt.figure(facecolor='0.85')
plt.suptitle('Q3) Boston data Lasso Regression (with $ L_{1} $ regularization)', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(boston_test_error_lasso_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, boston_train_error_lasso_global[le], 'b')
    plt.plot(lamda, boston_test_error_lasso_global[le], 'r')
    plt.plot(lamda, boston_cv_global[le], '--r')
    plt.legend(['Training Error', 'Test Error', 'Cross val error'])
    plt.xlabel('$ \lambda $'), plt.ylabel('Mean square error')
    plt.ylim(ymin_error - min(ymin_error, 20), ymax_error + min(20, ymax_error*0.1))
    
    plt.title('Training and test error\n for '+str(percent_data[le])+' train examples')
    plt.grid(axis='both')
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, boston_r2_score_lasso_global[le], 'k')
    plt.xlabel('$ \lambda $'), plt.ylabel('$R^{2}$ Score')
    plt.ylim(ymin_r2 - max(0.1,ymin_r2*0.1), ymax_r2 + ymax_r2*0.1)
    plt.title('$R^{2}$ score \n for '+str(percent_data[le])+' train examples')
    plt.grid(axis='both')
    

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()


#%%
#########################################################################################
##########################           Question 4           ###############################
#########################################################################################

percent_data = np.array([50, 60, 70, 80, 90, 95, 99])
percent_data = np.array(percent_data*len(boston_X_data)/100, np.int)

boston_test_error = []
boston_train_error = []
boston_r2_score = []

for m in percent_data:
    ########### Shuffling the data everytime ###############
    boston_shuffle = np.concatenate((boston_X_data, np.expand_dims(boston_Y_data,1)),axis=1)
    np.random.shuffle(boston_shuffle) # shuffel along first axis only
    
    ########## Data Normalizaton and Bias #####################
    boston_shuffle_normal = (boston_shuffle - np.mean(boston_shuffle,axis=0))/np.var(boston_shuffle,axis=0) #np.linalg.norm(boston_shuffle,axis=0)
    boston_shuffle_normal = np.concatenate((np.expand_dims(np.ones(len(boston_shuffle_normal)),1), boston_shuffle_normal),axis=1)
    
    ########### Spliting X and Y again ##################
    boston_X_data_shffle = boston_shuffle_normal[:,:-1]
    boston_Y_data_shffle = boston_shuffle_normal[:,-1]
    
    ########### Data spliting train & test according to 'percent_data' #####
    boston_X_train = boston_X_data_shffle[:m,:]
    boston_X_test = boston_X_data_shffle[m:,:]
    
    boston_Y_train = boston_Y_data_shffle[:m]
    boston_Y_test = boston_Y_data_shffle[m:]
    
    ########## Least square regresion fitting #############
    boston_XTX_inv = np.linalg.inv(np.matmul(np.transpose(boston_X_train), boston_X_train))
    boston_W_star = np.matmul(boston_XTX_inv, np.matmul(np.transpose(boston_X_train),boston_Y_train))
    
    ########## To scale results #########################
    boston_var = np.var(boston_shuffle[:,-1])
    boston_mean = np.mean(boston_shuffle[:,-1])
    
    ########## Predection and Reporting ################
    boston_Y_pred = np.matmul(boston_X_test, boston_W_star)
    boston_Y_train_pred = np.matmul(boston_X_train, boston_W_star)    
    
    test_error_boston = np.mean((boston_Y_test*boston_var - boston_Y_pred*boston_var)**2) # multiply result with variance again
    train_error_boston = np.mean((boston_Y_train*boston_var - boston_Y_train_pred*boston_var)**2)
    
    total_variance_boston = np.var(boston_Y_train*boston_var)
    explain_variance_boston = np.var(boston_Y_train_pred*boston_var)
    r2_score_boston = explain_variance_boston/total_variance_boston
    
    boston_test_error.append(test_error_boston) 
    boston_train_error.append(train_error_boston) 
    boston_r2_score.append(r2_score_boston)

    
########### Ploting the reports ####################
plt.figure(facecolor='0.85')
plt.suptitle('Q4) Boston data Least Square Regression (without regularization) own code', fontsize=16, fontweight='bold')
plt.subplot(121)
plt.plot(percent_data, boston_train_error, 'b')
plt.plot(percent_data, boston_test_error,'r')
plt.legend(['Training Error', 'Test Error'])
plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
plt.title('Training and test error')
plt.grid(axis='both')

plt.subplot(122)
plt.plot(percent_data, boston_r2_score,'k')
plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
plt.title('$R^{2}$ score')
plt.grid(axis='both')
plt.tight_layout()
plt.show()

#%% l2 regression

percent_data = np.array([50, 60, 70, 80, 90, 95, 99])
percent_data = np.array(percent_data*len(boston_X_data)/100, np.int)

lamda = np.array([0, 0.01, 0.1, 1]) # possible to increase number of lambda values

boston_test_error_ridge_global = []
boston_train_error_ridge_global = []
boston_r2_score_ridge_global = []

for la in lamda:
    
    boston_test_error_ridge = []
    boston_train_error_ridge = []
    boston_r2_score_ridge = []
    
    for m in percent_data:
        ########### Shuffling the data everytime ###############
        boston_shuffle = np.concatenate((boston_X_data, np.expand_dims(boston_Y_data,1)),axis=1)
        np.random.shuffle(boston_shuffle) # shuffel along first axis only
        
        ########## Data Normalizaton and Bias #####################
        boston_shuffle_normal = (boston_shuffle - np.mean(boston_shuffle,axis=0))/np.var(boston_shuffle,axis=0) #np.linalg.norm(boston_shuffle,axis=0)
        boston_shuffle_normal = np.concatenate((np.expand_dims(np.ones(len(boston_shuffle_normal)),1), boston_shuffle_normal),axis=1)
        
        ########### Spliting X and Y again ##################
        boston_X_data_shffle = boston_shuffle_normal[:,:-1]
        boston_Y_data_shffle = boston_shuffle_normal[:,-1]
        
        ########### Data spliting train & test according to 'percent_data' #####
        boston_X_train = boston_X_data_shffle[:m,:]
        boston_X_test = boston_X_data_shffle[m:,:]
        
        boston_Y_train = boston_Y_data_shffle[:m]
        boston_Y_test = boston_Y_data_shffle[m:]
        
        ########## Ridge regresion fitting #############
        boston_XTX_inv_ridge = np.linalg.inv(np.matmul(np.transpose(boston_X_train), boston_X_train)+la*np.eye(boston_X_train.shape[-1]))
        boston_W_star = np.matmul(boston_XTX_inv_ridge, np.matmul(np.transpose(boston_X_train),boston_Y_train))
        
        ########## To scale results #########################
        boston_var = np.var(boston_shuffle[:,-1])
        boston_mean = np.mean(boston_shuffle[:,-1])
        
        ########## Predection and Reporting ################
        boston_Y_pred = np.matmul(boston_X_test, boston_W_star)
        boston_Y_train_pred = np.matmul(boston_X_train, boston_W_star)    
    
        
        test_error_boston = np.mean((boston_Y_test*boston_var - boston_Y_pred*boston_var)**2) # multiply result with variance again
        train_error_boston = np.mean((boston_Y_train*boston_var - boston_Y_train_pred*boston_var)**2)
        
        total_variance_boston = np.var(boston_Y_train*boston_var)
        explain_variance_boston = np.var(boston_Y_train_pred*boston_var)
        r2_score_boston = explain_variance_boston/total_variance_boston
                
        boston_test_error_ridge.append(test_error_boston)
        boston_train_error_ridge.append(train_error_boston)
        boston_r2_score_ridge.append(r2_score_boston)
        
    boston_test_error_ridge_global.append(boston_test_error_ridge)
    boston_train_error_ridge_global.append(boston_train_error_ridge)
    boston_r2_score_ridge_global.append(boston_r2_score_ridge)

########### Ploting the reports ####################
ymin_error = np.min(np.array(boston_test_error_ridge_global+boston_train_error_ridge_global))
ymax_error = np.max(np.array(boston_test_error_ridge_global+boston_train_error_ridge_global))

ymin_r2 = np.min(np.array(boston_r2_score_ridge_global))
ymax_r2 = np.max(np.array(boston_r2_score_ridge_global))

plt.figure(facecolor='0.85')
plt.suptitle('Q4) Boston data Ridge Regression (with $ L_{2} $ regularization) own code', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(boston_test_error_ridge_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(percent_data, boston_train_error_ridge_global[le], 'b')
    plt.plot(percent_data, boston_test_error_ridge_global[le], 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
    plt.ylim(ymin_error - min(ymin_error, 20), ymax_error + min(20, ymax_error*0.1))
    plt.title('Training and test error\n for $ \lambda = $'+str(lamda[le]))
    plt.grid(axis='both')
    
    plt.subplot(fig_count+4)
    plt.plot(percent_data, boston_r2_score_ridge_global[le], 'k')
    plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
    plt.ylim(ymin_r2 - ymin_r2*0.1, ymax_r2 + ymax_r2*0.1)
    plt.title('$R^{2}$ score \n for $ \lambda = $'+str(lamda[le]))
    plt.grid(axis='both')
    

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()

#%% l2 with lamda on X

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
        
        ########## Data Normalizaton and Bias #####################
        boston_shuffle_normal = (boston_shuffle - np.mean(boston_shuffle,axis=0))/np.var(boston_shuffle,axis=0) #np.linalg.norm(boston_shuffle,axis=0)
        boston_shuffle_normal = np.concatenate((np.expand_dims(np.ones(len(boston_shuffle_normal)),1), boston_shuffle_normal),axis=1)
        
        ########### Spliting X and Y again ##################
        boston_X_data_shffle = boston_shuffle_normal[:,:-1]
        boston_Y_data_shffle = boston_shuffle_normal[:,-1]
        
        ########### Data spliting train & test according to 'percent_data' #####
        boston_X_train = boston_X_data_shffle[:m,:]
        boston_X_test = boston_X_data_shffle[m:,:]
        
        boston_Y_train = boston_Y_data_shffle[:m]
        boston_Y_test = boston_Y_data_shffle[m:]
        
        ########## Ridge regresion fitting #############
        boston_XTX_inv_ridge = np.linalg.inv(np.matmul(np.transpose(boston_X_train), boston_X_train)+la*np.eye(boston_X_train.shape[-1]))
        boston_W_star = np.matmul(boston_XTX_inv_ridge, np.matmul(np.transpose(boston_X_train),boston_Y_train))
        
        ########## To scale results #########################
        boston_var = np.var(boston_shuffle[:,-1])
        boston_mean = np.mean(boston_shuffle[:,-1])
        
        ########## Predection and Reporting ################
        boston_Y_pred = np.matmul(boston_X_test, boston_W_star)
        boston_Y_train_pred = np.matmul(boston_X_train, boston_W_star)    
    
        
        test_error_boston = np.mean((boston_Y_test*boston_var - boston_Y_pred*boston_var)**2) # multiply result with variance again
        train_error_boston = np.mean((boston_Y_train*boston_var - boston_Y_train_pred*boston_var)**2)
        
        total_variance_boston = np.var(boston_Y_train*boston_var)
        explain_variance_boston = np.var(boston_Y_train_pred*boston_var)
        r2_score_boston = explain_variance_boston/total_variance_boston
        
        boston_test_error_ridge.append(test_error_boston)
        boston_train_error_ridge.append(train_error_boston)
        boston_r2_score_ridge.append(r2_score_boston)
    
    
    boston_test_error_ridge_global.append(boston_test_error_ridge)
    boston_train_error_ridge_global.append(boston_train_error_ridge)
    boston_r2_score_ridge_global.append(boston_r2_score_ridge)

########### Ploting the reports ####################
ymin_error = np.min(np.array(boston_test_error_ridge_global+boston_train_error_ridge_global))
ymax_error = np.max(np.array(boston_test_error_ridge_global+boston_train_error_ridge_global))

ymin_r2 = np.min(np.array(boston_r2_score_ridge_global))
ymax_r2 = np.max(np.array(boston_r2_score_ridge_global))

plt.figure(facecolor='0.85')
plt.suptitle('Q4) Boston data Ridge Regression (with $ L_{2} $ regularization) own code', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(boston_test_error_ridge_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, boston_train_error_ridge_global[le], 'b')
    plt.plot(lamda, boston_test_error_ridge_global[le], 'r')
    plt.legend(['Training Error', 'Test Error'])
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


#########################################################################################
##########################           Question 5           ###############################
#########################################################################################

percent_data = np.array([50, 60, 70, 80, 90, 95, 99])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

diabetes_test_error = []
diabetes_train_error = []
diabetes_r2_score = []

for m in percent_data:
    ########### Shuffling the data everytime ###############
    diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
    np.random.shuffle(diabetes_shuffle) # shuffel along first axis only
    
    ########### Spliting X and Y again ##################
    diabetes_X_data_shffle = diabetes_shuffle[:,:-1]
    diabetes_Y_data_shffle = diabetes_shuffle[:,-1]
    
    ########### Data spliting train & test according to 'percent_data' #####
    diabetes_X_train = diabetes_X_data_shffle[:m,:]
    diabetes_X_test = diabetes_X_data_shffle[m:,:]
    
    diabetes_Y_train = diabetes_Y_data_shffle[:m]
    diabetes_Y_test = diabetes_Y_data_shffle[m:]
    
    ########## Least square regresion fitting #############
    diabetes_least_reg = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    diabetes_least_reg.fit(diabetes_X_train, diabetes_Y_train)
    
    ########## Predection and Reporting ################
    diabetes_Y_pred = diabetes_least_reg.predict(diabetes_X_test)
    diabetes_Y_train_pred = diabetes_least_reg.predict(diabetes_X_train)
    
    test_error_diabetes = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
    train_error_diabetes = mean_squared_error(diabetes_Y_train, diabetes_Y_train_pred)
    r2_score_diabetes = r2_score(diabetes_Y_train, diabetes_Y_train_pred)
    
    diabetes_test_error.append(test_error_diabetes)
    diabetes_train_error.append(train_error_diabetes)
    diabetes_r2_score.append(r2_score_diabetes)
    
########### Ploting the reports ####################
plt.figure(facecolor='0.85')
plt.suptitle('Q5.1) Diabetes data Least Square Regression (without regularization)', fontsize=16, fontweight='bold')
plt.subplot(121)
plt.plot(percent_data, diabetes_train_error, 'b')
plt.plot(percent_data, diabetes_test_error,'r')
plt.legend(['Training Error', 'Test Error'])
plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
plt.title('Training and test error')
plt.grid(axis='both')

plt.subplot(122)
plt.plot(percent_data, diabetes_r2_score,'k')
plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
plt.title('$R^{2}$ score')
plt.grid(axis='both')
plt.tight_layout()
plt.show()

#%% l2 regression

percent_data = np.array([50, 60, 70, 80, 90, 95, 99])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

lamda = np.array([0, 0.01, 0.1, 1]) # possible to increase number of lambda values

diabetes_test_error_ridge_global = []
diabetes_train_error_ridge_global = []
diabetes_r2_score_ridge_global = []

for la in lamda:
    
    diabetes_test_error_ridge = []
    diabetes_train_error_ridge = []
    diabetes_r2_score_ridge = []
    
    for m in percent_data:
        ########### Shuffling the data everytime ###############
        diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
        np.random.shuffle(diabetes_shuffle) # shuffel along first axis only
        
        ########### Spliting X and Y again ##################
        diabetes_X_data_shffle = diabetes_shuffle[:,:-1]
        diabetes_Y_data_shffle = diabetes_shuffle[:,-1]
        
        ########### Data spliting train & test according to 'percent_data' #####
        diabetes_X_train = diabetes_X_data_shffle[:m,:]
        diabetes_X_test = diabetes_X_data_shffle[m:,:]
        
        diabetes_Y_train = diabetes_Y_data_shffle[:m]
        diabetes_Y_test = diabetes_Y_data_shffle[m:]
        
        ########## Ridge regresion fitting #############
        diabetes_ridge_reg = linear_model.Ridge(alpha=la, fit_intercept=True, normalize=True, solver='auto')
        diabetes_ridge_reg.fit(diabetes_X_train, diabetes_Y_train)
        
        ########## Predection and Reporting ################
        diabetes_Y_pred = diabetes_ridge_reg.predict(diabetes_X_test)
        diabetes_Y_train_pred = diabetes_ridge_reg.predict(diabetes_X_train)
        
        
        test_error_diabetes = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
        train_error_diabetes = mean_squared_error(diabetes_Y_train, diabetes_Y_train_pred)
        r2_score_diabetes = r2_score(diabetes_Y_train, diabetes_Y_train_pred)
        
        diabetes_test_error_ridge.append(test_error_diabetes)
        diabetes_train_error_ridge.append(train_error_diabetes)
        diabetes_r2_score_ridge.append(r2_score_diabetes)
        
    diabetes_test_error_ridge_global.append(diabetes_test_error_ridge)
    diabetes_train_error_ridge_global.append(diabetes_train_error_ridge)
    diabetes_r2_score_ridge_global.append(diabetes_r2_score_ridge)

########### Ploting the reports ####################
ymin_error = np.min(np.array(diabetes_test_error_ridge_global+diabetes_train_error_ridge_global))
ymax_error = np.max(np.array(diabetes_test_error_ridge_global+diabetes_train_error_ridge_global))

ymin_r2 = np.min(np.array(diabetes_r2_score_ridge_global))
ymax_r2 = np.max(np.array(diabetes_r2_score_ridge_global))

plt.figure(facecolor='0.85')
plt.suptitle('Q5.1) Diabetes data Ridge Regression (with $ L_{2} $ regularization)', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(diabetes_test_error_ridge_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(percent_data, diabetes_train_error_ridge_global[le], 'b')
    plt.plot(percent_data, diabetes_test_error_ridge_global[le], 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
    plt.ylim(ymin_error - min(ymin_error, 200), ymax_error + max(200, ymax_error*0.1))
    plt.title('Training and test error\n for $ \lambda = $'+str(lamda[le]))
    plt.grid(axis='both')
    
    plt.subplot(fig_count+4)
    plt.plot(percent_data, diabetes_r2_score_ridge_global[le], 'k')
    plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
    plt.ylim(ymin_r2 - ymin_r2*0.1, ymax_r2 + ymax_r2*0.1)
    plt.title('$R^{2}$ score \n for $ \lambda = $'+str(lamda[le]))
    plt.grid(axis='both')
    

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()

#%%
#########################################################################################
##########################           Question 5.2           #############################
#########################################################################################

percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

lamda = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]) # possible to increase number of lambda values

diabetes_test_error_ridge_global = []
diabetes_train_error_ridge_global = []
diabetes_r2_score_ridge_global = []
diabetes_cv_global = []

for m in percent_data:
    
    diabetes_test_error_ridge = []
    diabetes_train_error_ridge = []
    diabetes_r2_score_ridge = []
    diabetes_cv = []
    
    for la in lamda:
        ########### Shuffling the data everytime ###############
        diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
        np.random.shuffle(diabetes_shuffle) # shuffel along first axis only
        
        ########### Spliting X and Y again ##################
        diabetes_X_data_shffle = diabetes_shuffle[:,:-1]
        diabetes_Y_data_shffle = diabetes_shuffle[:,-1]
        
        ########### Data spliting train & test according to 'percent_data' #####
        diabetes_X_train = diabetes_X_data_shffle[:m,:]
        diabetes_X_test = diabetes_X_data_shffle[m:,:]
        
        diabetes_Y_train = diabetes_Y_data_shffle[:m]
        diabetes_Y_test = diabetes_Y_data_shffle[m:]
        
        ########## Ridge regresion fitting #############
        diabetes_ridge_reg = linear_model.Ridge(alpha=la, fit_intercept=True, normalize=True, solver='auto')
        diabetes_ridge_reg.fit(diabetes_X_train, diabetes_Y_train)
        
        ########## Predection and Reporting ################
        diabetes_Y_pred = diabetes_ridge_reg.predict(diabetes_X_test)
        diabetes_Y_train_pred = diabetes_ridge_reg.predict(diabetes_X_train)
        
        
        test_error_diabetes = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
        train_error_diabetes = mean_squared_error(diabetes_Y_train, diabetes_Y_train_pred)
        r2_score_diabetes = r2_score(diabetes_Y_train, diabetes_Y_train_pred)
        
        ########## Cross validation K = 5 ##################
        cross_val_diabetes = np.abs(np.mean(cross_validate(diabetes_ridge_reg, diabetes_X_train, diabetes_Y_train, cv=5, scoring='neg_mean_squared_error')['test_score']))
        
        diabetes_test_error_ridge.append(test_error_diabetes)
        diabetes_train_error_ridge.append(train_error_diabetes)
        diabetes_r2_score_ridge.append(r2_score_diabetes)
        diabetes_cv.append(cross_val_diabetes)
    
    
    diabetes_test_error_ridge_global.append(diabetes_test_error_ridge)
    diabetes_train_error_ridge_global.append(diabetes_train_error_ridge)
    diabetes_r2_score_ridge_global.append(diabetes_r2_score_ridge)
    diabetes_cv_global.append(diabetes_cv)

########### Ploting the reports ####################
ymin_error = np.min(np.array(diabetes_test_error_ridge_global+diabetes_train_error_ridge_global+diabetes_cv_global))
ymax_error = np.max(np.array(diabetes_test_error_ridge_global+diabetes_train_error_ridge_global+diabetes_cv_global))

ymin_r2 = np.min(np.array(diabetes_r2_score_ridge_global))
ymax_r2 = np.max(np.array(diabetes_r2_score_ridge_global))

plt.figure(facecolor='0.85')
plt.suptitle('Q5.2) Diabetes data Ridge Regression (with $ L_{2} $ regularization)', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(diabetes_test_error_ridge_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, diabetes_train_error_ridge_global[le], 'b')
    plt.plot(lamda, diabetes_test_error_ridge_global[le], 'r')
    plt.plot(lamda, diabetes_cv_global[le], '--r')
    plt.legend(['Training Error', 'Test Error', 'Cross val error'])
    plt.xlabel('$ \lambda $'), plt.ylabel('Mean square error')
    plt.ylim(ymin_error - min(ymin_error, 200), ymax_error + max(200, ymax_error*0.1))
    plt.title('Training and test error\n for '+str(percent_data[le])+' train examples')
    plt.grid(axis='both')
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, diabetes_r2_score_ridge_global[le], 'k')
    plt.xlabel('$ \lambda $'), plt.ylabel('$R^{2}$ Score')
    plt.ylim(ymin_r2 - ymin_r2*0.1, ymax_r2 + ymax_r2*0.1)
    plt.title('$R^{2}$ score \n for '+str(percent_data[le])+' train examples')
    plt.grid(axis='both')
    

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()


#%%
#########################################################################################
##########################           Question 5.3           #############################
#########################################################################################

percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

lamda = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]) # possible to increase number of lambda values

diabetes_test_error_lasso_global = []
diabetes_train_error_lasso_global = []
diabetes_r2_score_lasso_global = []
diabetes_cv_global = []

for m in percent_data:
    
    diabetes_test_error_lasso = []
    diabetes_train_error_lasso = []
    diabetes_r2_score_lasso = []
    diabetes_cv = []
    
    for la in lamda:
        ########### Shuffling the data everytime ###############
        diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
        np.random.shuffle(diabetes_shuffle) # shuffel along first axis only
        
        ########### Spliting X and Y again ##################
        diabetes_X_data_shffle = diabetes_shuffle[:,:-1]
        diabetes_Y_data_shffle = diabetes_shuffle[:,-1]
        
        ########### Data spliting train & test according to 'percent_data' #####
        diabetes_X_train = diabetes_X_data_shffle[:m,:]
        diabetes_X_test = diabetes_X_data_shffle[m:,:]
        
        diabetes_Y_train = diabetes_Y_data_shffle[:m]
        diabetes_Y_test = diabetes_Y_data_shffle[m:]
        
        ########## lasso regresion fitting #############
        diabetes_lasso_reg = linear_model.Lasso(alpha=la, fit_intercept=True, normalize=True, selection='cyclic')
        diabetes_lasso_reg.fit(diabetes_X_train, diabetes_Y_train)
        
        ########## Predection and Reporting ################
        diabetes_Y_pred = diabetes_lasso_reg.predict(diabetes_X_test)
        diabetes_Y_train_pred = diabetes_lasso_reg.predict(diabetes_X_train)
        
        
        test_error_diabetes = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
        train_error_diabetes = mean_squared_error(diabetes_Y_train, diabetes_Y_train_pred)
        r2_score_diabetes = r2_score(diabetes_Y_train, diabetes_Y_train_pred)
        
        ########## Cross validation K = 5 ##################
        cross_val_diabetes = np.abs(np.mean(cross_validate(diabetes_lasso_reg, diabetes_X_train, diabetes_Y_train, cv=5, scoring='neg_mean_squared_error')['test_score']))
        
        diabetes_test_error_lasso.append(test_error_diabetes)
        diabetes_train_error_lasso.append(train_error_diabetes)
        diabetes_r2_score_lasso.append(r2_score_diabetes)
        diabetes_cv.append(cross_val_diabetes)
    
    
    diabetes_test_error_lasso_global.append(diabetes_test_error_lasso)
    diabetes_train_error_lasso_global.append(diabetes_train_error_lasso)
    diabetes_r2_score_lasso_global.append(diabetes_r2_score_lasso)
    diabetes_cv_global.append(diabetes_cv)

########### Ploting the reports ####################
ymin_error = np.min(np.array(diabetes_test_error_lasso_global+diabetes_train_error_lasso_global+diabetes_cv_global))
ymax_error = np.max(np.array(diabetes_test_error_lasso_global+diabetes_train_error_lasso_global+diabetes_cv_global))

ymin_r2 = np.min(np.array(diabetes_r2_score_lasso_global))
ymax_r2 = np.max(np.array(diabetes_r2_score_lasso_global))

plt.figure(facecolor='0.85')
plt.suptitle('Q5.3) Diabetes data Lasso Regression (with $ L_{1} $ regularization)', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(diabetes_test_error_lasso_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, diabetes_train_error_lasso_global[le], 'b')
    plt.plot(lamda, diabetes_test_error_lasso_global[le], 'r')
    plt.plot(lamda, diabetes_cv_global[le], '--r')
    plt.legend(['Training Error', 'Test Error', 'Cross val error'])
    plt.xlabel('$ \lambda $'), plt.ylabel('Mean square error')
    plt.ylim(ymin_error - min(ymin_error, 200), ymax_error + max(200, ymax_error*0.1))
    
    plt.title('Training and test error\n for '+str(percent_data[le])+' train examples')
    plt.grid(axis='both')
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, diabetes_r2_score_lasso_global[le], 'k')
    plt.xlabel('$ \lambda $'), plt.ylabel('$R^{2}$ Score')
    plt.ylim(ymin_r2 - max(0.1,ymin_r2*0.1), ymax_r2 + ymax_r2*0.1)
    plt.title('$R^{2}$ score \n for '+str(percent_data[le])+' train examples')
    plt.grid(axis='both')
    

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()


#%%
#########################################################################################
##########################           Question 5.4           ###############################
#########################################################################################

percent_data = np.array([50, 60, 70, 80, 90, 95, 99])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

diabetes_test_error = []
diabetes_train_error = []
diabetes_r2_score = []

for m in percent_data:
    ########### Shuffling the data everytime ###############
    diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
    np.random.shuffle(diabetes_shuffle) # shuffel along first axis only
    
    ########## Data Normalizaton and Bias #####################
    diabetes_shuffle_normal = (diabetes_shuffle - np.mean(diabetes_shuffle,axis=0))/np.var(diabetes_shuffle,axis=0) #np.linalg.norm(diabetes_shuffle,axis=0)
    diabetes_shuffle_normal = np.concatenate((np.expand_dims(np.ones(len(diabetes_shuffle_normal)),1), diabetes_shuffle_normal),axis=1)
    
    ########### Spliting X and Y again ##################
    diabetes_X_data_shffle = diabetes_shuffle_normal[:,:-1]
    diabetes_Y_data_shffle = diabetes_shuffle_normal[:,-1]
    
    ########### Data spliting train & test according to 'percent_data' #####
    diabetes_X_train = diabetes_X_data_shffle[:m,:]
    diabetes_X_test = diabetes_X_data_shffle[m:,:]
    
    diabetes_Y_train = diabetes_Y_data_shffle[:m]
    diabetes_Y_test = diabetes_Y_data_shffle[m:]
    
    ########## Least square regresion fitting #############
    diabetes_XTX_inv = np.linalg.inv(np.matmul(np.transpose(diabetes_X_train), diabetes_X_train))
    diabetes_W_star = np.matmul(diabetes_XTX_inv, np.matmul(np.transpose(diabetes_X_train),diabetes_Y_train))
    
    ########## To scale results #########################
    diabetes_var = np.var(diabetes_shuffle[:,-1])
    diabetes_mean = np.mean(diabetes_shuffle[:,-1])
    
    ########## Predection and Reporting ################
    diabetes_Y_pred = np.matmul(diabetes_X_test, diabetes_W_star)
    diabetes_Y_train_pred = np.matmul(diabetes_X_train, diabetes_W_star)    
    
    test_error_diabetes = np.mean((diabetes_Y_test*diabetes_var - diabetes_Y_pred*diabetes_var)**2) # multiply result with variance again
    train_error_diabetes = np.mean((diabetes_Y_train*diabetes_var - diabetes_Y_train_pred*diabetes_var)**2)
    
    total_variance_diabetes = np.var(diabetes_Y_train*diabetes_var)
    explain_variance_diabetes = np.var(diabetes_Y_train_pred*diabetes_var)
    r2_score_diabetes = explain_variance_diabetes/total_variance_diabetes
    
    diabetes_test_error.append(test_error_diabetes) 
    diabetes_train_error.append(train_error_diabetes) 
    diabetes_r2_score.append(r2_score_diabetes)

    
########### Ploting the reports ####################
plt.figure(facecolor='0.85')
plt.suptitle('Q5.4) Diabetes data Least Square Regression (without regularization) own code', fontsize=16, fontweight='bold')
plt.subplot(121)
plt.plot(percent_data, diabetes_train_error, 'b')
plt.plot(percent_data, diabetes_test_error,'r')
plt.legend(['Training Error', 'Test Error'])
plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
plt.title('Training and test error')
plt.grid(axis='both')

plt.subplot(122)
plt.plot(percent_data, diabetes_r2_score,'k')
plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
plt.title('$R^{2}$ score')
plt.grid(axis='both')
plt.tight_layout()
plt.show()

#%% l2 regression

percent_data = np.array([50, 60, 70, 80, 90, 95, 99])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

lamda = np.array([0, 0.01, 0.1, 1]) # possible to increase number of lambda values

diabetes_test_error_ridge_global = []
diabetes_train_error_ridge_global = []
diabetes_r2_score_ridge_global = []

for la in lamda:
    
    diabetes_test_error_ridge = []
    diabetes_train_error_ridge = []
    diabetes_r2_score_ridge = []
    
    for m in percent_data:
        ########### Shuffling the data everytime ###############
        diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
        np.random.shuffle(diabetes_shuffle) # shuffel along first axis only
        
        ########## Data Normalizaton and Bias #####################
        diabetes_shuffle_normal = (diabetes_shuffle - np.mean(diabetes_shuffle,axis=0))/np.var(diabetes_shuffle,axis=0) #np.linalg.norm(diabetes_shuffle,axis=0)
        diabetes_shuffle_normal = np.concatenate((np.expand_dims(np.ones(len(diabetes_shuffle_normal)),1), diabetes_shuffle_normal),axis=1)
        
        ########### Spliting X and Y again ##################
        diabetes_X_data_shffle = diabetes_shuffle_normal[:,:-1]
        diabetes_Y_data_shffle = diabetes_shuffle_normal[:,-1]
        
        ########### Data spliting train & test according to 'percent_data' #####
        diabetes_X_train = diabetes_X_data_shffle[:m,:]
        diabetes_X_test = diabetes_X_data_shffle[m:,:]
        
        diabetes_Y_train = diabetes_Y_data_shffle[:m]
        diabetes_Y_test = diabetes_Y_data_shffle[m:]
        
        ########## Ridge regresion fitting #############
        diabetes_XTX_inv_ridge = np.linalg.inv(np.matmul(np.transpose(diabetes_X_train), diabetes_X_train)+la*np.eye(diabetes_X_train.shape[-1]))
        diabetes_W_star = np.matmul(diabetes_XTX_inv_ridge, np.matmul(np.transpose(diabetes_X_train),diabetes_Y_train))
        
        ########## To scale results #########################
        diabetes_var = np.var(diabetes_shuffle[:,-1])
        diabetes_mean = np.mean(diabetes_shuffle[:,-1])
        
        ########## Predection and Reporting ################
        diabetes_Y_pred = np.matmul(diabetes_X_test, diabetes_W_star)
        diabetes_Y_train_pred = np.matmul(diabetes_X_train, diabetes_W_star)    
    
        
        test_error_diabetes = np.mean((diabetes_Y_test*diabetes_var - diabetes_Y_pred*diabetes_var)**2) # multiply result with variance again
        train_error_diabetes = np.mean((diabetes_Y_train*diabetes_var - diabetes_Y_train_pred*diabetes_var)**2)
        
        total_variance_diabetes = np.var(diabetes_Y_train*diabetes_var)
        explain_variance_diabetes = np.var(diabetes_Y_train_pred*diabetes_var)
        r2_score_diabetes = explain_variance_diabetes/total_variance_diabetes
                
        diabetes_test_error_ridge.append(test_error_diabetes)
        diabetes_train_error_ridge.append(train_error_diabetes)
        diabetes_r2_score_ridge.append(r2_score_diabetes)
        
    diabetes_test_error_ridge_global.append(diabetes_test_error_ridge)
    diabetes_train_error_ridge_global.append(diabetes_train_error_ridge)
    diabetes_r2_score_ridge_global.append(diabetes_r2_score_ridge)

########### Ploting the reports ####################
ymin_error = np.min(np.array(diabetes_test_error_ridge_global+diabetes_train_error_ridge_global))
ymax_error = np.max(np.array(diabetes_test_error_ridge_global+diabetes_train_error_ridge_global))

ymin_r2 = np.min(np.array(diabetes_r2_score_ridge_global))
ymax_r2 = np.max(np.array(diabetes_r2_score_ridge_global))

plt.figure(facecolor='0.85')
plt.suptitle('Q5.4) Diabetes data Ridge Regression (with $ L_{2} $ regularization) own code', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(diabetes_test_error_ridge_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(percent_data, diabetes_train_error_ridge_global[le], 'b')
    plt.plot(percent_data, diabetes_test_error_ridge_global[le], 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
    plt.ylim(ymin_error - min(ymin_error, 200), ymax_error + max(20, ymax_error*0.1))
    plt.title('Training and test error\n for $ \lambda = $'+str(lamda[le]))
    plt.grid(axis='both')
    
    plt.subplot(fig_count+4)
    plt.plot(percent_data, diabetes_r2_score_ridge_global[le], 'k')
    plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
    plt.ylim(ymin_r2 - ymin_r2*0.1, ymax_r2 + ymax_r2*0.1)
    plt.title('$R^{2}$ score \n for $ \lambda = $'+str(lamda[le]))
    plt.grid(axis='both')
    

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()

#%% l2 with lambda on X

percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

lamda = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2, 3, 4, 5]) # possible to increase number of lambda values

diabetes_test_error_ridge_global = []
diabetes_train_error_ridge_global = []
diabetes_r2_score_ridge_global = []
diabetes_cv_global = []

for m in percent_data:
    
    diabetes_test_error_ridge = []
    diabetes_train_error_ridge = []
    diabetes_r2_score_ridge = []
    diabetes_cv = []
    
    for la in lamda:
        ########### Shuffling the data everytime ###############
        diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
        np.random.shuffle(diabetes_shuffle) # shuffel along first axis only
        
        ########## Data Normalizaton and Bias #####################
        diabetes_shuffle_normal = (diabetes_shuffle - np.mean(diabetes_shuffle,axis=0))/np.var(diabetes_shuffle,axis=0) #np.linalg.norm(diabetes_shuffle,axis=0)
        diabetes_shuffle_normal = np.concatenate((np.expand_dims(np.ones(len(diabetes_shuffle_normal)),1), diabetes_shuffle_normal),axis=1)
        
        ########### Spliting X and Y again ##################
        diabetes_X_data_shffle = diabetes_shuffle_normal[:,:-1]
        diabetes_Y_data_shffle = diabetes_shuffle_normal[:,-1]
        
        ########### Data spliting train & test according to 'percent_data' #####
        diabetes_X_train = diabetes_X_data_shffle[:m,:]
        diabetes_X_test = diabetes_X_data_shffle[m:,:]
        
        diabetes_Y_train = diabetes_Y_data_shffle[:m]
        diabetes_Y_test = diabetes_Y_data_shffle[m:]
        
        ########## Ridge regresion fitting #############
        diabetes_XTX_inv_ridge = np.linalg.inv(np.matmul(np.transpose(diabetes_X_train), diabetes_X_train)+la*np.eye(diabetes_X_train.shape[-1]))
        diabetes_W_star = np.matmul(diabetes_XTX_inv_ridge, np.matmul(np.transpose(diabetes_X_train),diabetes_Y_train))
        
        ########## To scale results #########################
        diabetes_var = np.var(diabetes_shuffle[:,-1])
        diabetes_mean = np.mean(diabetes_shuffle[:,-1])
        
        ########## Predection and Reporting ################
        diabetes_Y_pred = np.matmul(diabetes_X_test, diabetes_W_star)
        diabetes_Y_train_pred = np.matmul(diabetes_X_train, diabetes_W_star)    
    
        
        test_error_diabetes = np.mean((diabetes_Y_test*diabetes_var - diabetes_Y_pred*diabetes_var)**2) # multiply result with variance again
        train_error_diabetes = np.mean((diabetes_Y_train*diabetes_var - diabetes_Y_train_pred*diabetes_var)**2)
        
        total_variance_diabetes = np.var(diabetes_Y_train*diabetes_var)
        explain_variance_diabetes = np.var(diabetes_Y_train_pred*diabetes_var)
        r2_score_diabetes = explain_variance_diabetes/total_variance_diabetes
        
        diabetes_test_error_ridge.append(test_error_diabetes)
        diabetes_train_error_ridge.append(train_error_diabetes)
        diabetes_r2_score_ridge.append(r2_score_diabetes)
    
    
    diabetes_test_error_ridge_global.append(diabetes_test_error_ridge)
    diabetes_train_error_ridge_global.append(diabetes_train_error_ridge)
    diabetes_r2_score_ridge_global.append(diabetes_r2_score_ridge)

########### Ploting the reports ####################
ymin_error = np.min(np.array(diabetes_test_error_ridge_global+diabetes_train_error_ridge_global))
ymax_error = np.max(np.array(diabetes_test_error_ridge_global+diabetes_train_error_ridge_global))

ymin_r2 = np.min(np.array(diabetes_r2_score_ridge_global))
ymax_r2 = np.max(np.array(diabetes_r2_score_ridge_global))

plt.figure(facecolor='0.85')
plt.suptitle('Q5.4) Diabetes data Ridge Regression (with $ L_{2} $ regularization) own code', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(diabetes_test_error_ridge_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, diabetes_train_error_ridge_global[le], 'b')
    plt.plot(lamda, diabetes_test_error_ridge_global[le], 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('$ \lambda $'), plt.ylabel('Mean square error')
    plt.ylim(ymin_error - min(ymin_error, 200), ymax_error + max(200, ymax_error*0.1))
    plt.title('Training and test error\n for '+str(percent_data[le])+' train examples')
    plt.grid(axis='both')
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, diabetes_r2_score_ridge_global[le], 'k')
    plt.xlabel('$ \lambda $'), plt.ylabel('$R^{2}$ Score')
    plt.ylim(ymin_r2 - ymin_r2*0.1, ymax_r2 + ymax_r2*0.1)
    plt.title('$R^{2}$ score \n for '+str(percent_data[le])+' train examples')
    plt.grid(axis='both')
    

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()
