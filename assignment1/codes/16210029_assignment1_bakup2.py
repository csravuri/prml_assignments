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
#    np.random.shuffle(boston_shuffle) # shuffel along first axis only
    
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
plt.suptitle('Boston data Least Square Regression (without regularization)', fontsize=16, fontweight='bold')
plt.subplot(121)
plt.plot(percent_data, boston_train_error, 'b')
plt.plot(percent_data, boston_test_error,'r')
plt.legend(['Training Error', 'Test Error'])
plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
plt.title('Training and test error')

plt.subplot(122)
plt.plot(percent_data, boston_r2_score,'k')
plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
#plt.xlim(300,510), plt.ylim(0,1)
plt.title('$R^{2}$ score')
plt.tight_layout()
plt.show()

#%% l2 reg

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
#        np.random.shuffle(boston_shuffle) # shuffel along first axis only
        
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
plt.suptitle('Boston data Ridge Regression (with $ L_{2} $ regularization)', fontsize=16, fontweight='bold')
fig_count=240

for le in range(len(boston_test_error_ridge_global)):
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(percent_data, boston_train_error_ridge_global[le], 'b')
    plt.plot(percent_data, boston_test_error_ridge_global[le], 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
    plt.ylim(ymin_error - min(ymin_error, 20), ymax_error + min(20, ymax_error*0.1))
    plt.title('Training and test error\n for $ \lambda = $'+str(la))
    plt.grid(axis='both')
    
    plt.subplot(fig_count+4)
    plt.plot(percent_data, boston_r2_score_ridge_global[le], 'k')
    plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
    plt.ylim(ymin_r2 - ymin_r2*0.1, ymax_r2 + ymax_r2*0.1)
    plt.title('$R^{2}$ score \n for $ \lambda = $'+str(la))
    plt.grid(axis='both')
    

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()




#%%
#########################################################################################
##########################           Question 2           ###############################
#########################################################################################

percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(boston_X_data)/100, np.int)

lamda = range(6) # possible to increase number of lambda values

plt.figure(facecolor='0.85')
plt.suptitle('Boston data Ridge Regression (with $ L_{2} $ regularization)', fontsize=16, fontweight='bold')
fig_count=240
for m in percent_data:
    
    boston_test_error_ridge = []
    boston_train_error_ridge = []
    boston_r2_score_ridge = []
    
    for la in lamda:
        ########### Shuffling the data everytime ###############
        boston_shuffle = np.concatenate((boston_X_data, np.expand_dims(boston_Y_data,1)),axis=1)
#        np.random.shuffle(boston_shuffle) # shuffel along first axis only
        
        ########### Spliting X and Y again ##################
        boston_X_data_shffle = boston_shuffle[:,:-1]
        boston_Y_data_shffle = boston_shuffle[:,-1]
        
        ########### Data spliting train & test according to 'percent_data' #####
        boston_X_train = boston_X_data_shffle[:m,:]
        boston_X_test = boston_X_data_shffle[m:,:]
        
        boston_Y_train = boston_Y_data_shffle[:m]
        boston_Y_test = boston_Y_data_shffle[m:]
        
        ########## Ridge regresion fitting #############
        boston_ridge_reg = linear_model.Ridge(alpha=la, fit_intercept=True, normalize=False, solver='auto')
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
        
    ########### Ploting the reports ####################
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, boston_train_error_ridge, 'b')
    plt.plot(lamda, boston_test_error_ridge, 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('$ \lambda $'), plt.ylabel('Mean square error')
    plt.title('Training and test error\n for '+str(m)+' train examples')
    
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, boston_r2_score_ridge, 'k')
    plt.xlabel('$ \lambda $'), plt.ylabel('$R^{2}$ Score')
    plt.title('$R^{2}$ score \n for '+str(m)+' train examples')

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()

#%%
#########################################################################################
##########################           Question 3           ###############################
#########################################################################################

percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(boston_X_data)/100, np.int)

lamda = range(5) # possible to increase number of lambda values
plt.figure(facecolor='0.85')
plt.suptitle('Boston data Lasso Regression (with $ L_{1} $ regularization)', fontsize=16, fontweight='bold')
fig_count=240
for m in percent_data:
    
    boston_test_error_lasso = []
    boston_train_error_lasso = []
    boston_r2_score_lasso = []
    
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
        
        ########## Least square regresion fitting #############
        boston_lasso_reg = linear_model.Lasso(alpha=la+0.1, fit_intercept=True, normalize=False, selection='cyclic')
        boston_lasso_reg.fit(boston_X_train, boston_Y_train)
        
        ########## Predection and Reporting ################
        boston_Y_pred = boston_lasso_reg.predict(boston_X_test)
        boston_Y_train_pred = boston_lasso_reg.predict(boston_X_train)
        
        
        test_error_boston = mean_squared_error(boston_Y_test, boston_Y_pred)
        train_error_boston = mean_squared_error(boston_Y_train, boston_Y_train_pred)
        r2_score_boston = r2_score(boston_Y_train, boston_Y_train_pred)
        
        boston_test_error_lasso.append(test_error_boston)
        boston_train_error_lasso.append(train_error_boston)
        boston_r2_score_lasso.append(r2_score_boston)
    
    ########### Ploting the reports ####################
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, boston_train_error_lasso, 'b')
    plt.plot(lamda, boston_test_error_lasso, 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('$ \lambda $'), plt.ylabel('Mean square error')
    plt.title('Training and test error\n for '+str(m)+' train examples')
    
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, boston_r2_score_lasso, 'k')
    plt.xlabel('$ \lambda $'), plt.ylabel('$R^{2}$ Score')
    plt.title('$R^{2}$ score \n for '+str(m)+' train examples')

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()

#%%
#########################################################################################
##########################           Question 4           ###############################
#########################################################################################


percent_data = np.array([70, 80, 90, 99])
percent_data = np.array(percent_data*len(boston_X_data)/100, np.int)

boston_test_error = []
boston_train_error = []
boston_r2_score = []

for m in percent_data:
    ########### Shuffling the data everytime ###############
    boston_shuffle = np.concatenate((boston_X_data, np.expand_dims(boston_Y_data,1)),axis=1)
#    np.random.shuffle(boston_shuffle) # shuffel along first axis only
    
    ########## Data Normalizaton and Bias #####################
    boston_shuffle_normal = (boston_shuffle - np.mean(boston_shuffle,axis=0))/np.linalg.norm(boston_shuffle,axis=0)
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
    
    ########## Predection and Reporting ################
    boston_Y_pred = np.matmul(boston_X_test, boston_W_star)
    boston_Y_train_pred = np.matmul(boston_X_train, boston_W_star)    
    
    test_error_boston = mean_squared_error(boston_Y_test, boston_Y_pred)
    train_error_boston = mean_squared_error(boston_Y_train, boston_Y_train_pred)
    
    total_variance_boston = np.var(boston_Y_train)
    explain_variance_boston = np.var(boston_Y_train_pred)
    r2_score_boston = explain_variance_boston/total_variance_boston
    
    boston_test_error.append(test_error_boston) #*np.linalg.norm(boston_shuffle,axis=0)[-1]+ np.mean(boston_shuffle,axis=0)[-1])
    boston_train_error.append(train_error_boston) #*np.linalg.norm(boston_shuffle,axis=0)[-1] + np.mean(boston_shuffle,axis=0)[-1])
    boston_r2_score.append(r2_score_boston)
    
########### Ploting the reports ####################
plt.figure(facecolor='0.85')
plt.suptitle('Boston data Least Square Regression (without regularization) own code', fontsize=16, fontweight='bold')
plt.subplot(121)
plt.plot(percent_data, boston_train_error, 'b')
plt.plot(percent_data, boston_test_error,'r')
plt.legend(['Training Error', 'Test Error'])
plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
plt.title('Training and test error \n for Boston housing dataset')

plt.subplot(122)
plt.plot(percent_data, boston_r2_score,'k')
plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
plt.title('$R^{2}$ score \n for Boston housing dataset')
plt.tight_layout()
plt.show()


        
# something is wrong !!!
#np.array([8.9953825029971721, 22.778379521800787, 23.236357264706726, 21.821129388812526])/np.array([3.002200174723348e-05, 7.6022620447190314e-05, 7.7551116716596854e-05, 7.2827807424449186e-05])


percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(boston_X_data)/100, np.int)

lamda = range(5) # possible to increase number of lambda values
plt.figure(facecolor='0.85')
plt.suptitle('Boston data Ridge Regression (with $ L_{2} $ regularization) own code', fontsize=16, fontweight='bold')
fig_count=240
for m in percent_data:
    
    boston_test_error_ridge = []
    boston_train_error_ridge = []
    boston_r2_score_ridge = []
    
    for la in lamda:
        ########### Shuffling the data everytime ###############
        boston_shuffle = np.concatenate((boston_X_data, np.expand_dims(boston_Y_data,1)),axis=1)
#        np.random.shuffle(boston_shuffle) # shuffel along first axis only
        
        ########## Data Normalizaton and Bias #####################
        boston_shuffle_normal = (boston_shuffle - np.mean(boston_shuffle,axis=0))/np.linalg.norm(boston_shuffle,axis=0)
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
        
        ########## Predection and Reporting ################
        boston_Y_pred = np.matmul(boston_X_test, boston_W_star)
        boston_Y_train_pred = np.matmul(boston_X_train, boston_W_star)    
    
        
        test_error_boston = mean_squared_error(boston_Y_test, boston_Y_pred)
        train_error_boston = mean_squared_error(boston_Y_train, boston_Y_train_pred)
        
        total_variance_boston = np.var(boston_Y_train)
        explain_variance_boston = np.var(boston_Y_train_pred)
        r2_score_boston = explain_variance_boston/total_variance_boston
        
        boston_test_error_ridge.append(test_error_boston)
        boston_train_error_ridge.append(train_error_boston)
        boston_r2_score_ridge.append(r2_score_boston)
        
    ########### Ploting the reports ####################
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, boston_train_error_ridge, 'b')
    plt.plot(lamda, boston_test_error_ridge, 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('$ \lambda $'), plt.ylabel('Mean square error')
    plt.title('Training and test error\n for '+str(m)+' train examples')
    
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, boston_r2_score_ridge, 'k')
    plt.xlabel('$ \lambda $'), plt.ylabel('$R^{2}$ Score')
    plt.title('$R^{2}$ score \n for '+str(m)+' train examples')

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()


#########################################################################################
##########################           Question 5           ###############################
#########################################################################################


