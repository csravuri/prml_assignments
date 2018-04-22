
####################  Required Libraries  ###################
import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#################### Loading Data    #################### (diabetes)
diabetes = datasets.load_diabetes()
diabetes_X_data = diabetes['data']
diabetes_Y_data = diabetes['target']

#1
# shuffle it dont forget!!!
########### Shuffling the data everytime ###############
#diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
#diabetes_shuffle = np.random.permutation(diabetes_shuffle) # shuffel along first axis only
#
############ Spliting X and Y again ##################
#diabetes_X_data = diabetes_shuffle[:,:-1]
#diabetes_Y_data = diabetes_shuffle[:,-1]

#percent_data = np.array([70,80,90,99],np.int)*len(diabetes_X_data)/100
#percent_data = np.array(range(50,99,5))

percent_data = np.array([70, 80, 90, 99])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

diabetes_test_error = []
diabetes_train_error = []
diabetes_r2_score = []

for m in percent_data:
    ########### Shuffling the data everytime ###############
    diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
#    np.random.shuffle(diabetes_shuffle) # shuffel along first axis only
    
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
    
    
plt.figure()
plt.subplot(121)
#plt.grid(b=True, axis='both', color='y', linestyle='-.')
plt.plot(percent_data, diabetes_train_error, 'b')
plt.plot(percent_data, diabetes_test_error,'r')
plt.legend(['Training Error', 'Test Error'])
plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
plt.title('Training and test error \n for diabetes housing dataset')

plt.subplot(122)
#plt.grid(b=True, axis='both', color='y', linestyle='-.')
plt.plot(percent_data, diabetes_r2_score,'k')
plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
plt.title('$R^{2}$ score \n for diabetes housing dataset')
plt.tight_layout()
plt.show()

#1 End

#print(diabetes_train_error)
#2 start

percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

lamda = range(5) # possible to increase number of lambda values
plt.figure()
fig_count=240
for m in percent_data:
    
    diabetes_test_error_ridge = []
    diabetes_train_error_ridge = []
    diabetes_r2_score_ridge = []
    
    for la in lamda:
        ########### Shuffling the data everytime ###############
        diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
#        np.random.shuffle(diabetes_shuffle) # shuffel along first axis only
        
        ########### Spliting X and Y again ##################
        diabetes_X_data_shffle = diabetes_shuffle[:,:-1]
        diabetes_Y_data_shffle = diabetes_shuffle[:,-1]
        
        ########### Data spliting train & test according to 'percent_data' #####
        diabetes_X_train = diabetes_X_data_shffle[:m,:]
        diabetes_X_test = diabetes_X_data_shffle[m:,:]
        
        diabetes_Y_train = diabetes_Y_data_shffle[:m]
        diabetes_Y_test = diabetes_Y_data_shffle[m:]
        
        ########## Ridge regresion fitting #############
        diabetes_ridge_reg = linear_model.Ridge(alpha=la, fit_intercept=True, normalize=False, solver='auto')
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
        
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, diabetes_train_error_ridge, 'b')
    plt.plot(lamda, diabetes_test_error_ridge, 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('$ \lambda $'), plt.ylabel('Mean square error')
    plt.title('Training and test error\n for '+str(m)+' train examples')
    
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, diabetes_r2_score_ridge, 'k')
    plt.xlabel('$ \lambda $'), plt.ylabel('$R^{2}$ Score')
    plt.title('$R^{2}$ score \n for '+str(m)+' train examples')

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()
        
#2 End

#3 start

percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

lamda = range(5) # possible to increase number of lambda values
plt.figure()
fig_count=240
for m in percent_data:
    
    diabetes_test_error_lasso = []
    diabetes_train_error_lasso = []
    diabetes_r2_score_lasso = []
    
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
        
        ########## Least square regresion fitting #############
        diabetes_lasso_reg = linear_model.Lasso(alpha=la+0.1, fit_intercept=True, normalize=False, selection='cyclic')
        diabetes_lasso_reg.fit(diabetes_X_train, diabetes_Y_train)
        
        ########## Predection and Reporting ################
        diabetes_Y_pred = diabetes_lasso_reg.predict(diabetes_X_test)
        diabetes_Y_train_pred = diabetes_lasso_reg.predict(diabetes_X_train)
        
        
        test_error_diabetes = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
        train_error_diabetes = mean_squared_error(diabetes_Y_train, diabetes_Y_train_pred)
        r2_score_diabetes = r2_score(diabetes_Y_train, diabetes_Y_train_pred)
        
        diabetes_test_error_lasso.append(test_error_diabetes)
        diabetes_train_error_lasso.append(train_error_diabetes)
        diabetes_r2_score_lasso.append(r2_score_diabetes)
        
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, diabetes_train_error_lasso, 'b')
    plt.plot(lamda, diabetes_test_error_lasso, 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('$ \lambda $'), plt.ylabel('Mean square error')
    plt.title('Training and test error\n for '+str(m)+' train examples')
    
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, diabetes_r2_score_lasso, 'k')
    plt.xlabel('$ \lambda $'), plt.ylabel('$R^{2}$ Score')
    plt.title('$R^{2}$ score \n for '+str(m)+' train examples')

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()


 # 3 end

# 4 start

#plt.close('all')


percent_data = np.array([70, 80, 90, 99])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

diabetes_test_error = []
diabetes_train_error = []
diabetes_r2_score = []

for m in percent_data:
    ########### Shuffling the data everytime ###############
    diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
#    np.random.shuffle(diabetes_shuffle) # shuffel along first axis only
    
    ########## Data Normalizaton and Bias #####################
    diabetes_shuffle_normal = (diabetes_shuffle - np.mean(diabetes_shuffle,axis=0))/np.linalg.norm(diabetes_shuffle,axis=0)
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
#    diabetes_least_reg = linear_model.LinearRegression(fit_intercept=True, normalize=False)
#    diabetes_least_reg.fit(diabetes_X_train, diabetes_Y_train)
    diabetes_XTX_inv = np.linalg.inv(np.matmul(np.transpose(diabetes_X_train), diabetes_X_train))
    diabetes_W_star = np.matmul(diabetes_XTX_inv, np.matmul(np.transpose(diabetes_X_train),diabetes_Y_train))
    
    
    ########## Predection and Reporting ################
#    diabetes_Y_pred = diabetes_least_reg.predict(diabetes_X_test)
#    diabetes_Y_train_pred = diabetes_least_reg.predict(diabetes_X_train)
    diabetes_Y_pred = np.matmul(diabetes_X_test, diabetes_W_star)
    diabetes_Y_train_pred = np.matmul(diabetes_X_train, diabetes_W_star)    
    
    test_error_diabetes = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
    train_error_diabetes = mean_squared_error(diabetes_Y_train, diabetes_Y_train_pred)
    r2_score_diabetes = r2_score(diabetes_Y_train, diabetes_Y_train_pred)
    
    diabetes_test_error.append(test_error_diabetes) #*np.linalg.norm(diabetes_shuffle,axis=0)[-1]+ np.mean(diabetes_shuffle,axis=0)[-1])
    diabetes_train_error.append(train_error_diabetes) #*np.linalg.norm(diabetes_shuffle,axis=0)[-1] + np.mean(diabetes_shuffle,axis=0)[-1])
    diabetes_r2_score.append(r2_score_diabetes)
    
    
plt.figure()
plt.subplot(121)
plt.plot(percent_data, diabetes_train_error, 'b')
plt.plot(percent_data, diabetes_test_error,'r')
plt.legend(['Training Error', 'Test Error'])
plt.xlabel('Number of training examples'), plt.ylabel('Mean square error')
plt.title('Training and test error \n for diabetes housing dataset')

plt.subplot(122)
plt.plot(percent_data, diabetes_r2_score,'k')
plt.xlabel('Number of training examples'), plt.ylabel('$R^{2}$ Score')
plt.title('$R^{2}$ score \n for diabetes housing dataset')
plt.tight_layout()
plt.show()


print(diabetes_train_error)
        
# something is wrong !!!
#np.array([8.9953825029971721, 22.778379521800787, 23.236357264706726, 21.821129388812526])/np.array([3.002200174723348e-05, 7.6022620447190314e-05, 7.7551116716596854e-05, 7.2827807424449186e-05])


percent_data = np.array([99, 90, 80, 70])
percent_data = np.array(percent_data*len(diabetes_X_data)/100, np.int)

lamda = range(5) # possible to increase number of lambda values
plt.figure()
fig_count=240
for m in percent_data:
    
    diabetes_test_error_ridge = []
    diabetes_train_error_ridge = []
    diabetes_r2_score_ridge = []
    
    for la in lamda:
        ########### Shuffling the data everytime ###############
        diabetes_shuffle = np.concatenate((diabetes_X_data, np.expand_dims(diabetes_Y_data,1)),axis=1)
#        np.random.shuffle(diabetes_shuffle) # shuffel along first axis only
        
        ########## Data Normalizaton and Bias #####################
        diabetes_shuffle_normal = (diabetes_shuffle - np.mean(diabetes_shuffle,axis=0))/np.linalg.norm(diabetes_shuffle,axis=0)
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
#        diabetes_ridge_reg = linear_model.Ridge(alpha=la, fit_intercept=True, normalize=False, solver='auto')
#        diabetes_ridge_reg.fit(diabetes_X_train, diabetes_Y_train)
        diabetes_XTX_inv_ridge = np.linalg.inv(np.matmul(np.transpose(diabetes_X_train), diabetes_X_train)+la*np.eye(diabetes_X_train.shape[-1]))
        diabetes_W_star = np.matmul(diabetes_XTX_inv_ridge, np.matmul(np.transpose(diabetes_X_train),diabetes_Y_train))
        
        ########## Predection and Reporting ################
#        diabetes_Y_pred = diabetes_ridge_reg.predict(diabetes_X_test)
#        diabetes_Y_train_pred = diabetes_ridge_reg.predict(diabetes_X_train)
        diabetes_Y_pred = np.matmul(diabetes_X_test, diabetes_W_star)
        diabetes_Y_train_pred = np.matmul(diabetes_X_train, diabetes_W_star)    
    
        
        test_error_diabetes = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
        train_error_diabetes = mean_squared_error(diabetes_Y_train, diabetes_Y_train_pred)
        r2_score_diabetes = r2_score(diabetes_Y_train, diabetes_Y_train_pred)
        
        diabetes_test_error_ridge.append(test_error_diabetes)
        diabetes_train_error_ridge.append(train_error_diabetes)
        diabetes_r2_score_ridge.append(r2_score_diabetes)
        
    fig_count+=1
    plt.subplot(fig_count)
    plt.plot(lamda, diabetes_train_error_ridge, 'b')
    plt.plot(lamda, diabetes_test_error_ridge, 'r')
    plt.legend(['Training Error', 'Test Error'])
    plt.xlabel('$ \lambda $'), plt.ylabel('Mean square error')
    plt.title('Training and test error\n for '+str(m)+' train examples')
    
    
    plt.subplot(fig_count+4)
    plt.plot(lamda, diabetes_r2_score_ridge, 'k')
    plt.xlabel('$ \lambda $'), plt.ylabel('$R^{2}$ Score')
    plt.title('$R^{2}$ score \n for '+str(m)+' train examples')

plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.show()

