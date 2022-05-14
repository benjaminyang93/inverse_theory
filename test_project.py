#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:39:07 2022

@author: benjaminyang

Geophysical Inverse Theory Project
"""

##############################
#####   Import Modules   #####
##############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor

##############################################
#####   Read and Filter Synthetic Data   #####
##############################################

# Data directory
base_dir = '/Users/benjaminyang/Documents/Inverse Theory/Project/'

# Read in data 
df = pd.DataFrame()
f_in = ['in01','in02','in03','in04']
f_out = ['out01','out02','out03','out04']
files = f_in + f_out
#files = os.listdir(base_dir+'Data')
for f in files:
    df[f] = pd.read_csv(base_dir+'Data/%s.txt'%f,index_col=0,header=None,delim_whitespace=True)

rmse_xgb_1 = [] # linear function
rmse_xgb_2 = [] # bandpass filter
rmse_xgb_3 = [] # nonlinear function
rmse_xgb_4 = [] # variable length filter
for i in [1,2,3,4]:
    # Perform train/test splits
    x = df[f_in]
    y = df['out0'+str(i)]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=1)
    
    #######################
    #####   XGBoost   #####
    #######################
    
    # Instantiate model 
    xgb = XGBRegressor(random_state=1)
    
    # Train the model on training data
    #xgb.fit(x_train.values.reshape(-1,1), y_train)
    xgb.fit(x_train, y_train)
    
    # Use the forest's predict method on the test data
    #y_pred_xgb = xgb.predict(x_test.values.reshape(-1,1))
    y_pred_xgb = xgb.predict(x_test)
    print("Prediction for test set: {}".format(y_pred_xgb))
    
    '''
    # Instantiate XGBoost model with constant random state for reproducibility
    xgb = XGBRegressor(random_state=1)
    
    # Show current parameters
    print('XGB parameters currently in use:\n')
    print(xgb.get_params())
    
    # Create parameter grid 
    param_grid = {
        'n_estimators': [400,500,600],
        'max_depth': [7,8,9],
        'learning_rate': [0.05,0.06,0.07],
        'colsample_bytree': [0.7,0.8,0.9],
        'subsample': [0.7,0.8,0.9]
    }
    
    # Instantiate grid search for "optimal" parameters using 3 fold cross validation  
    # and use all available cores
    grid_search = GridSearchCV(estimator=xgb,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    
    # Fit grid search to the data and make predictions
    grid_search.fit(x_train, y_train)
    grid_search.best_params_
    xgb = grid_search.best_estimator_
    y_pred_xgb = xgb.predict(x_test)
    '''
    # Model evaluation statistics 
    r2_xgb = metrics.r2_score(y_test, y_pred_xgb)
    mae_xgb = metrics.mean_absolute_error(y_test, y_pred_xgb)
    mse_xgb = metrics.mean_squared_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb))
    print('XGB stats...')
    #print('R squared: {:.2f}'.format(xgb.score(x,y)*100))
    print('R squared:', r2_xgb)
    print('Mean Absolute Error:', mae_xgb)
    print('Mean Square Error:', mse_xgb)
    print('Root Mean Square Error:', rmse_xgb)
    
    # Predicted vs. observed PM2.5 
    xgb_diff = pd.DataFrame({'Observed value': y_test, 'Predicted value': y_pred_xgb})
    xgb_diff.head()
    
    # Compute residuals
    e = y_test - y_pred_xgb

    # Create two-panel plots of data and residuals
    fig1=plt.figure(figsize=(16,8))
    fig1.suptitle('XGBoost Model Performance #'+str(i),fontsize=22)
    #ax1.scatter(x=e.index, y=e, s=5)
    ax1=fig1.add_subplot(2,1,1)
    e.sort_index().plot()
    ax1.annotate('RMSE = %.2f'%rmse_xgb,xy=(0.85,0.96),xycoords='axes fraction',fontsize=18,horizontalalignment='left',verticalalignment='top')
    ax1.set_xlabel('Time',fontsize=18)
    ax1.set_ylabel('Residual',fontsize=18)
    ax1.set_xlim([0,len(y)])
    ax1.set_ylim([-75,75])
    
    ax2=fig1.add_subplot(2,1,2)
    xgb_diff.sort_index().plot(ax=ax2)
    ax2.set_xlabel('Time',fontsize=18)
    ax2.set_ylabel('Data',fontsize=18)
    ax2.set_xlim([0,len(y)])
    ax2.set_ylim([-150,150])
    ax2.legend(loc='upper right',fancybox=True,shadow=True,fontsize=14,ncol=2,bbox_to_anchor=(1,1.1))
    
    fig1.tight_layout()
    #fig1.subplots_adjust(top=0.05,bottom=0.05)
    fig1.savefig(base_dir+'Figures/test1_%s.png'%i,dpi=300,bbox_inches='tight')

    # Run model 100 times to obtain confidence intervals
    for j in np.arange(0,100):
        # Perform train/test splits
        x = df[f_in]
        y = df['out0'+str(i)]
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5)
        
        #######################
        #####   XGBoost   #####
        #######################
        
        # Instantiate model
        xgb = XGBRegressor()
        
        # Train the model on training data
        #xgb.fit(x_train.values.reshape(-1,1), y_train)
        xgb.fit(x_train, y_train)
        
        # Use the forest's predict method on the test data
        #y_pred_xgb = xgb.predict(x_test.values.reshape(-1,1))
        y_pred_xgb = xgb.predict(x_test)
        
        rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb))
        if i==1:
            rmse_xgb_1.append(rmse_xgb)
        if i==2:
            rmse_xgb_2.append(rmse_xgb)
        if i==3:
            rmse_xgb_3.append(rmse_xgb)
        if i==4:
            rmse_xgb_4.append(rmse_xgb)
            
# Compute mean, standard deviations, and 95 percent confidence intervals
rmse_mean = [np.mean(rmse_xgb_1),np.mean(rmse_xgb_2),np.mean(rmse_xgb_3),np.mean(rmse_xgb_4)]
rmse_std = [np.std(rmse_xgb_1),np.std(rmse_xgb_2),np.std(rmse_xgb_3),np.std(rmse_xgb_4)]
rmse_2std = [2*np.std(rmse_xgb_1),2*np.std(rmse_xgb_2),2*np.std(rmse_xgb_3),2*np.std(rmse_xgb_4)]

n = len(rmse_xgb_1) # number of obs
z = 1.96 # for a 95% CI
ci = [z*(np.std(rmse_xgb_1)/np.sqrt(n)),z*(np.std(rmse_xgb_2)/np.sqrt(n)),z*(np.std(rmse_xgb_3)/np.sqrt(n)),z*(np.std(rmse_xgb_3)/np.sqrt(n))]

fig2=plt.figure(figsize=(8,4))
tc_label = ['Linear Function','Bandpass Filter','Nonlinear Function','Variable Length Filter']
plt.errorbar(tc_label,rmse_mean, yerr=rmse_2std, fmt='o-', color='Black', elinewidth=3,capthick=3,errorevery=1, alpha=1, ms=4, capsize = 5)
plt.bar(tc_label,rmse_mean,tick_label=tc_label)
plt.ylabel('RMSE',fontsize=14)
plt.title('Mean RMSE with 95% Confidence Intervals (2 SD)',fontsize=14)
#plt.ylim([0,30])
fig2.tight_layout()
fig2.savefig(base_dir+'Figures/test_ci.png',dpi=300,bbox_inches='tight')

##########################################################
#####   Experiment with Past History (Time Delays)   #####
##########################################################

# Read in input and output files 
d_in= pd.read_csv(base_dir+'Data/lin_in01.txt',index_col=0,header=None,delim_whitespace=True)
d_out = pd.read_csv(base_dir+'Data/filt01.txt',index_col=0,header=None,delim_whitespace=True)

rmse_xgb_1 = [] # no time shift
rmse_xgb_2 = [] # add time shift 1
rmse_xgb_3 = [] # add time shift 2
rmse_xgb_4 = [] # add time shift 3
for i in [0,1,2,3]:
    # Perform train/test splits
    x = pd.DataFrame()
    for j in np.arange(0,i+1):
        x[j] = np.roll(d_in,j).reshape(-1,) # time shift
    y = d_out
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=1)
    
    #######################
    #####   XGBoost   #####
    #######################
    
    # Instantiate model 
    xgb = XGBRegressor(random_state=1)
    
    # Train the model on training data
    #xgb.fit(x_train.values.reshape(-1,1), y_train)
    xgb.fit(x_train, y_train)
    
    # Use the forest's predict method on the test data
    #y_pred_xgb = xgb.predict(x_test.values.reshape(-1,1))
    y_pred_xgb = xgb.predict(x_test)
    print("Prediction for test set: {}".format(y_pred_xgb))
    
    '''
    # Instantiate XGBoost model with constant random state for reproducibility
    xgb = XGBRegressor(random_state=1)
    
    # Show current parameters
    print('XGB parameters currently in use:\n')
    print(xgb.get_params())
    
    # Create parameter grid 
    param_grid = {
        'n_estimators': [400,500,600],
        'max_depth': [7,8,9],
        'learning_rate': [0.05,0.06,0.07],
        'colsample_bytree': [0.7,0.8,0.9],
        'subsample': [0.7,0.8,0.9]
    }
    
    # Instantiate grid search for "optimal" parameters using 3 fold cross validation  
    # and use all available cores
    grid_search = GridSearchCV(estimator=xgb,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    
    # Fit grid search to the data and make predictions
    grid_search.fit(x_train, y_train)
    grid_search.best_params_
    xgb = grid_search.best_estimator_
    y_pred_xgb = xgb.predict(x_test)
    '''
    # Model evaluation statistics 
    r2_xgb = metrics.r2_score(y_test, y_pred_xgb)
    mae_xgb = metrics.mean_absolute_error(y_test, y_pred_xgb)
    mse_xgb = metrics.mean_squared_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb))
    print('XGB stats...')
    #print('R squared: {:.2f}'.format(xgb.score(x,y)*100))
    print('R squared:', r2_xgb)
    print('Mean Absolute Error:', mae_xgb)
    print('Mean Square Error:', mse_xgb)
    print('Root Mean Square Error:', rmse_xgb)
    
    # Predicted vs. observed PM2.5 
    xgb_diff = pd.DataFrame({'Observed value': np.squeeze(y_test), 'Predicted value': list(y_pred_xgb)})
    xgb_diff.head()
    
    # Compute residuals
    e = np.squeeze(y_test) - y_pred_xgb

    # Create two-panel plots of data and residuals
    fig3=plt.figure(figsize=(16,8))
    fig3.suptitle('XGBoost Model Performance | Add Time Shift = '+str(i),fontsize=22)

    ax1=fig3.add_subplot(2,1,1)
    e.sort_index().plot()
    ax1.annotate('RMSE = %.2f'%rmse_xgb,xy=(0.85,0.96),xycoords='axes fraction',fontsize=18,horizontalalignment='left',verticalalignment='top')
    ax1.set_xlabel('Time',fontsize=18)
    ax1.set_ylabel('Residual',fontsize=18)
    ax1.set_xlim([0,len(y)])
    ax1.set_ylim([-10,10])
    
    ax2=fig3.add_subplot(2,1,2)
    xgb_diff.sort_index().plot(ax=ax2)
    ax2.set_xlabel('Time',fontsize=18)
    ax2.set_ylabel('Data',fontsize=18)
    ax2.set_xlim([0,len(y)])
    #ax2.set_ylim([-75,75])
    ax2.legend(loc='upper right',fancybox=True,shadow=True,fontsize=14,ncol=2,bbox_to_anchor=(1,1.1))
    
    fig3.tight_layout()
    #fig3.subplots_adjust(top=0.05,bottom=0.05)
    fig3.savefig(base_dir+'Figures/test2_%s.png'%i,dpi=300,bbox_inches='tight')

    # Run model 100 times to obtain confidence intervals
    for j in np.arange(0,10):
        # Perform train/test splits
        x = pd.DataFrame()
        for j in np.arange(0,i+1):
            x[j] = np.roll(d_in,j).reshape(-1,) # time shift
        y = d_out
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5)
        
        #######################
        #####   XGBoost   #####
        #######################
        
        # Instantiate model 
        xgb = XGBRegressor()
        
        # Train the model on training data
        #xgb.fit(x_train.values.reshape(-1,1), y_train)
        xgb.fit(x_train, y_train)
        
        # Use the forest's predict method on the test data
        #y_pred_xgb = xgb.predict(x_test.values.reshape(-1,1))
        y_pred_xgb = xgb.predict(x_test)
        
        rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb))
        if i==0:
            rmse_xgb_1.append(rmse_xgb)
        if i==1:
            rmse_xgb_2.append(rmse_xgb)
        if i==2:
            rmse_xgb_3.append(rmse_xgb)
        if i==3:
            rmse_xgb_4.append(rmse_xgb)
            
# Compute mean, standard deviations, and 95 percent confidence intervals
rmse_mean_2 = [np.mean(rmse_xgb_1),np.mean(rmse_xgb_2),np.mean(rmse_xgb_3),np.mean(rmse_xgb_4)]
rmse_std_2 = [np.std(rmse_xgb_1),np.std(rmse_xgb_2),np.std(rmse_xgb_3),np.std(rmse_xgb_4)]
rmse_2std_2 = [2*np.std(rmse_xgb_1),2*np.std(rmse_xgb_2),2*np.std(rmse_xgb_3),2*np.std(rmse_xgb_4)]

n = len(rmse_xgb_1) # number of obs
z = 1.96 # for a 95% CI
ci = [z*(np.std(rmse_xgb_1)/np.sqrt(n)),z*(np.std(rmse_xgb_2)/np.sqrt(n)),z*(np.std(rmse_xgb_3)/np.sqrt(n)),z*(np.std(rmse_xgb_3)/np.sqrt(n))]

fig4=plt.figure(figsize=(8,4))
tc_label = ['No Time Shift','Add Time Shift 1','Add Time Shift 2','Add Time Shift 3']
plt.errorbar(tc_label,rmse_mean_2, yerr=rmse_2std_2, fmt='o-', color='Black', elinewidth=3,capthick=3,errorevery=1, alpha=1, ms=4, capsize = 5)
plt.bar(tc_label,rmse_mean_2,tick_label=tc_label)
plt.ylabel('RMSE',fontsize=14)
plt.title('Mean RMSE with 95% Confidence Intervals (2 SD)',fontsize=14)
#plt.ylim([0,30])
fig4.tight_layout()
fig4.savefig(base_dir+'Figures/test2_ci.png',dpi=300,bbox_inches='tight')

