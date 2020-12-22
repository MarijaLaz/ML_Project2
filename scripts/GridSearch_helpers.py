# This file contains the helpers for the Reproducing part of the project
import sklearn
import numpy as np
import scipy as ski
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


def crossvalidation(X_train, y_train, estimator, method, hyperparameters, kfold,verbose=True): 
    '''
        Function for external kfold crossvalidation with internal kfold crossvalidation using GridSearch on the training set
          X_train, y_train -> train data set
          estimator        -> the regressor model
          method:string    -> the name of the model
          hyperparameters:dictionary -> the parameters for the model
          kfolf            -> number of folds for crossvalidation
          verbose          -> enable prints
          -----
          Returns
          best_r           -> the best r score 
          best_R2          -> the best R2 score
          lowest_RMSE      -> the lowest RMSE
          lowest_MAE       -> the lowest MAE
          parameters_results -> the corresponding parameters in the folds used to obtain those results
    '''
    r = np.zeros(kfold)
    R2 = np.zeros(kfold)
    RMSE = np.zeros(kfold)
    MAE=np.zeros(kfold)

    best_parameters = []

    kf = KFold(n_splits=kfold)
    i=0
    for train_index, test_index in kf.split(X_train):
        X_tr_fold, X_te_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_tr_fold, y_te_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        grid_search = GridSearchCV(estimator = estimator, param_grid = hyperparameters, cv = kfold, n_jobs = -1)
        grid_search.fit(X_tr_fold,np.ravel(y_tr_fold))
        if verbose:
            print("Best Parameters for {} fold: {}".format(i,grid_search.best_params_))

        best_parameters.append(grid_search.best_params_)

        if (method=='Random Forest'):
            n = grid_search.best_params_["n_estimators"]
            m = grid_search.best_params_["max_depth"]
            regressor=RandomForestRegressor(n,max_depth=m)

        if (method=='SVR'):
            c = grid_search.best_params_["C"]
            g = grid_search.best_params_["gamma"]
            regressor=SVR(C=c,gamma=g)

        if (method=='Ridge Regression'):
            a = grid_search.best_params_["alpha"]
            regressor=Ridge(alpha=a)
    
        if (method=='Gradient Boosting'):
            l = grid_search.best_params_["learning_rate"]
            n = grid_search.best_params_["n_estimators"]
            regressor=GradientBoostingRegressor(learning_rate=l,n_estimators=n)

        regressor.fit(X_tr_fold,np.ravel(y_tr_fold))
        y_predict = regressor.predict(X_te_fold)
        R2[i] = regressor.score(X_te_fold,y_te_fold)
        r[i] = np.corrcoef(np.ravel(y_te_fold), y_predict)[0,1]
        RMSE[i] = sklearn.metrics.mean_squared_error(y_te_fold, y_predict)
        MAE[i] = sklearn.metrics.mean_absolute_error(y_te_fold, y_predict)
        if verbose:
            print('fold:{}, r:{}, R2:{}, RMSE:{}, MAE:{}'.format(i, r[i],R2[i],RMSE[i], MAE[i]))
        i+=1
    best_r = np.amax(r)
    best_R2 = np.amax(R2)
    lowest_RMSE = np.amin(RMSE)
    lowest_MAE = np.amin(MAE)
    r_argmax = np.argmax(r)
    R2_argmax = np.argmax(R2)
    RMSE_argmin = np.argmin(RMSE)
    MAE_argmin = np.argmin(MAE)
    if verbose:
        print('Best results:')
        print(' r ={} in i={},\n R2={} in i={},\n RMSE={} in i={},\n MAE={} in fold={}'.format(best_r, r_argmax, best_R2, R2_argmax, lowest_RMSE, RMSE_argmin, lowest_MAE, MAE_argmin)) 
  
    parameters_results = [best_parameters[i] for i in [r_argmax,R2_argmax,RMSE_argmin,MAE_argmin]] 
    return best_r,best_R2,lowest_RMSE,lowest_MAE,parameters_results



