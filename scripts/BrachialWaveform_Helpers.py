#This file contains helper functions for the Exploitation of brachial waveform feature analysis
import numpy as np
import pandas as pd
import scipy as ski
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def derivative_features(waveform):
    '''
    Compute the max, min, time at peak and time at minimum of the derivative of the waveforms
    Return the derivative features computed
    '''
    wv = waveform.to_numpy()
    features = pd.DataFrame(columns=['peak der','minimum derivative', 'time at peak der', 'time at minimum der'])
    max_val = np.zeros(wv.shape[0])
    idx_max = np.zeros(wv.shape[0])
    min_val = np.zeros(wv.shape[0])
    idx_min = np.zeros(wv.shape[0])
    for i in range(wv.shape[0]):
        waveform_pos = wv[i,wv[i,:]>0]
        x_axis = range(waveform_pos.shape[0])
        f = ski.interpolate.CubicSpline(x_axis,waveform_pos)
        f_dif = ski.interpolate.CubicSpline.derivative(f)
        max_val[i] = np.amax(f_dif(x_axis))
        min_val[i] = np.amin(f_dif(x_axis))
        idx_max[i] = np.argmax(f_dif(x_axis))
        idx_min[i] = np.argmin(f_dif(x_axis))
    features['peak der'] = max_val
    features['minimum derivative'] = min_val
    features['time at peak der'] = idx_max
    features['time at minimum der'] = idx_min
    features.index += 1
    return features


def total_area(wave):
    '''
    Calculate the total area (between the two minimum points) under the waveform
    Return the calculated area
    '''
    # initialize arrays
    wave_np=wave.to_numpy()
    zero_id=(wave==0)
    area=np.zeros(wave_np.shape[0])

    #calculate the integral for each waveform
    for i in range(wave_np.shape[0]):
        sys_time=np.argmax(wave_np[i])
        init =np.argmin(wave_np[i, :sys_time])
        final =np.argmin(wave_np[i,sys_time:])
        x_axis=range(wave.iloc[i].size)
        wv = ski.interpolate.CubicSpline(x_axis,wave_np[i])
        area[i]=ski.interpolate.CubicSpline.integrate(wv,init,final)
  
  
    area_data=pd.DataFrame(area, columns=['Total Area'])
    area_data.index+=1
    return area_data

def extract_features(waveform, heartRate, derivative=False, area=False):
    '''
    Extract important features from the waveforms to create the dataset
    derivative -> taked derivative features if true
    area -> calculates the area features if true
    Return the constructed DataFrame with the features
    '''
    data = pd.DataFrame()

    #extract baseline features, used for all the predictio s
    data['brSBP'] = waveform.max(axis=1)
    data['brDBP'] = waveform[waveform>0.1].min(axis=1)
    data['brPP'] = data['brSBP'] - data['brDBP']
    data['MAP'] = waveform.mean(axis=1)
    data['HR'] = heartRate

    #extrac derivative associated features
    if(derivative):
        der_data = derivative_features(waveform)
        data = data.join(der_data)

    #extract integral (area) associated features
    if(area):
        data['TotalArea'] = total_area(waveform)

    return data



def GradBoost(x,y, learning_rate, n_estimators, fold=10, test_size=0.20, verbose=True):
    '''
    Gradient Boosting regression with cross validation on training set (default 10-fold)
    learning_rate-> parameter for Gradient boosting regression
    n_estimator-> parameter for Gradient boosting regression
    fold-> number of folds for cross validation
    test_size-> Training and testing sets are separated (default test size = 0.2)
    Verbose = true allows printing results
    Returns r, R2, RMSE and MAE values and an array with the predictions and the true values fro the testing set
    '''
  
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size)

    #Normalize training dataset and apply that normalozation to the test dataset
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train=pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test=pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    #Initialize measurements
    r = np.zeros(fold)
    R2 = np.zeros(fold)
    RMSE = np.zeros(fold)
    MAE=np.zeros(fold)

    #create indexed for cross validation
    kf = KFold(n_splits=fold)
    i=0

    #Do cross validation on training set
    for train_index, test_index in kf.split(X_train):
        X_tr_fold, X_te_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_tr_fold, y_te_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        regressor = GradientBoostingRegressor(learning_rate=learning_rate,n_estimators=n_estimators)
        regressor.fit(X_tr_fold,np.ravel(y_tr_fold))
        y_predict = regressor.predict(X_te_fold)

        R2[i] = regressor.score(X_te_fold,y_te_fold)
        r[i] = np.corrcoef(np.ravel(y_te_fold), y_predict)[0,1]
        RMSE[i] = np.sqrt(sklearn.metrics.mean_squared_error(y_te_fold, y_predict))
        MAE[i] = sklearn.metrics.mean_absolute_error(y_te_fold, y_predict)

        if(verbose):
            print('i:{}, r:{}, R2:{}, RMSE:{}, MAE:{}'.format(i, r[i],R2[i],RMSE[i], MAE[i]))
        i+=1
  
    if(verbose):
        print('Mean results:')
        print(' r ={},  R2={},  RMSE={},  MAE={}'.format(np.mean(r[:10]), np.mean(R2[:10]), np.mean(RMSE[:10]), np.mean(MAE[:10])))

    #Training de model and testing with testing set
    regressor = GradientBoostingRegressor(learning_rate=learning_rate,n_estimators=n_estimators)
    regressor.fit(X_train,np.ravel(y_train))
    y_predict = regressor.predict(X_test)
    R2 = regressor.score(X_test,y_test)
    r = np.corrcoef(np.ravel(y_test), y_predict)[0,1]
    RMSE = np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_predict))
    MAE = sklearn.metrics.mean_absolute_error(y_test, y_predict)
    if(verbose):
        print('Test results')
        print('r:{}, R2:{}, RMSE:{}, MAE:{}'.format(r,R2,RMSE, MAE))

    return r,R2,RMSE, MAE, [y_predict, y_test]



