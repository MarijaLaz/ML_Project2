#This file contain basic helper functions used in different parts of the project
import numpy as np
import pandas as pd
import scipy as ski
from sklearn.preprocessing import MinMaxScaler

def split_data(X, Y,  ratio):
    '''
        Split the dataset X,Y according to the ratio given
    '''
    mask = np.random.rand(len(X)) < ratio
    x_train = X[mask].reset_index(drop=True)
    y_train = Y[mask].reset_index(drop=True)
    x_test = X[~mask].reset_index(drop=True)
    y_test = Y[~mask].reset_index(drop=True)
    return x_train, x_test, y_train, y_test, mask

def inverse(x):
    '''
        Used for Ees conversion.
        Prediction values given are 1/Ees
        Unit conversion from Pascal/mL to mmHg/mL 
    '''
    output=x/(133.33*1e6)
    return 1/output

def standardize(x,mean=None,std=None,test=False):
    '''
        test -> False : used for standardizing the train data
                True  : used for standardizing test data
        Standardize the data with zero mean and unit variance
    '''
    if not test:
        mean=np.mean(x,axis=0)
        std = np.std(x,axis=0)
    return (x-mean)/std, mean, std


def split_pressure(waveform, ees,data):
    '''
        Split waveform, Ees and Data in two groups: hypertense and not according to physiological values,
        i.e. values extracted from the waveform 
    '''
    idx_hypertenseSBP = waveform.max(axis=1) > 135
    idx_hypertenseDBP = waveform[waveform > .01].min(axis=1) > 85

    wv_hypertense = waveform[np.logical_and(idx_hypertenseSBP,idx_hypertenseDBP)]
    wv_normal = waveform[~np.logical_and(idx_hypertenseSBP,idx_hypertenseDBP)]

    data_hypertense = data[np.logical_and(idx_hypertenseSBP,idx_hypertenseDBP)]
    data_normal = data[~np.logical_and(idx_hypertenseSBP,idx_hypertenseDBP)]

    ees_hypertense = ees[np.logical_and(idx_hypertenseSBP,idx_hypertenseDBP)]
    ees_normal = ees[~np.logical_and(idx_hypertenseSBP,idx_hypertenseDBP)]

    return idx_hypertenseSBP, idx_hypertenseDBP, wv_hypertense, wv_normal, data_hypertense,data_normal,ees_hypertense,ees_normal


def derivative(waveform):
    '''
        Calculate the derivative of the waveform using CubicSplice function
    '''
    wv = waveform.to_numpy()
    derivative = np.zeros([wv.shape[0],wv.shape[1]])

    for i in range(wv.shape[0]):
        waveform_pos = wv[i,wv[i,:]>0]
        x_axis = range(waveform_pos.shape[0])
        f = ski.interpolate.CubicSpline(x_axis,waveform_pos)
        f_dif = ski.interpolate.CubicSpline.derivative(f)
        f_dif_values = f_dif(x_axis)
        f_dif_values = np.pad(f_dif_values, (0,wv.shape[1]-f_dif_values.shape[0]))
        derivative[i] = f_dif_values

    return derivative


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
        Yield num_batches batches of size batch_size from the dataset y,tx
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
