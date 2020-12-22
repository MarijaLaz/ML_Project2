# this file contains the object models for CNN and helper functions

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from itertools import product
from scripts.helpers import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class ConvModel1Channel(torch.nn.Module):
    def __init__(self):
        """1D Convolution NN model using 1 channel"""
        super().__init__()


        self.model = nn.Sequential(
          nn.Conv1d(in_channels=1, out_channels=2, kernel_size=5, stride=2, padding=2),
          nn.ReLU(),
          nn.Conv1d(in_channels=2, out_channels=4, kernel_size=5, stride=2, padding=2),
          nn.ReLU(),
          nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=2),
          nn.ReLU(),
          nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),
          #nn.Dropout(),
          nn.MaxPool1d(3,stride=2),
          #nn.Dropout(),
          nn.Flatten(),
          nn.Linear(16*2, 1)
        )
    def forward(self,x):
        return self.model(x)

class ConvModel2Channels(torch.nn.Module):
    def __init__(self):
        """1D Convolution NN model using 2 channels"""
        super().__init__()


        self.model = nn.Sequential(
          nn.Conv1d(in_channels=2, out_channels=8, kernel_size=5, stride=2, padding=2),
          nn.ReLU(),
          nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),
          nn.ReLU(),
          nn.Conv1d(in_channels=16, out_channels=18, kernel_size=5, stride=2, padding=2),
          nn.ReLU(),
          nn.Conv1d(in_channels=18, out_channels=24, kernel_size=5, stride=2, padding=2),
          #nn.Dropout(),
          nn.MaxPool1d(3,stride=2),
          #nn.Dropout(),
          nn.Flatten(),
          nn.Linear(24*2, 1)
        )
    def forward(self,x):
        return self.model(x)

class ConvModel3Channels(torch.nn.Module):
    def __init__(self):
        """1D Convolution NN model using 3 channels"""
        super().__init__()


        self.model = nn.Sequential(
          nn.Conv1d(in_channels=3, out_channels=6, kernel_size=5, stride=2, padding=2),
          nn.ReLU(),
          nn.Conv1d(in_channels=6, out_channels=9, kernel_size=5, stride=2, padding=2),
          nn.ReLU(),
          nn.Conv1d(in_channels=9, out_channels=12, kernel_size=5, stride=2, padding=2),
          nn.ReLU(),
          nn.Conv1d(in_channels=12, out_channels=15, kernel_size=5, stride=2, padding=2),
          #nn.Dropout(),
          nn.MaxPool1d(3,stride=2),
          #nn.Dropout(),
          nn.Flatten(),
          nn.Linear(15*2, 1)
        )
    def forward(self,x):
        return self.model(x)

def train(model, criterion, dataset_train, dataset_test, optimizer, NUM_EPOCHS,BATCH_SIZE):
    """
        @param model: torch.nn.Module
        @param criterion: torch.nn.modules.loss._Loss
        @param dataset_train: list
        @param dataset_test: list
        @param optimizer: torch.optim.Optimizer
        @param NUM_EPOCHS: int
        @param BATCH_SIZE: int
        -------
        Returns
        epochs_R2          -> the R2 score after each epoch from the testing set
        epochs_loss        -> the loss  after each epoch from the testing set
        epochs_R2_train    -> the R2 score after each epoch from the training set
        epochs_loss_train  -> the loss after each epoch from the training set
        epochs_r           -> the r corr coeff after each epoch from the testing set
        
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# code executes faster if GPU enabled
    epochs_R2 = []
    epochs_loss = []
    epochs_R2_train = []
    epochs_loss_train = []
    epochs_r = []
    print("Starting training")
    for epoch in range(NUM_EPOCHS):
        # Train an epoch
        model.train()
        R2 = []
        MSE = []
        for batch_y, batch_x in batch_iter(dataset_train[1], dataset_train[0], BATCH_SIZE,int(dataset_train[0].shape[0]/BATCH_SIZE)):
            batch_x, batch_y = torch.from_numpy(batch_x).float().to(device), torch.from_numpy(batch_y).float().view(-1).to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.view(-1),batch_y)

            loss.backward()
            optimizer.step()

            R2.append(r2_score(batch_y.cpu().detach().numpy(),outputs.view(-1).cpu().detach().numpy()))
            MSE.append(mean_squared_error(batch_y.cpu().detach().numpy(),outputs.view(-1).cpu().detach().numpy()))

    mean_R2 = np.mean(R2)
    mean_MSE = np.mean(np.array(MSE))
    epochs_R2_train.append(mean_R2)
    epochs_loss_train.append(mean_MSE)

    # Test the quality on the test set
    model.eval()

    r = []
    R2 = []
    MSE = []

    for batch_y, batch_x in batch_iter(dataset_test[1], dataset_test[0], BATCH_SIZE,int(dataset_test[0].shape[0]/BATCH_SIZE)):
        batch_x, batch_y = torch.from_numpy(batch_x).float().to(device), torch.from_numpy(batch_y).float().view(-1).to(device)

        prediction = model(batch_x)

        r.append(np.corrcoef(batch_y.cpu().detach().numpy(), prediction.view(-1).cpu().detach().numpy())[0,1])
        R2.append(r2_score(batch_y.cpu().detach().numpy(),prediction.view(-1).cpu().detach().numpy()))
        MSE.append(mean_squared_error(batch_y.cpu().detach().numpy(),prediction.view(-1).cpu().detach().numpy()))

    mean_MSE = np.mean(np.array(MSE))
    mean_r = np.mean(r)
    mean_R2 = np.mean(R2)
    epochs_R2.append(mean_R2)
    epochs_loss.append(mean_MSE)
    epochs_r.append(mean_r)

    if(epoch%50==0):
        print("\tEpoch {}   | R2 score: {}, r score: {}".format(epoch, mean_R2,mean_r))
    if(epoch==NUM_EPOCHS-1):
        print("\tEpoch {}   | R2 score: {}, r score: {}".format(epoch, mean_R2,mean_r))
    return epochs_R2,epochs_loss,epochs_R2_train,epochs_loss_train,epochs_r


def hypertuning_NN(dataset_train, dataset_test, parameters,NUM_EPOCHS,channels,verbose=True):
    '''
    Hypertuning the parameters for NN network
        dataset_train  -> [X_train,y_train] list containing the training set 
        dataset_test   -> [X_test,y_test] list containing the test set 
        parameters     -> dictionary of parameters for hyper tuning
        NUM_EPOCHS     -> number of epoch used for NN training
        channels:int   -> 1,2 or 3 , which NN model to use (with 1,2 or 3 channels)
        verbose        -> for enabiling prints
        -------
        Returns
        results        -> all the results obtained from using different combinations of the parameters
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    param_combin = []
    for l_rate, b_size in product(*parameters):
        param_combin.append([l_rate,b_size])
        if channels==1:
            model = ConvModel1Channel().to(device)
        if channels==2:
            model = ConvModel2Channels().to(device)
        if channels==3:
            model = ConvModel3Channels().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=l_rate)
        BATCH_SIZE = b_size
        if verbose:
            print("PARAMETERS: LEARNING RATE = {}, BATCH_SIZE = {}".format(l_rate,b_size))
        results.append(train(model,criterion,dataset_train,dataset_test, optimizer,NUM_EPOCHS,BATCH_SIZE))
    return results

