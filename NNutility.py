# -*- coding: utf-8 -*-
"""
Created on 22/01/21


@author: Cedric Beaulac
Clean place to drop NN Related stuff for LVS experiments
"""

import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from numpy import *
import pandas as pd

def train(args, model, device, dataset, optimizer, epoch,yindex):
    model.train()
    data,labels=dataset
    perm = np.random.permutation(data.shape[0])
    Alldata = torch.tensor(data[perm, :])
    Alltarget = torch.tensor(labels[perm,yindex])
    for i in range(0, args.Number_batch):
        data, target = Alldata[(i * args.batch_size):((i + 1) * args.batch_size),:].to(device), Alltarget[(i * args.batch_size):((i + 1) * args.batch_size)].double().to(device)
        data = data.view(args.batch_size,1,32,32).type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(args, model, device, dataset,epoch,yindex):
    model.eval()
    test_loss = 0
    correct = 0
    data,labels=dataset
    data = torch.tensor(data)
    target = torch.tensor(labels[:,yindex])
    with torch.no_grad():
            data, target = data.to(device), target.to(device)
            data = data.view(shape(data)[0],1,32,32).type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.LongTensor)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= data.shape[0]

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) on epoch {} \n'.format(
        #test_loss, correct, shape(data)[0],100. * correct / shape(data)[0],epoch))


def test_accuracy(args, model, device, dataset,yindex):
    correct = 0
    data,labels=dataset
    data = torch.tensor(data)
    target = torch.tensor(labels[:,yindex])
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        data = data.view(shape(data)[0],1,32,32).type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct/data.shape[0]
            
    return acc

def test_MSE(args, model, device, dataset,yindex):
    data,labels=dataset
    data = torch.tensor(data)
    target = torch.tensor(labels[:,yindex])
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        data = data.view(shape(data)[0],1,32,32).type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        output = model(data)
        MSE = nn.MSELoss(target,output)
            
    return MSEdef test_MSE(args, model, device, dataset,yindex):
    data,labels=dataset
    data = torch.tensor(data)
    target = torch.tensor(labels[:,yindex])
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        data = data.view(shape(data)[0],1,32,32).type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        output = model(data)
        MSE = nn.MSELoss(target,output)
            
    return MSE

####################################
# Defining our NN
####################################
# NN For padded data (32 x 32)
####################################
    
class VAE(nn.Module):
    def __init__(self):
        super(Net28_2k, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, ny)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output   
    

    
