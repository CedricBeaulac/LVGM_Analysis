# -*- coding: utf-8 -*-
"""
Created on 22/01/21

@author: Cedric Beaulac 

LVM-MEGA
"""


####################################
# Import the good stuff (bunch of libraries)
####################################
from __future__ import print_function

import numpy as np
from numpy import *

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.autograd import Variable

import argparse

import torchvision
from torchvision import transforms
from torchvision import datasets, transforms
from torchvision.utils import save_image

from scipy import stats, integrate

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import os

import cv2

import seaborn as sns

import matplotlib.pyplot as plt


####################################
# Import my libraries
####################################

from utility import *
from NNutility import *



####################################
# Generate data sets
####################################


n=5000
pi=[0.3,0.3,0.4]
mu=[[0,10],[-5,0],[5,2]]
std=[[1,2],[0.5,0.5],[2,2]]

sample = GMM_Gen2(n,pi,mu,std)

plt.scatter(sample[:,0].numpy(),sample[:,1].numpy())


###########################################
# Fixing arguments
###########################################


ObsDim = shape(mu)[1]
BSize = 500
NB = int(n/BSize)

parser = argparse.ArgumentParser(description='LVM MEGA')
parser.add_argument('--batch-size', type=int, default=BSize, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--Number-batch', type=int, default=NB, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--beta', type=int, default=1, metavar='N',
                        help='beta parameter for beta-vae')
parser.add_argument('--alpha', type=int, default=0.1*140, metavar='N',
                        help='Unsupervised control')

parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


###########################################
# Defining model and training
###########################################


LDim = 2
HDim = 10

model =  PVAE(ObsDim,HDim,LDim).to(device)
optimizer = optim.Adam(model.parameters())
for epoch in range(1, 1000 + 1):
    ptrain(args, model, device, sample, optimizer, epoch)


#################################
# Testing process (Garbage)
#################################
    
    
nt = 5000

# Print new points
NewPoint = np.random.normal(loc=np.zeros(LDim), scale=np.ones(LDim), size=(nt, LDim))
NewPoint = torch.tensor(NewPoint)
NewPoint = NewPoint.type(torch.FloatTensor)

mux, logvarx = model.decode(NewPoint)

varx = logvarx.exp_()
varx = torch.diag_embed(varx[:,],0,1)
param = torch.cat((mux.view(nt,1,2),varx),1)
mvn = torch.distributions.multivariate_normal.MultivariateNormal(param[:,0,:],param[:,1:ObsDim+1,:])
testsample = mvn.sample()

#plt.scatter(mux[:,0].detach().numpy(),mux[:,1].detach().numpy())
plt.scatter(testsample[:,0].numpy(),testsample[:,1].numpy())


#################################
# First moment
#################################

# Generative model
xbar = np.mean(sample.numpy(),0)

EzEx = np.mean(mux.detach().numpy(),0)

EzEx/xbar

# Inference model
muz,logvarz = model.encoder(Data)

ExEz = np.mean(muz.detach().numpy(),0)

#################################
# Second moment
#################################

# Generative model
S2 = np.cov(np.transpose(sample))
xbar2 = np.outer(xbar,np.transpose(xbar))

LHS = S2+xbar2

E2= np.zeros((ObsDim,ObsDim))
for i in range(0,n):
    
    E2 += np.outer(sample[i,:], sample[i,:])
    
E2 = E2/(np.shape(sample)[0]-1)

#np.power(sample,2)

BigMatrix = np.zeros((ObsDim,ObsDim))
for i in range(0,n):
    
    V = np.diag(logvarx[i,:].detach().numpy())
    E2 = np.outer(mux[i,:].detach().numpy(), mux[i,:].detach().numpy())
    BigMatrix += V+E2
    
RHS = BigMatrix/n

#LHS-RHS

RHS/LHS

np.mean(RHS/LHS)

#Inference model


BigMatrix = np.zeros((LDim,LDim))
for i in range(0,np.shape(npData)[0]):
    
    V = np.diag(logvarz[i,:].detach().numpy())
    E2 = np.outer(muz[i,:].detach().numpy(), muz[i,:].detach().numpy())
    BigMatrix += V+E2
    
SMZ = BigMatrix/(np.shape(npData)[0])