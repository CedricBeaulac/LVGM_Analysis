# -*- coding: utf-8 -*-
"""
Created on 30/01/21

@author: Cedric Beaulac 

LVM-MEGA : Test on MNIST data 
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
from sklearn.mixture import GaussianMixture

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


big_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),batch_size=20000)
for batch_idx, (x,y) in enumerate(big_loader):
    x,y =x,y

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=500, shuffle=True, **kwargs)
for batch_idx, (tx,ty) in enumerate(big_loader):
    tx,ty =tx,ty

data = x.view(-1,784)

testdata = tx.view(-1,784)

###########################################
# Fixing arguments
###########################################


ObsDim = shape(data)[1]
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
# Defining models and training
###########################################

#VAE
LDim = 5
HDim = 400

model =  PVAE(ObsDim,HDim,LDim).to(device)
optimizer = optim.Adam(model.parameters())
for epoch in range(1, 500 + 1):
    ptrain(args, model, device, data, optimizer, epoch)
    ptest(args, model, device, testdata, epoch)
    
    
#GMM
gm = GaussianMixture(n_components=20, random_state=0).fit(data)


#pPCA
ldim = 250
mu,W,sig2 = ppca(data,ldim,5000)


mu,W,sig2 = ppca_ll(data,ldim)
#################################
# Generate points from fitted models
#################################
    
 
ntz = 1000

#VAE 
NewPoint = np.random.normal(loc=np.zeros(LDim), scale=np.ones(LDim), size=(ntz, LDim))
NewPoint = torch.tensor(NewPoint)
NewPoint = NewPoint.type(torch.FloatTensor)

mux, logvarx = model.decode(NewPoint)

varx = logvarx.exp_()
varx = torch.diag_embed(varx[:,],0,1)
param = torch.cat((mux.view(ntz,1,ObsDim),varx),1)
mvn = torch.distributions.multivariate_normal.MultivariateNormal(param[:,0,:],param[:,1:ObsDim+1,:])
vaetestsample = mvn.sample()

im = vaetestsample[0:100].view(100, 1, 28, 28)
save_image(im,'VAE' + str(epoch) + '.png', nrow=10)

im = mux[0:100].view(100, 1, 28, 28)
save_image(im,'MeansVAE' + str(epoch) + '.png', nrow=10)

#GMM
gmmtestsample = gm.sample(100)
plt.scatter(gmmtestsample[0][:,0],gmmtestsample[0][:,1])

#gmmtestsample[0][gmmtestsample[0]<0]=0
im = torch.tensor(gmmtestsample[0][0:100]).view(100, 1, 28, 28)
save_image(im,'gmm' + str(epoch) + '.png', nrow=10)

#pPCA
NewPoints = np.random.normal(loc=np.zeros(ldim), scale=np.ones(ldim), size=(ntz, ldim))
NewPoints = torch.tensor(NewPoints)
NewPoints = torch.transpose(NewPoints,0,1)
X = torch.matmul(torch.tensor(W).float(),NewPoints.float())
means = torch.transpose(X,0,1) + mu.repeat(ntz,1)
eps = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(ObsDim),sig2*torch.eye(ObsDim))
epssample = eps.sample([ntz])
ppcasample = means+epssample

ppcasample[ppcasample<0]=0
im = ppcasample[0:100].view(100, 1, 28, 28)
save_image(im,'ppca' + str(epoch) + '.png', nrow=10)

#################################
# First moment
#################################

# Generative model
xbar = np.mean(data.numpy(),0)

# VAE
EzEx = np.mean(mux.detach().numpy(),0)

megaf =np.zeros(3)  

EzEx-xbar



#################################
# Second moment
#################################

# Generative model
S2 = np.cov(np.transpose(data))
xbar2 = np.outer(xbar,np.transpose(xbar))

LHS = S2+xbar2

E2= np.zeros((ObsDim,ObsDim))
for i in range(0,n):
    
    E2 += np.outer(data[i,:], data[i,:])
    
E2 = E2/(np.shape(data)[0]-1)

#np.power(data,2)

MEGAF =np.zeros(3)

#VAE
BigMatrix = np.zeros((ObsDim,ObsDim))
for i in range(0,ntz):
    
    V = np.diag(logvarx[i,:].detach().numpy())
    E2 = np.outer(mux[i,:].detach().numpy(), mux[i,:].detach().numpy())
    BigMatrix += V+E2
    
RHS = BigMatrix/n

Gap = LHS-RHS


MEGAF[2] = frobnorm(Gap)

#GMM
BigMatrix = np.zeros((ObsDim,ObsDim))
for i in range(0,ntz):
    
    V = gm.covariances_[gmmtestsample[1][i]]
    E2 = np.outer(gm.means_[gmmtestsample[1][i]], gm.means_[gmmtestsample[1][i]])
    BigMatrix += V+E2
    
RHS = BigMatrix/n

Gap = LHS-RHS


MEGAF[0] = frobnorm(Gap)

#pPCA
BigMatrix = np.zeros((ObsDim,ObsDim))
for i in range(0,n):
    
    V = sig.pow(2).numpy()*np.eye(ObsDim)
    E2 = np.outer(means[i].numpy(), means[i].numpy())
    BigMatrix += V+E2
    
RHS = BigMatrix/n

Gap = LHS-RHS


MEGAF[1] = frobnorm(Gap)

MEGAF
#################################
# Inference model
#################################


BigMatrix = np.zeros((LDim,LDim))
for i in range(0,np.shape(npData)[0]):
    
    V = np.diag(logvarz[i,:].detach().numpy())
    E2 = np.outer(muz[i,:].detach().numpy(), muz[i,:].detach().numpy())
    BigMatrix += V+E2
    
SMZ = BigMatrix/(np.shape(npData)[0])


muz,logvarz = model.encoder(data)

ExEz = np.mean(muz.detach().numpy(),0)


#################################
# Checking how our pPCA is doing
#################################

S = np.cov(np.transpose(data))
a,b = linalg.eigh(S)
