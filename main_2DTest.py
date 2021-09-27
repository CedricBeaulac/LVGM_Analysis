# -*- coding: utf-8 -*-
"""
Created on 12/02/21

@author: Cedric Beaulac 

LVM-MEGA : Test on  2D data sets
Including: Normal, GMM, VAE, B-VAE
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
from sklearn import cluster, datasets, mixture

import os

import cv2

import seaborn as sns

import matplotlib.pyplot as plt


####################################
# Import my libraries
####################################

from utility import *
from NNutility import *
#from MEGA_Pytorch import *
from MEGA_Numpy import *

####################################
# Generate data sets
####################################


n=500

pi=[0.3,0.4,0.3]
mu=[[0,-5],[-4,0],[8,-3]]
std=[[1,0.5],[0.5,0.5],[0.5,1]]

data = GMM_Gen(n,pi,mu,std)

plt.scatter(data[:,0].numpy(),data[:,1].numpy())

plt.scatter(data[0:200,0].numpy(),data[0:200,1].numpy())


#noisy_moons = datasets.make_moons(n_samples=n, noise=.10)

plt.scatter(noisy_moons[0][:,0],noisy_moons[0][:,1])

data= torch.tensor(noisy_moons[0]).float()

noisy_circles = datasets.make_circles(n_samples=n, noise=.10, factor=0.5)

plt.scatter(noisy_circles[0][:,0],noisy_circles[0][:,1])

data= torch.tensor(noisy_circles[0]).float()

###########################################
# Fixing arguments
###########################################


ObsDim = 2
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
LDim = 2
HDim = 5

model1 =  PVAE(ObsDim,HDim,LDim).to(device)
optimizer = optim.Adam(model1.parameters())
for epoch in range(1, 2000 + 1):
    ptrain(args, model1, device, data, optimizer, epoch,1)
    #ptest(args, model, device, testdata, epoch)
    
    
model2 =  PVAE(ObsDim,HDim,LDim).to(device)
optimizer = optim.Adam(model2.parameters())
for epoch in range(1, 2000 + 1):
    ptrain(args, model2, device, data, optimizer, epoch,2)
    #ptest(args, model, device, testdata, epoch)
    
    
model3 =  PVAE(ObsDim,HDim,LDim).to(device)
optimizer = optim.Adam(model3.parameters())
for epoch in range(1, 2000 + 1):
    ptrain_MEGA(args, model3, device, data, optimizer, epoch,2,0.1,2000)
    #ptest(args, model, device, testdata, epoch)
    
    
#Single Gaussian
sm = GaussianMixture(n_components=1, random_state=0).fit(data)

#GMM
gmm2 = GaussianMixture(n_components=2, random_state=0).fit(data)

#GMM
gmm3 = GaussianMixture(n_components=3, random_state=0).fit(data)

#GMM
gmm4 = GaussianMixture(n_components=4, random_state=0).fit(data)

#GMM
gmm5 = GaussianMixture(n_components=5, random_state=0).fit(data)

#GMM
gmm10 = GaussianMixture(n_components=10, random_state=0).fit(data)

#GMM
gmm20 = GaussianMixture(n_components=20, random_state=0).fit(data)

#GMM
gmm50 = GaussianMixture(n_components=50, random_state=0).fit(data)


#################################
# Generate points from fitted models
#################################
    
 
ntz = 5000

#VAE 
NewPoint = np.random.normal(loc=np.zeros(LDim), scale=np.ones(LDim), size=(ntz, LDim))
NewPoint = torch.tensor(NewPoint)
NewPoint = NewPoint.type(torch.FloatTensor)

vaemux, vaelogvarx = model1.decode(NewPoint)

varx = vaelogvarx.exp_()
varx = torch.diag_embed(varx[:,],0,1)
param = torch.cat((vaemux.view(ntz,1,ObsDim),varx),1)
mvn = torch.distributions.multivariate_normal.MultivariateNormal(param[:,0,:],param[:,1:ObsDim+1,:])
vaetestsample = mvn.sample()

fig, ax = plt.subplots()
ax.scatter(vaetestsample[0:200,0].numpy(),vaetestsample[0:200,1].numpy(), c='r', label='Generated data')
ax.scatter(data[0:200,0].numpy(),data[0:200,1].numpy(), c='b', label='Real sample')

plt.ylim((-2,2))
plt.xlim((-2,3))
legend = ax.legend(loc='lower right' )

plt.show()

#bVAE 
NewPoint = np.random.normal(loc=np.zeros(LDim), scale=np.ones(LDim), size=(ntz, LDim))
NewPoint = torch.tensor(NewPoint)
NewPoint = NewPoint.type(torch.FloatTensor)

bvaemux, bvaelogvarx = model2.decode(NewPoint)

varx = bvaelogvarx.exp_()
varx = torch.diag_embed(varx[:,],0,1)
param = torch.cat((bvaemux.view(ntz,1,ObsDim),varx),1)
mvn = torch.distributions.multivariate_normal.MultivariateNormal(param[:,0,:],param[:,1:ObsDim+1,:])
vaetestsample = mvn.sample()

fig, ax = plt.subplots()
ax.scatter(vaetestsample[0:200,0].numpy(),vaetestsample[0:200,1].numpy(), c='r', label='Generated data')
ax.scatter(data[0:200,0].numpy(),data[0:200,1].numpy(), c='b', label='Real sample')


plt.ylim((-10,10))
plt.xlim((-10,15))
legend = ax.legend(loc='lower right' )

plt.show()

#Single Gaussian
smtestsample = sm.sample(ntz)
smsample = np.random.permutation(smtestsample[0])

fig, ax = plt.subplots()
ax.scatter(smsample[0:200,0],smsample[0:200,1], c='r', label='Generated data')
ax.scatter(data[0:200,0].numpy(),data[0:200,1].numpy(), c='b', label='Real sample')


plt.ylim((-2,2))
plt.xlim((-2,3))
legend = ax.legend(loc='lower right' )

plt.show()


#rVAE 
NewPoint = np.random.normal(loc=np.zeros(LDim), scale=np.ones(LDim), size=(ntz, LDim))
NewPoint = torch.tensor(NewPoint)
NewPoint = NewPoint.type(torch.FloatTensor)

rvaemux, rvaelogvarx = model3.decode(NewPoint)

varx = rvaelogvarx.exp_()
varx = torch.diag_embed(varx[:,],0,1)
param = torch.cat((rvaemux.view(ntz,1,ObsDim),varx),1)
mvn = torch.distributions.multivariate_normal.MultivariateNormal(param[:,0,:],param[:,1:ObsDim+1,:])
vaetestsample = mvn.sample()

fig, ax = plt.subplots()
ax.scatter(vaetestsample[0:200,0].numpy(),vaetestsample[0:200,1].numpy(), c='r', label='Generated data')
ax.scatter(data[0:200,0].numpy(),data[0:200,1].numpy(), c='b', label='Real sample')


plt.ylim((-10,10))
plt.xlim((-10,15))
legend = ax.legend(loc='lower right' )

plt.show()

#Single Gaussian
smtestsample = sm.sample(ntz)
smsample = np.random.permutation(smtestsample[0])

fig, ax = plt.subplots()
ax.scatter(smsample[0:200,0],smsample[0:200,1], c='r', label='Generated data')
ax.scatter(data[0:200,0].numpy(),data[0:200,1].numpy(), c='b', label='Real sample')


plt.ylim((-2,2))
plt.xlim((-2,3))
legend = ax.legend(loc='lower right' )

plt.show()


#GMM
gmmtestsample = gmm20.sample(ntz)
gmmsample = np.random.permutation(gmmtestsample[0])

fig, ax = plt.subplots()
ax.scatter(gmmsample[0:200,0],gmmsample[0:200,1], c='r', label='Generated data')
ax.scatter(data[0:200,0].numpy(),data[0:200,1].numpy(), c='b', label='Real sample')


plt.ylim((-2,2))
plt.xlim((-2,3))
legend = ax.legend(loc='lower right' )

plt.show()
#################################
# Compute MEGA for tested model
#################################
data=data.numpy()
# VAE
MEGA_VAE = MEGA(data, vaemux.detach().numpy(),torch.diag_embed(vaelogvarx).detach().numpy())

# B-VAE
MEGA_BVAE = MEGA(data,bvaemux.detach().numpy(),torch.diag_embed(bvaelogvarx).detach().numpy())

# SM
MEGA_SM = MEGA(data,np.repeat(sm.means_,ntz,0),np.repeat(sm.covariances_,ntz,0))


# GMM
MEGA_GMM3 = MEGA(data,gmm3.means_[gmmtestsample[1][:]],gmm3.covariances_[gmmtestsample[1]])

# GMM
MEGA_GMM5 = MEGA(data,gmm5.means_[gmmtestsample[1][:]],gmm5.covariances_[gmmtestsample[1]])
    
#################################
# Comparing our FME to a sample of new data (SE)
#################################

gm = GaussianMixture(n_components=3, random_state=0).fit(data)
inside = gm.covariances_+np.matmul(gm.means_.reshape(-1,2,1),gm.means_.reshape(-1,1,2))
exact=np.zeros([shape(gm.means_)[1],shape(gm.means_)[1]])

for j in range(0,shape(gm.weights_)[0]):
    exact += gm.weights_[j]*inside[j]
 
    
no =500
gendata = np.zeros(no)
FME = np.zeros(no)  
y = np.zeros(no)
for i in range(0,no):
        
    nt = (i+1)*10
    y[i] = nt 
    gmmtestsample = gm.sample(nt)
    xtx = np.mean(np.matmul(gmmtestsample[0].reshape(-1,2,1),gmmtestsample[0].reshape(-1,1,2)),0)     
    gendata[i] = frobnorm(exact-xtx)
    E2 = np.matmul(gm.means_[gmmtestsample[1]].reshape(-1,2,1), gm.means_[gmmtestsample[1]].reshape(-1,1,2))  
    RHS =np.mean(E2+gm.covariances_[gmmtestsample[1]],0)
    FME[i] = frobnorm(exact-RHS)
           

fig, ax = plt.subplots()
ax.plot(y, gendata, 'r', label='SE')
ax.plot(y, FME, 'b', label='FME')


legend = ax.legend(loc='upper right' )

plt.show()



#################################
# Using MEGA for regularization
#################################
data=data.numpy()

score = np.zeros(6)
aic = np.zeros(6)

# SM
smtestsample = sm.sample(ntz)
MEGA_SM = MEGA(data,np.repeat(sm.means_,ntz,0),np.repeat(sm.covariances_,ntz,0))
score[0] = sm.lower_bound_-sum(MEGA_SM)*(1)
aic[0] = sm.aic(data)

# GMM
gmm2testsample = gmm2.sample(ntz)
MEGA_GMM2 = MEGA(data,gmm2.means_[gmm2testsample[1][:]],gmm2.covariances_[gmm2testsample[1]])
score[1] = gmm2.lower_bound_-sum(MEGA_GMM2)*(2)
aic[1] = gmm2.aic(data)

# GMM
gmm3testsample = gmm3.sample(ntz)
MEGA_GMM3 = MEGA(data,gmm3.means_[gmm3testsample[1][:]],gmm3.covariances_[gmm3testsample[1]])
score[2] = gmm3.lower_bound_-sum(MEGA_GMM3)*(3)
aic[2] = gmm3.aic(data)

# GMM
gmm4testsample = gmm4.sample(ntz)
MEGA_GMM4 = MEGA(data,gmm4.means_[gmm4testsample[1][:]],gmm4.covariances_[gmm4testsample[1]])
score[3] = gmm4.lower_bound_-sum(MEGA_GMM4)*(4)
aic[3] = gmm4.aic(data)

# GMM
gmm5testsample = gmm5.sample(ntz)
MEGA_GMM5 = MEGA(data,gmm5.means_[gmm5testsample[1][:]],gmm5.covariances_[gmm5testsample[1]])
score[4] = gmm5.lower_bound_-sum(MEGA_GMM5)*(5)
aic[4] = gmm5.aic(data)

# GMM
gmm10testsample = gmm10.sample(ntz)
MEGA_GMM10 = MEGA(data,gmm10.means_[gmm10testsample[1][:]],gmm10.covariances_[gmm10testsample[1]])
score[5] = gmm10.lower_bound_-sum(MEGA_GMM10)*(10)
aic[5] = gmm10.aic(data)


# GMM
gmm20testsample = gmm20.sample(ntz)
MEGA_GMM20 = MEGA(data,gmm20.means_[gmm20testsample[1][:]],gmm20.covariances_[gmm20testsample[1]])
score[6] = gmm20.lower_bound_-sum(MEGA_GMM20)*(20)
aic[6] = gmm20.aic(data)

# GMM
gmm50testsample = gmm50.sample(ntz)
MEGA_GMM50 = MEGA(data,gmm50.means_[gmm50testsample[1][:]],gmm50.covariances_[gmm50testsample[1]])
score[7] = gmm50.lower_bound_-sum(MEGA_GMM50)*(50)
aic[7] = gmm50.aic(data)


fig, ax = plt.subplots()
ax.plot([1,2,3,4,5,10],score)
ax.set(xlabel='Number of components', ylabel='LL-MEGA',
       title='Curve of penalize likelihood for GMMS with different number of components')
ax.grid()

fig, ax = plt.subplots()
ax.plot([1,2,3,4,5,10],aic)
ax.set(xlabel='Number of components', ylabel='AIC',
       title='AIC for GMMS with different number of components')
ax.grid()


gmmtestsample = gmm3.sample(ntz)
gmmsample = np.random.permutation(gmmtestsample[0])

fig, ax = plt.subplots()
ax.scatter(gmmsample[0:200,0],gmmsample[0:200,1], c='r', label='Generated data')
ax.scatter(data[0:200,0],data[0:200,1], c='b', label='Real sample')

legend = ax.legend(loc='lower right' )

plt.show()
