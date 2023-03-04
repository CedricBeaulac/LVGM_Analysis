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

#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


####################################
# Generate data sets
####################################


n=1000

pi=[0.3,0.4,0.3]
mu=[[0,-5],[-4,0],[5,-4]]
std=[[1,0.5],[0.5,0.5],[0.5,1]]

data = GMM_Gen(n,pi,mu,std)

plt.scatter(data[:,0].numpy(),data[:,1].numpy())
plt.show()

###########################################
# Defining models and training
###########################################

#VAE
LDim = 2
HDim = 5

data = torch.tensor(data)

# Basic probabilitisc VAE (https://arxiv.org/abs/1312.6114)
model1 =  PVAE(ObsDim,HDim,LDim).to(device)
optimizer = optim.Adam(model1.parameters())
for epoch in range(1, 2000 + 1):
    ptrain(args, model1, device, data, optimizer, epoch,1)
    #ptest(args, model, device, testdata, epoch)

# Beta VAE with beta=2
model2 =  PVAE(ObsDim,HDim,LDim).to(device)
optimizer = optim.Adam(model1.parameters())
for epoch in range(1, 2000 + 1):
    ptrain(args, model1, device, data, optimizer, epoch,2)
    #ptest(args, model, device, testdata, epoch)
    
###########################################
# Fitting GMMs and computing MEGA
###########################################

ntz = 1000

gmm2 = GaussianMixture(n_components=2, random_state=0).fit(data)

gmm3 = GaussianMixture(n_components=3, random_state=0).fit(data)

#################################
# Generate points from fitted models
#################################
    
 
ntz = 1000

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
ax.scatter(vaetestsample[0:200,0].numpy(),vaetestsample[0:200,1].numpy(), c='r',marker='+', label='Generated data')
ax.scatter(data[0:200,0],data[0:200,1], c='b',marker='.', label='Real sample')
plt.ylim((-7,6))
plt.xlim((-7,8))
legend = ax.legend(loc='upper right', prop={'size': 15})
plt.savefig('ML-Review-NewPlots\Blob_Model1.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

MMD_VAE = MMD(torch.tensor(vaetestsample).double(),torch.tensor(data).double(),kernel='rbf'),

#bVAE 
NewPoint = np.random.normal(loc=np.zeros(LDim), scale=np.ones(LDim), size=(ntz, LDim))
NewPoint = torch.tensor(NewPoint)
NewPoint = NewPoint.type(torch.FloatTensor)

bvaemux, bvaelogvarx = model2.decode(NewPoint)

varx = bvaelogvarx.exp_()
varx = torch.diag_embed(varx[:,],0,1)
param = torch.cat((bvaemux.view(ntz,1,ObsDim),varx),1)
mvn = torch.distributions.multivariate_normal.MultivariateNormal(param[:,0,:],param[:,1:ObsDim+1,:])
bvaetestsample = mvn.sample()

fig, ax = plt.subplots()
ax.scatter(bvaetestsample[0:200,0].numpy(),bvaetestsample[0:200,1].numpy(), c='r',marker='+', label='Generated data')
ax.scatter(data[0:200,0],data[0:200,1], c='b',marker='.', label='Real sample')
plt.ylim((-7,6))
plt.xlim((-7,8))
legend = ax.legend(loc='upper right', prop={'size': 15})
plt.savefig('ML-Review-NewPlots\Blob_Model2.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

MMD_BVAE = MMD(bvaetestsample.double(),data.double(),kernel='rbf')

#2GMM
gmm2testsample = gmm2.sample(ntz)
gmm2sample = np.random.permutation(gmm2testsample[0])

fig, ax = plt.subplots()
ax.scatter(gmm2sample[0:200,0],gmm2sample[0:200,1], c='r',marker='+', label='Generated data')
ax.scatter(data[0:200,0],data[0:200,1], c='b', marker='.', label='Real sample')
plt.ylim((-7,6))
plt.xlim((-7,8))
legend = ax.legend(loc='upper right', prop={'size': 15})
plt.savefig('ML-Review-NewPlots\Blob_Model3.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

MMD_GMM2 = MMD(torch.tensor(gmm2testsample[0]).double(),torch.tensor(data).double(),kernel='rbf')

#GMM
gmm3testsample = gmm3.sample(ntz)
gmm3sample = np.random.permutation(gmm3testsample[0])

fig, ax = plt.subplots()
ax.scatter(gmm3sample[0:200,0],gmm3sample[0:200,1], c='r',marker='+', label='Generated data')
ax.scatter(data[0:200,0],data[0:200,1], c='b',marker='.', label='Real sample')
plt.ylim((-7,6))
plt.xlim((-7,8))
legend = ax.legend(loc='upper right', prop={'size': 15})
plt.savefig('ML-Review-NewPlots\Blob_Model4.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


MMD_GMM3 = MMD(torch.tensor(gmm3testsample[0]).double(),torch.tensor(data).double(),kernel='rbf')


#################################
# Compute MEGA for tested model
#################################
data=data.numpy()
# VAE
MEGA_VAE = MEGA(torch.tensor(data), vaemux.detach(),torch.diag_embed(vaelogvarx).detach())

# B-VAE
MEGA_BVAE = MEGA(torch.tensor(data),bvaemux.detach(),torch.diag_embed(bvaelogvarx).detach())


# GMM
mult = torch.distributions.categorical.Categorical(torch.tensor(gmm.weights_))
components =  mult.sample([ntz,1])
means = gmm3.means_[components].reshape(ntz,2)
covariances = gmm3.covariances_[components].reshape(ntz,2,2)
MEGA_GMM3 = MEGA(torch.tensor(data),torch.tensor(means),torch.tensor(covariances))

# GMM2
mult = torch.distributions.categorical.Categorical(torch.tensor(smm.weights_))
components =  mult.sample([ntz,1])
means = gmm2.means_[components].reshape(ntz,2)
covariances = gmm2.covariances_[components].reshape(ntz,2,2)
MEGA_GMM2 = MEGA(torch.tensor(data),torch.tensor(means),torch.tensor(covariances))
    
    
MEGA_VAE
MMD_VAE
MEGA_BVAE
MMD_BVAE
MEGA_GMM2
MMD_GMM2
MEGA_GMM3 
MMD_GMM3  


###########################################
# Again for Halfmoons
# Generate data
###########################################


data = datasets.make_moons(n_samples=n, noise=.10)

plt.scatter(data[0][:,0],data[0][:,1])
plt.show()

#VAE
LDim = 2
HDim = 5

data = torch.tensor(data[0]).float()

# Basic probabilitisc VAE (https://arxiv.org/abs/1312.6114)
model1 =  PVAE(ObsDim,HDim,LDim).to(device)
optimizer = optim.Adam(model1.parameters())
for epoch in range(1, 2000 + 1):
    ptrain(args, model1, device, data, optimizer, epoch,0.5)
    #ptest(args, model, device, testdata, epoch)

# Beta VAE with beta=2
model2 =  PVAE(ObsDim,HDim,LDim).to(device)
optimizer = optim.Adam(model2.parameters())
for epoch in range(1, 2000 + 1):
    ptrain(args, model2, device, data, optimizer, epoch,2)
    #ptest(args, model, device, testdata, epoch)
    
###########################################
# Fitting GMMs and computing MEGA
###########################################

ntz = 1000


TrData = torch.tensor(data[0:800,])
VData = torch.tensor(data[800:1000,])


gmm2 = GaussianMixture(n_components=2, random_state=0).fit(data)

gmm3 = GaussianMixture(n_components=3, random_state=0).fit(data)





#################################
# Generate points from fitted models
#################################
    
 
ntz = 1000

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
ax.scatter(vaetestsample[0:200,0].numpy(),vaetestsample[0:200,1].numpy(), c='r',marker='+', label='Generated data')
ax.scatter(data[0:200,0],data[0:200,1], c='b', marker='.', label='Real sample')
plt.ylim((-2,3))
plt.xlim((-2,3))
legend = ax.legend(loc='upper right', prop={'size': 15})
plt.savefig('ML-Review-NewPlots\Moon_Model1.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

MMD_VAE = MMD(torch.tensor(vaetestsample).double(),torch.tensor(data).double(),kernel='rbf'),

#bVAE 
NewPoint = np.random.normal(loc=np.zeros(LDim), scale=np.ones(LDim), size=(ntz, LDim))
NewPoint = torch.tensor(NewPoint)
NewPoint = NewPoint.type(torch.FloatTensor)

bvaemux, bvaelogvarx = model2.decode(NewPoint)

varx = bvaelogvarx.exp_()
varx = torch.diag_embed(varx[:,],0,1)
param = torch.cat((bvaemux.view(ntz,1,ObsDim),varx),1)
mvn = torch.distributions.multivariate_normal.MultivariateNormal(param[:,0,:],param[:,1:ObsDim+1,:])
bvaetestsample = mvn.sample()

fig, ax = plt.subplots()
ax.scatter(bvaetestsample[0:200,0].numpy(),bvaetestsample[0:200,1].numpy(), c='r',marker='+', label='Generated data')
ax.scatter(data[0:200,0],data[0:200,1], c='b',marker='.', label='Real sample')
plt.ylim((-2,3))
plt.xlim((-2,3))
legend = ax.legend(loc='upper right', prop={'size': 15})
plt.savefig('ML-Review-NewPlots\Moon_Model2.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

MMD_BVAE = MMD(bvaetestsample.double(),data.double(),kernel='rbf')

#2GMM
gmm2testsample = gmm2.sample(ntz)
gmm2sample = np.random.permutation(gmm2testsample[0])

fig, ax = plt.subplots()
ax.scatter(gmm2sample[0:200,0],gmm2sample[0:200,1], c='r',marker='+', label='Generated data')
ax.scatter(data[0:200,0],data[0:200,1], c='b', marker='.', label='Real sample')
plt.ylim((-2,3))
plt.xlim((-2,3))
legend = ax.legend(loc='upper right', prop={'size': 15})
plt.savefig('ML-Review-NewPlots\Moon_Model3.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

MMD_GMM2 = MMD(torch.tensor(gmm2testsample[0]).double(),torch.tensor(data).double(),kernel='rbf')

#GMM
gmm3testsample = gmm3.sample(ntz)
gmm3sample = np.random.permutation(gmm3testsample[0])

fig, ax = plt.subplots()
ax.scatter(gmm3sample[0:200,0],gmm3sample[0:200,1], c='r',marker='+', label='Generated data')
ax.scatter(data[0:200,0],data[0:200,1], c='b',marker='.', label='Real sample')
plt.ylim((-2,3))
plt.xlim((-2,3))
legend = ax.legend(loc='upper right', prop={'size': 15})
plt.savefig('ML-Review-NewPlots\Moon_Model4.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


MMD_GMM3 = MMD(torch.tensor(gmm3testsample[0]).double(),torch.tensor(data).double(),kernel='rbf')


#################################
# Compute MEGA for tested model
#################################
data=data.numpy()
# VAE
MEGA_VAE = MEGA(torch.tensor(data), vaemux.detach(),torch.diag_embed(vaelogvarx).detach())

# B-VAE
MEGA_BVAE = MEGA(torch.tensor(data),bvaemux.detach(),torch.diag_embed(bvaelogvarx).detach())

ntz = 5000
# GMM
mult = torch.distributions.categorical.Categorical(torch.tensor(gmm.weights_))
components =  mult.sample([ntz,1])
means = gmm3.means_[components].reshape(ntz,2)
covariances = gmm3.covariances_[components].reshape(ntz,2,2)
MEGA_GMM3 = MEGA(torch.tensor(data),torch.tensor(means),torch.tensor(covariances))

# GMM2
mult = torch.distributions.categorical.Categorical(torch.tensor(smm.weights_))
components =  mult.sample([ntz,1])
means = gmm2.means_[components].reshape(ntz,2)
covariances = gmm2.covariances_[components].reshape(ntz,2,2)
MEGA_GMM2 = MEGA(torch.tensor(data),torch.tensor(means),torch.tensor(covariances))
    
    
MEGA_VAE
MMD_VAE
MEGA_BVAE
MMD_BVAE
MEGA_GMM2
MMD_GMM2
MEGA_GMM3 
MMD_GMM3  


#################################
# Comparing our FME to a sample of new data (SE)
#################################

n=1000
pi=[0.3,0.4,0.3]
mu=[[0,-5],[-4,0],[8,-3]]
std=[[1,0.5],[0.5,0.5],[0.5,1]]

exact2=np.zeros([shape(gm.means_)[1],shape(gm.means_)[1]])
for j in range(0,shape(pi)[0]):
    exact2 += pi[j]*(np.diag(std[j])+np.matmul(np.reshape(mu[j],(2,1)),np.reshape(mu[j],(1,2))))


data = GMM_Gen(n,pi,mu,std)
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
    gendata[i] = frobnorm(exact2-xtx)
    E2 = np.matmul(gm.means_[gmmtestsample[1]].reshape(-1,2,1), gm.means_[gmmtestsample[1]].reshape(-1,1,2))  
    RHS =np.mean(E2+gm.covariances_[gmmtestsample[1]],0)
    FME[i] = frobnorm(exact2-RHS)
           

fig, ax = plt.subplots()
ax.plot(y, gendata, 'r--', label='SE',linewidth=2)
ax.plot(y, FME, 'b-', label='FME',linewidth=2)
legend = ax.legend(loc='upper right' )
plt.show()
