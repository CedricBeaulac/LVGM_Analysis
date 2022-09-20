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

from MEGA_Pytorch import *


####################################
# Defining VAEs
####################################

####################################
# PVAE: Probabilistic VAE 
# Encoder has 1 hidden layer
# Decoder has one hidden layer
####################################
class PVAE(nn.Module):
    def __init__(self, x_dim=10,h_dim=400,z_dim=20):
        super(PVAE, self).__init__()

        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc41 = nn.Linear(h_dim, x_dim)
        self.fc42 = nn.Linear(h_dim, x_dim)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        mux, logvarx = self.decode(z)
        return mux,logvarx, mu, logvar

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc41(h3),self.fc42(h3)


    
    
    
####################################
# Define the training procedure
####################################
# Reconstruction + KL divergence losses summed over all elements and batch
def ploss_function(mux,logvarx, x, muz, logvarz,beta,device):
    
    sigmax = logvarx.mul(0.5).exp_()
    pxn = torch.distributions.normal.Normal(mux, sigmax)
    logpx = torch.sum(pxn.log_prob(x.view(-1,784)))


    KLD = -0.5 * torch.sum(1 + logvarz - muz.pow(2) - logvarz.exp())

    

    return logpx - beta*KLD

    

def ptrain(args, model, device, data, optimizer, epoch,beta):
    model.train()
    perm = np.random.permutation(data.shape[0])
    Alldata = torch.tensor(data[perm, :])
    train_loss = 0
    for i in range(0, args.Number_batch):
        datas = Alldata[(i * args.batch_size):((i + 1) * args.batch_size),:].to(device)
        optimizer.zero_grad()
        mux,logvarx, muz, logvarz = model(datas)
        loss = -ploss_function(mux,logvarx, datas, muz, logvarz,beta,device)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    #train_loss /= data.shape[0]         
    #print('====> Epoch: {} Average loss: {:.4f}'.format(
    #      epoch, train_loss ))
    
def ptest(args, model, device, data, epoch,beta):
    mux,logvarx, mu, logvar = model(data)
    loss = -ploss_function(mux,logvarx, data, mu, logvar,beta,device)
    
    n = min(data.size(0), 8)
    comparison = torch.cat([data.view(-1, 1, 28, 28)[:n],
                                mux.view(-1, 1, 28, 28)[:n]])
    save_image(comparison.cpu(),'results/reconstruction_' + str(epoch) 
                   + '.png', nrow=n) 
    print('====> Epoch: {} Average test loss: {:.4f}'.format(
          epoch, loss/data.shape[0] ))
    


####################################
# Using MEGA for Regularization: Define the training procedure
####################################
# Reconstruction + KL divergence losses summed over all elements and batch
def ploss_function_regu(mux,logvarx, x, muz, logvarz,beta,alpha,device,model,ntz):
    
    sigmax = logvarx.mul(0.5).exp_()
    #sigmax = torch.ones(logvarx.shape)
    #sigmax = sigmax.to(device)
    pxn = torch.distributions.normal.Normal(mux, sigmax)
    logpx = torch.sum(pxn.log_prob(x))


    KLD = 0.5 * torch.sum(1 + logvarz - muz.pow(2) - logvarz.exp())

    LDim = shape(muz)[1]

    #VAE 
    NewPoint = np.random.normal(loc=np.zeros(LDim), scale=np.ones(LDim), size=(ntz, LDim))
    NewPoint = torch.tensor(NewPoint)
    NewPoint = NewPoint.type(torch.FloatTensor)

    vaemux, vaelogvarx = model.decode(NewPoint)
    Mega = MEGA(x,vaemux,torch.diag_embed(vaelogvarx))
    

    return logpx + beta*KLD - alpha*(Mega[0]+Mega[1])
    
def ptrain_MEGA(args, model, device, data, optimizer, epoch,beta,alpha,ntz):
    model.train()
    perm = np.random.permutation(data.shape[0])
    Alldata = torch.tensor(data[perm, :])
    train_loss = 0
    for i in range(0, args.Number_batch):
        datas = Alldata[(i * args.batch_size):((i + 1) * args.batch_size),:].to(device)
        optimizer.zero_grad()
        mux,logvarx, muz, logvarz = model(datas)
        loss = -ploss_function_regu(mux,logvarx, datas, muz, logvarz,beta,alpha,device,model,ntz)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    #train_loss /= data.shape[0]         
    #print('====> Epoch: {} Average loss: {:.4f}'.format(
    #      epoch, train_loss ))
        
def ptest_MEGA(args, model, device, data, epoch,beta,alpha,ntz):
    mux,logvarx, muz, logvarz = model(data)
    loss = -ploss_function_regu(mux,logvarx, data, muz, logvarz,beta,alpha,device,model,ntz)
    
    n = min(data.size(0), 8)
    comparison = torch.cat([data.view(-1, 1, 28, 28)[:n],
                                mux.view(-1, 1, 28, 28)[:n]])
    save_image(comparison.cpu(),'results/reconstruction_' + str(epoch) 
                   + '.png', nrow=n) 
    print('====> Epoch: {} Average test loss: {:.4f}'.format(
          epoch, loss/data.shape[0] ))
