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


####################################
# Defining our VAE
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

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        mux, logvarx = self.decode(z)
        return mux,logvarx, mu, logvar
    
    
    
####################################
# Define the training procedure
####################################
# Reconstruction + KL divergence losses summed over all elements and batch
def ploss_function(mux,logvarx, x, muz, logvarz,beta,device):
    
    sigmax = logvarx.mul(0.5).exp_()
    #sigmax = torch.ones(logvarx.shape)
    #sigmax = sigmax.to(device)
    pxn = torch.distributions.normal.Normal(mux, sigmax)
    logpx = torch.sum(pxn.log_prob(x))


    KLD = 0.5 * torch.sum(1 + logvarz - muz.pow(2) - logvarz.exp())


    return logpx + beta*KLD

def ptrain(args, model, device, data, optimizer, epoch):
    model.train()
    perm = np.random.permutation(data.shape[0])
    Alldata = torch.tensor(data[perm, :])
    train_loss = 0
    for i in range(0, args.Number_batch):
        datas = Alldata[(i * args.batch_size):((i + 1) * args.batch_size),:].to(device)
        optimizer.zero_grad()
        mux,logvarx, mu, logvar = model(datas)
        loss = -ploss_function(mux,logvarx, datas, mu, logvar,args.beta,device)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= data.shape[0]         
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss ))
