# -*- coding: utf-8 -*-
"""


@author: Cedric Beaulac
PhD Project #5
VAE Experiment on MNIST with MEGA (10/07/20)
Using pre-loaded data and data loader
"""


####################################
# Import packages
####################################
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import itertools

from keras.datasets import mnist
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import cv2

from MEGA_Pytorch import *

parser = argparse.ArgumentParser(description='VAE MNIST Example (Cedric Beaulac)')

parser.add_argument('--obsdim', type=int, default=784, metavar='N',
                    help='Observation dimension (default: 784)')
parser.add_argument('--ldim', type=int, default=5, metavar='N',
                    help='Observation dimension (default: 2 for visualization)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--beta', type=int, default=0.35, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--alpha', type=int, default=0.0000, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--ntz', type=int, default=100, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device("cuda")

kwargs = {'num_workers': 1, 'pin_memory': True} 

# Load data set

data_train =  datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor())
train_images = data_train.train_data[data_train.train_labels==4].float()/255

data_test=  datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.ToTensor())
test_images = data_test.train_data[data_test.train_labels==4].float()/255



train_loader = torch.utils.data.DataLoader(train_images,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_images,
    batch_size=args.batch_size, shuffle=True, **kwargs)

m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
M_vec = m.icdf(torch.tensor(np.arange(0.05,1,0.05)))
Manifold = torch.tensor([p for p in itertools.product(M_vec, repeat=2)]).float()
   

###################################
# Define the VAE
####################################
class PVAE(nn.Module):
    def __init__(self):
        super(PVAE, self).__init__()

        self.fc1 = nn.Linear(args.obsdim, 200)
        self.fc21 = nn.Linear(200, args.ldim)
        self.fc22 = nn.Linear(200, args.ldim)
        self.fc3 = nn.Linear(args.ldim, 200)
        self.fc41 = nn.Linear(200, args.obsdim)
        self.fc42 = nn.Linear(200, args.obsdim)

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
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        mux, logvarx = self.decode(z)
        return mux,logvarx, mu, logvar




####################################
# Define the training procedure
####################################
# Reconstruction + KL divergence losses summed over all elements and batch
def ploss_function(mux,logvarx, x, mu, logvar,beta):
    sigmax = torch.add(logvarx.mul(0.5).exp_(),1e-5)
    #sigmax = 0.1*torch.ones(logvarx.shape).to(device)
    pxn = torch.distributions.normal.Normal(mux, sigmax)
    logpx = torch.mean(pxn.log_prob(x.view(-1,784)))


    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    return logpx - beta*KLD

def ptrain(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        mux,logvarx, mu, logvar = model(data)
        loss = -ploss_function(mux,logvarx, data, mu, logvar,args.beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    train_loss /= len(train_loader.dataset)            
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss ))
    return(train_loss)
    


def ptest(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            mux,logvarx, mu, logvar = model(data)
            test_loss += -ploss_function(mux,logvarx, data, mu, logvar,args.beta).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data.view(args.batch_size, 1, 28, 28)[:n],
                                      mux.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return(test_loss)



####################################
# Run to train and print results
####################################
model = PVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
trainloss= torch.zeros(args.epochs)
testloss= torch.zeros(args.epochs)
for epoch in range(1, args.epochs + 1):
    trl=ptrain(epoch)
    trainloss[epoch-1]=-trl
    tel=ptest(epoch)
    testloss[epoch-1]=-tel
    with torch.no_grad():
        sample = torch.randn(64, args.ldim).to(device)
        mux,logvarx = model.decode(sample)
        #sigmax = logvarx.mul(0.5).exp_()
        sigmax = 0.1*torch.ones(logvarx.shape).to(device)
        pxn = torch.distributions.normal.Normal(mux, sigmax)
        samples = pxn.sample()
        #Manifold_space,varspace = model.decode(Manifold.to(device))
        save_image(samples.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
        #save_image(Manifold_space.view(361, 1, 28, 28),
        #               'results/Manifold_' + str(epoch) + '.png',nrow=19)
        #save_image(varspace.view(81, 1, 28, 28),
                       #'results/Variance_' + str(epoch) + '.png',nrow=9)




    
df = pd.DataFrame(np.column_stack((np.arange(1,args.epochs+1,1),testloss,trainloss))) 
df.columns = ['epoch','testloss','trainloss']  
plt.plot( 'epoch', 'testloss', data=df, marker='', color='blue', linewidth=2)
plt.plot( 'epoch', 'trainloss', data=df, marker='', color='red', linewidth=2)

####################################
# Visualize bottleneck space
####################################

big_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),batch_size=60000)
for batch_idx, (x,y) in enumerate(big_loader):
    x,y =x,y
    
    
mu, logvar = model.encode(x.view(-1, 784).to(device))
z = model.reparameterize(mu, logvar)
z = z.cpu().data.numpy()
pz = pd.DataFrame(np.column_stack((-z[:,0],z[:,1],y)))
pz.columns = ['z1','z2','classes']
pz.classes=pz.classes.astype('category')
sns.lmplot( x="z2", y="z1", data=pz,hue='classes',fit_reg=False)    

            
####################################
# Define the training procedure
####################################

def ploss_function_regu(mux,logvarx, x, mu, logvar,beta,alpha,model,ntz):
    sigmax = logvarx.mul(0.5).exp_()
    sigmax = torch.add(logvarx.mul(0.5).exp_(),1e-8)
    #sigmax = torch.ones(logvarx.shape).to(device)
    pxn = torch.distributions.normal.Normal(mux, sigmax)
    logpx = torch.mean(pxn.log_prob(x.view(-1,784)))


    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    LDim = mu.size()[1]

    #VAE 
    NewPoint = np.random.normal(loc=np.zeros(LDim), scale=np.ones(LDim), size=(ntz, LDim))
    NewPoint = torch.tensor(NewPoint)
    NewPoint = NewPoint.type(torch.FloatTensor)

    vaemux, vaelogvarx = model.decode(NewPoint.to(device))
    Mega = MEGA(x.view(-1).cpu(),vaemux.cpu(),torch.diag_embed(vaelogvarx).cpu())
    

    #return logpx - (beta*KLD + alpha*(Mega[0]+torch.sqrt(Mega[1])))
    #return logpx - (beta*KLD + alpha*(Mega[0]))
    #return logpx - (beta*KLD + alpha*(torch.sqrt(Mega[1])))
    return logpx - (beta*KLD - alpha*(Mega[0]+Mega[1]))


def ptrain_MEGA(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        mux,logvarx, mu, logvar = model(data)
        loss = -ploss_function_regu(mux,logvarx, data, mu, logvar,args.beta,args.alpha,model,args.ntz)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    train_loss /= len(train_loader.dataset)            
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss ))
    return(train_loss)
    


def ptest_MEGA(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            data = data.to(device)
            mux,logvarx, mu, logvar = model(data)
            test_loss += -ploss_function_regu(mux,logvarx, data, mu, logvar,args.beta,args.alpha,model,args.ntz).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data.view(data.size(0), 1, 28, 28)[:n],
                                      mux.view(data.size(0), 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return(test_loss)



####################################
# Run to train and print results
####################################
model = PVAE().to(device)
optimizer = optim.Adam(model.parameters())

trainloss= torch.zeros(args.epochs)
testloss= torch.zeros(args.epochs)
for epoch in range(1, args.epochs + 1):
    trl=ptrain_MEGA(epoch)
    trainloss[epoch-1]=-trl
    tel=ptest_MEGA(epoch)
    testloss[epoch-1]=-tel
    with torch.no_grad():
        sample = torch.randn(64, args.ldim).to(device)
        mux,logvarx = model.decode(sample)
        sigmax = logvarx.mul(0.5).exp_()
        #sigmax = torch.add(logvarx.mul(0.5).exp_(),1e-8)
        pxn = torch.distributions.normal.Normal(mux, sigmax)
        samples = pxn.sample()
        #Manifold_space,varspace = model.decode(Manifold.to(device))
        save_image(samples.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
        save_image(mux.view(64, 1, 28, 28),
                       'results/mu_' + str(epoch) + '.png')
        save_image(sigmax.view(64, 1, 28, 28),
                       'results/sigma_' + str(epoch) + '.png')
        #save_image(varspace.view(81, 1, 28, 28),
                       #'results/Variance_' + str(epoch) + '.png',nrow=9)




    
df = pd.DataFrame(np.column_stack((np.arange(1,args.epochs+1,1),testloss,trainloss))) 
df.columns = ['epoch','testloss','trainloss']  
plt.plot( 'epoch', 'testloss', data=df, marker='', color='blue', linewidth=2)
plt.plot( 'epoch', 'trainloss', data=df, marker='', color='red', linewidth=2)

####################################
# Visualize bottleneck space
####################################

big_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),batch_size=60000)
for batch_idx, (x,y) in enumerate(big_loader):
    x,y =x,y
    
    
mu, logvar = model.encode(x.view(-1, 784).to(device))
z = model.reparameterize(mu, logvar)
z = z.cpu().data.numpy()
pz = pd.DataFrame(np.column_stack((-z[:,0],z[:,1],y)))
pz.columns = ['z1','z2','classes']
pz.classes=pz.classes.astype('category')
sns.lmplot( x="z2", y="z1", data=pz,hue='classes',fit_reg=False)    
       