# -*- coding: utf-8 -*-
"""
Created on 22/01/21

@author: Cedric Beaulac
Clean place to drop function for LVS experiments
"""

import torch
import numpy as np
from numpy import *

import pandas as pd

import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn

##################################  
# GMM_Gen: Gaussian Mixture Model Generator
# inputs: n- number of observations, pi- probabilities of components
# mu- mean of components and std- stand. dev. of component (no correlation atm)
##################################
def GMM_Gen(n,pi,mu,std):
    
    nk = torch.tensor(mu).size()[0]
    dim = torch.tensor(mu).size()[1]
    mult = torch.distributions.categorical.Categorical(torch.tensor(pi))
    ki =  mult.sample([n,1])
    one_hot = torch.nn.functional.one_hot(ki)
    mui = torch.tensordot(one_hot.float(),torch.tensor(mu).float(),1)
    stdi = torch.tensordot(one_hot.float(),torch.tensor(std).float(),1)
    stdi = torch.diag_embed(stdi[:,0,:],0,1,2)
    param = torch.cat((mui,stdi),1)
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(param[:,0,:],param[:,1:dim+1,:])
    sample = mvn.sample()
            
    return sample


##################################  
# power_iteration (Von Mises iteration)
# copied from https://en.wikipedia.org/wiki/Power_iteration
##################################
def power_iteration(A, num_simulations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


##################################  
# frobnorm : Frobenius norm of matrix A
##################################
def frobnorm(A):

    A2 = np.dot(A,A) 
    Trace = np.trace(A2)
    Frob = sqrt(Trace)

    return Frob



def frobvect(a):


    frob = sqrt(np.sum(np.power(a,2)))

    return Frob



##################################  
# ppca: Training a probabilistic PCA with EM (scales better)
##################################
def ppca(x,m,ite):
    

    #n,D are fixed by the problem
    #mu being the mean is the ML estimator, we fix it first 
    n,d = shape(x)
    mu = torch.mean(x,0).float()
    xd = torch.tensor(x-mu).float()

    
    #Random Initilization of parameters in need of training
    W = torch.tensor(np.random.normal(0,0.1,[d,m])).float()
    sig = torch.tensor(exp(np.random.normal(0,0.1,[1]))).float()
    


    # ite iterations of the EM algorithm
    for i in range(0,ite):
        
        #Setting stuff up
        M = torch.matmul(torch.transpose(W,0,1),W)+sig*torch.eye(m).float()
        Minv = torch.inverse(M).float()

        
        #E Step
        Ez =  torch.matmul(torch.matmul(Minv,torch.transpose(W,0,1)).double(),torch.transpose(xd,0,1).double()).float()
        Ez = Ez.view(n,m,1)
        Ezt = Ez.view(n,1,m)
        Ezz = sig*Minv + torch.matmul(Ez,Ezt)
        
        #M Step
        W1 = torch.sum(torch.matmul(xd.view(n,d,1),Ezt),0)
        W2 = torch.sum(Ezz,0)
        W = torch.matmul(W1.float(),W2.float())
        s1 = torch.sum(xd.pow(2),1).pow(0.5)
        s2 = torch.matmul(xd.view(n,1,d),torch.matmul(W.repeat(n,1,1),Ez)).view(n)
        s3 = torch.diagonal(torch.matmul(Ezz.float(),torch.matmul(torch.transpose(W,0,1),W).repeat(n,1,1)), dim1=-2, dim2=-1).sum(-1)
        sig2 = torch.sum(s1-(2*s2)+s3)/(n*d)
        
    return mu,W,sig2


##################################  
# ppca: LL
##################################
def ppca_ll(x,m):
    

    #n,D are fixed by the problem
    #mu being the mean is the ML estimator, we fix it first 
    n,d = shape(x)
    mu = torch.mean(x,0)
    xd = torch.tensor(x-mu)

    
    #Random Initilization of parameters in need of training
    S = np.cov(np.transpose(x.numpy()))
    a,b = linalg.eigh(S)
    
    sig2 = (1/(d-m))*np.sum(a[0:(d-m)])
    
    U = np.transpose(b[(d-m):])

    midle =sqrt(np.matmul(a[-m:],np.eye(m))-sig2*np.eye(m))

    W =np.matmul(np.matmul(U,midle),np.eye(m))    
    
    return mu,W,sig2