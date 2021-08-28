# -*- coding: utf-8 -*-
"""
Created on 27/03/21

@author: Cedric Beaulac 

LVM-MEGA : Implementation for MEGA1 and MEGA2 in PyTorch
"""
import torch
import numpy as np
##################################  
# frobnorm : Frobenius norm of the matrix A
##################################
def frobnorm(A):

    A2 = torch.matmul(A,A) 
    Trace = torch.trace(A2)
    Frob = torch.sqrt(Trace)

    return Frob

##################################  
# frobvect : Frobenius norm of the vector a
##################################
def frobvect(a):

    Frob = torch.sqrt(torch.sum(torch.pow(a,2)))

    return Frob



#################################
# MEGA: Returns MEGA1 and MEGA2 (using Frob norm)    
# Inputs: data (Data set,ndarray), mean (sample of E[x|z],tensor), var (sample of V[x|z],tensor)
# Output : MEGA1 (MEGA for first moment) and MEGA2 (MEGA for second moment)
#################################
def MEGA(data,mean,var):
    
    #Initialize MEGA
    MEGA = torch.zeros(2)
    nz = mean.size()[0]
    d = mean.size()[1]


    
    #First moment MEGA
    #DE
    xbar = torch.mean(data,0)
    #FME
    EzEx = torch.mean(mean,0)
    MEGA[0] = frobvect(EzEx-xbar)

    # Second moment MEGA
    # LHS (DE)
    S2 = np.cov(np.transpose(data))
    xbar2 = np.outer(xbar,np.transpose(xbar))
    LHS = torch.tensor(S2+xbar2)
    # RHS (FME)
    E2 = torch.matmul(mean.reshape(-1,mean.size()[1],1),mean.reshape(-1,1,mean.size()[1]))
    RHS = torch.mean(E2+var,0)
    
    MEGA[1] = torch.sqrt(frobnorm(LHS-RHS))

    
    return MEGA
