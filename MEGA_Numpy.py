# -*- coding: utf-8 -*-
"""
Created on 12/02/21

@author: Cedric Beaulac 

LVM-MEGA : Implementation for MEGA1 and MEGA2
"""
import numpy as np
##################################  
# frobnorm : Frobenius norm of the matrix A
##################################
def frobnorm(A):

    A2 = np.dot(A,A) 
    Trace = np.trace(A2)
    Frob = np.sqrt(Trace)

    return Frob

##################################  
# frobvect : Frobenius norm of the vector a
##################################
def frobvect(a):

    Frob = np.sqrt(np.sum(np.power(a,2)))

    return Frob



#################################
# MEGA: Returns MEGA1 and MEGA2 (using Frob norm)    
# Inputs: data (Data set), mean (sample of E[x|z]), var (sample of V[x|z])
# Output : MEGA1 (MEGA for first moment) and MEGA2 (MEGA for second moment)
#################################
def MEGA(data,mean,var):
    
    #Initialize MEGA
    MEGA = np.zeros(2)
    nz = np.shape(mean)[0]
    d = np.shape(mean)[1]


    
    #First moment MEGA
    #DE
    xbar = np.mean(data,0)
    #FME
    EzEx = np.mean(mean,0)
    MEGA[0] = frobvect(EzEx-xbar)

    # Second moment MEGA
    # LHS (DE)
    S2 = np.cov(np.transpose(data))
    xbar2 = np.outer(xbar,np.transpose(xbar))
    LHS = S2+xbar2
    # RHS (FME)
    E2 = np.matmul(mean.reshape(-1,np.shape(mean)[1],1),mean.reshape(-1,1,np.shape(mean)[1]))
    RHS = np.mean(E2+var,0)
    
    MEGA[1] = frobnorm(LHS-RHS)

    
    return MEGA
