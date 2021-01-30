# -*- coding: utf-8 -*-
"""
Created on 22/01/21

@author: Cedric Beaulac
Clean place to drop function for LVS experiments
"""

import torch
import numpy as np


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


def Frobenius(A):

    A2 = np.dot(A,A) 
    Trace = np.trace(A2)
    Frob = sqrt(Trace)

    return Frob

