# -*- coding: utf-8 -*-
"""
Created on 22/01/21

@author: Cedric Beaulac
Clean place to drop function for LVS experiments
"""

import torch
##################################  
# GMM Generator
##################################

def GMM_Gen(n,pi,mu,std):
    
    ki = np.random.multinomial(1,pi,size=n) 
    mui = np.matmul(ki,mu)
    stdi = np.matmul(ki,std)
    sample = np.random.normal(mui,stdi)
            
    return sample

def GMM_Gen2(n,pi,mu,std):
    
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

