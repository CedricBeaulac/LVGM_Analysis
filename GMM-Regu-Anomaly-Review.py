"""
@author: Cedric Beaulac
PhD Project #5
LVGM-MEGA: Created on (28/02/23)
GMM Regularization for anomaly detection
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
from random import sample

from keras.datasets import mnist
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GaussianMixture
from sklearn import cluster, datasets, mixture

import cv2

import scipy.io

from sklearn.datasets import load_boston

from MEGA_Numpy import *

###########################################
# Import Data
###########################################

WholeData = scipy.io.loadmat('ionosphere.mat')
data = np.array(WholeData['X'])
outlier_ind = np.array(WholeData ['y'])
df = pd.DataFrame(np.concatenate((data, outlier_ind),axis=1))

###########################################
# Fitting a sequence of GMM to be regularized
###########################################



nc = np.array([1,2,3,4,5,10,15,20])
models =list()
MEGAs =list()
ntz = 5000

## Fit the model and compute its MEGA
for i in range(nc.shape[0]):
    model = GaussianMixture(n_components=nc[i], random_state=0).fit(data) 
    models.append(model)
    mult = torch.distributions.categorical.Categorical(torch.tensor(models[i].weights_))
    components =  mult.sample([ntz,1])
    means = models[i].means_[components].reshape(ntz,33)
    covariances = models[i].covariances_[components].reshape(ntz,33,33)
    MEGAs.append(MEGA(data,means,covariances))


#################################
# Using MEGA for regularization
#################################
scores = list()
aics = list()
bics = list()

for i in range(nc.shape[0]):
    scores.append(models[i].score(data)-(0)*(MEGAs[i][0]+sqrt(MEGAs[i][1])))
    aics.append(models[i].aic(data))
    bics.append(models[i].bic(data))
    

fig, ax = plt.subplots()
ax.plot(nc,scores)
ax.set(xlabel='Number of components', ylabel='LL-MEGA',
       title='Curve of penalize likelihood for GMMS with different number of components')
ax.grid()
legend = ax.legend(loc='lower right' )
plt.show()


fig, ax = plt.subplots()
ax.plot(nc,aics)
ax.set(xlabel='Number of components', ylabel='AIC',
       title='AIC for different number of components')
ax.grid()
plt.savefig('ML-Review-NewPlots\AIC_Anomaly.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


fig, ax = plt.subplots()
ax.plot(nc,bics)
ax.set(xlabel='Number of components', ylabel='AIC',
       title='BIC for GMMS with different number of components')
ax.grid()
legend = ax.legend(loc='lower right' )
plt.show()

maximum = list()
alpha = np.linspace(0,500,501)

for j in range(np.shape(alpha)[0]):
    scores = list()
    for i in range(np.shape(nc)[0]):
        scores.append(models[i].score(data)-(alpha[j])*(MEGAs[i][0]+math.sqrt(MEGAs[i][1])))
    maximum.append(nc[argmax(scores)])   

    
fig, ax = plt.subplots()
ax.plot(alpha,maximum)
ax.set(xlabel='Alpha penalty parameter', ylabel='Number of components',
       title='Complexity Selection Path')
ax.grid()
plt.savefig('ML-Review-NewPlots\Alpha_path_Anomaly.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

# ------------------------------------------- #
# Step 4: GMM Predict Anomalies Using Percentage Threshold
# ------------------------------------------- #
# Get the score for each sample
score = models[2].score_samples(data)
# Save score as a column
df['score'] = pd.DataFrame(score)
# Get the score threshold for anomaly
pct_threshold = np.percentile(df['score'], 35)
# Print the score threshold
print(f'The threshold of the score is {pct_threshold:.2f}')
# Label the anomalies
df['anomaly_gmm_pct'] = df['score'].apply(lambda x: 1 if x < pct_threshold else 0)

sum(df['anomaly_gmm_pct'][df[33]==1])
sum(df[33])
sum(df['anomaly_gmm_pct'])



# ------------------------------------------- #
# Step 4: GMM Predict Anomalies Using Percentage Threshold
# ------------------------------------------- #
# Get the score for each sample
score = models[7].score_samples(data)
# Save score as a column
df['score'] = pd.DataFrame(score)
# Get the score threshold for anomaly
pct_threshold = np.percentile(df['score'], 35)
# Print the score threshold
print(f'The threshold of the score is {pct_threshold:.2f}')
# Label the anomalies
df['anomaly_gmm_pct'] = df['score'].apply(lambda x: 1 if x < pct_threshold else 0)

sum(df['anomaly_gmm_pct'][df[33]==1])
sum(df[33])
sum(df['anomaly_gmm_pct'])





