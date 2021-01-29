# -*- coding: utf-8 -*-
"""
Created on 22/01/21



@author: Cedric Beaulac 
Thank to https://github.com/pytorch/examples/blob/master/mnist/main.py

First test with VAE for HWD-BR data set
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

import os

import cv2

import seaborn as sns

import matplotlib.pyplot as plt


####################################
# Import my libraries
####################################
os.chdir("C:/Users/beaul/Dropbox/U of T/Year 5/ComputerVisionInference/Python_Project4")  

from utility import *
from VAEutility import *


n=10pi=[0.2,0.8],mu=[0,100],std=[1,1]