# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 01:21:43 2016

@author: Brian
"""

## py mcmc

import numpy as np
import pymc3 as pm
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd

size = 500
true_intercept = 1
true_slope = 2
x = np.linspace(0, 1, size)
# y = a + b*x
true_regression_line = true_intercept + true_slope * x
# add noise
model1 = true_regression_line + np.random.normal(scale=.5, size=size) #Noisy
model2 = true_regression_line + np.random.normal(scale=.2, size=size) #Less Noisy

np.random.seed = 0
permutation_set = np.random.permutation(size)
train_set = permutation_set[0:size//2]
test_set = permutation_set[size//2:size]

data = dict(x1=model1[train_set], x2=model2[train_set], y=true_regression_line[train_set])
with pm.Model() as model:
    # specify glm and pass in data. The resulting linear model, its likelihood and 
    # and all its parameters are automatically added to our model.
    pm.glm.glm('y ~ x1 + x2', data)
    step = pm.NUTS() # Instantiate MCMC sampling algorithm
    trace = pm.sample(2000, step, progressbar=False)
    

    
#### with our thing

# load ef 
# load rf
import pickle

print('\nStarted')
    #directory = '../input/'
directory = '../Dropbox/Kaggle_AllState/'
ensemble_dir = '../Dropbox/Kaggle_AllState/ensemble'
feats_dir = '../Dropbox/Kaggle_AllState/'
 
    # local
import platform
if (platform.system() == 'Windows'):
    directory = 'E:/Dropbox/Dropbox/Kaggle_AllState/'
    ensemble_dir = 'E:/Dropbox/Dropbox/Kaggle_AllState/ensemble'
    feats_dir = 'E:/Dropbox/Dropbox/Kaggle_AllState/'


ef_test = pickle.load(open(directory + 'ensembleet_oof_test.pkl', 'rb') )
ef_train = pickle.load(open(directory + 'ensembleet_oof_train.pkl', 'rb') )

rf_test = pickle.load(open(directory + 'ensemblerf_oof_test.pkl', 'rb') )
rf_train = pickle.load(open(directory + 'ensemblerf_oof_train.pkl', 'rb') ) 

x_test = np.concatenate((ef_test, rf_test), axis=1)


DATA_DIR = "./Data"
TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)

train = pd.read_csv(TRAIN_FILE, nrows=NROWS)
y_train = train['loss'].ravel()

data = dict(x1=ef_train, x2=rf_train, y=y_train)
with pm.Model() as model:
    # specify glm and pass in data. The resulting linear model, its likelihood and 
    # and all its parameters are automatically added to our model.
    pm.glm.glm('y ~ x1 + x2', data)
    step = pm.NUTS() # Instantiate MCMC sampling algorithm
    trace = pm.sample(2000, step, progressbar=True)