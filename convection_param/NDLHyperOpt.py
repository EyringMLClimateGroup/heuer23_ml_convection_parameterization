#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import psyplot.project as psy
import psyplot
import os
from os import DirEntry
import sys
import re
from itertools import chain
from numba import jit, njit
from HelperFuncs import *
from convection_param.ConvParamFuncs import *
from convection_param.GetTrainingData import *
import pickle
from tqdm.notebook import tqdm
import datetime
from pprint import pprint
from TaylorDiagram import TaylorDiagram
import time
import xgboost as xgb
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression,Ridge,MultiTaskLasso,MultiTaskElasticNet,LassoLars
from sklearn.linear_model import RidgeCV,MultiTaskLassoCV,MultiTaskElasticNetCV

# data = np.load('../Processed/TrainData/R2B5_vcg_20221118-153949.npz')
# data = np.load('../local_data/TrainData/20230111-165428-R2B5_y13y16_vcg-fluxes_rho_fluct.npz')
# data = np.load('../local_data/TrainData/20230131-171851-R2B5_y13y16_vcg-fluxes_rho_fluct.npz')
# data = np.load('../local_data/TrainData/20230216-161721-R2B5_y13y16_vcg-fluxes_rho_fluct_8days.npz')
data = np.load('../local_data/TrainData/20230216-163349-R2B5_y13y16_vcg-fluxes_rho_fluct_5days.npz')

X_train, X_val, X_test, Y_train, Y_val, Y_test, X_expl, Y_expl = \
data['X_train'], data['X_val'], data['X_test'], data['Y_train'], data['Y_val'], data['Y_test'], data['X_expl'], data['Y_expl']

from HelperFuncs import unique_unsorted

# vars_to_neglect = ['qr','qi','qs']
vars_to_neglect = ['qr','qs']
# vars_to_neglect = []
vars_to_neglect_mask = ~np.isin(unique_unsorted([e[0] for e in X_expl]), vars_to_neglect)
print(vars_to_neglect_mask)
X_train = X_train[:,:,vars_to_neglect_mask]
X_val = X_val[:,:,vars_to_neglect_mask]
X_test = X_test[:,:,vars_to_neglect_mask]
X_expl = np.array([e for e in X_expl if e[0] not in vars_to_neglect])

print('X_train shape: ', X_train.shape)
print('X_val shape: ', X_val.shape)
print('X_test shape: ', X_test.shape)
print('Y_train shape: ', Y_train.shape)
print('Y_val shape: ', Y_val.shape)
print('Y_test shape: ', Y_test.shape)
print('len X_expl', len(X_expl))
print('len Y_expl', len(Y_expl))

# If channeled data is input
X_train = X_train.reshape(X_train.shape[0], -1)
# X_train = X_train[~np.isclose(X_train.std(axis=0), 0)]
X_val = X_val.reshape(X_val.shape[0], -1)
# X_val = X_val[~np.isclose(X_val.std(axis=0), 0)]
X_test = X_test.reshape(X_test.shape[0], -1)
# X_test = X_test[~np.isclose(X_test.std(axis=0), 0)]
print('X_train shape: ', X_train.shape)
print('X_val shape: ', X_val.shape)
print('X_test shape: ', X_test.shape)

# Concatenation of train and val set because of used cross validation
X_train = np.concatenate([X_train, X_val])
Y_train = np.concatenate([Y_train, Y_val])

print('X_train concatenated shape: ', X_train.shape)
print('Y_train concatenated shape: ', Y_train.shape)

npr.seed(1245845)
# sample_size = X_train.shape[0]#10000#int(0.1*X_train.shape[0])
# sample_idx = npr.choice(X_train.shape[0], size=sample_size)#np.s_[:]#

# X_train_sample, Y_train_sample = X_train[sample_idx], Y_train[sample_idx]
X_train_sample, Y_train_sample = X_train, Y_train
print(X_train_sample.shape)
print(Y_train_sample.shape)

def get_sp_size(search_space):
    result = 1
    for key,space in search_space.items():
        result *= len(space)
    return result

ridge_sp = {'alpha': np.logspace(-1,4,10)}

lasso_sp = {'alpha': np.logspace(-1,4,10)}

elasticnet_sp = {'alpha': np.logspace(-1,4,10),
                           'l1_ratio': [0.1,0.3,0.5,0.7,0.9]}

lassolars_sp = {'alpha': np.logspace(-1,4,10)}

kernel_ridge_sp = {'alpha': np.logspace(-1,4,10),
                   'kernel': ['rbf','polynomial','linear']}

random_forest_sp = {'bootstrap': [False, True],
                   'max_features': ['sqrt',1. ],
                   'min_samples_split': [0.5,0.1,0.01],#[2, 0.5, 0.1, 0.01],
                   'n_estimators': [10, 100, 200]}#, 500, 1000]}

extra_trees_sp = {'bootstrap': [False, True],
                  'max_features': ['sqrt',1.],
                  'min_samples_split': [0.5, 0.1, 0.01],
                  'n_estimators': [10, 100, 200]}

gradient_boost_sp = {'estimator__learning_rate': [0.5,0.1,0.01,0.001],
                     'estimator__max_features': ['sqrt',1.],
                     'estimator__n_estimators': [10, 100, 200],
                     'estimator__subsample': [0.1, 0.5, 1.],
                     'estimator__max_depth': [1,3,5,7]}

hist_gradient_boost_sp = {'estimator__learning_rate': [0.5,0.1,0.01],
                          'estimator__max_leaf_nodes': [40,30,20],
                          'estimator__min_samples_leaf': [20,25,30],
                          'estimator__l2_regularization': [0,0.01,0.1]}

svr_sp = {'estimator__kernel': ['rbf','polynomial','linear'],
          'estimator__C': np.logspace(-2,2,10)}

print('Size of search spaces:')
print('ridge: ', get_sp_size(ridge_sp))
print('lasso: ', get_sp_size(lasso_sp))
print('elasticnet: ', get_sp_size(elasticnet_sp))
print('kernel ridge: ', get_sp_size(kernel_ridge_sp))
print('RF: ', get_sp_size(random_forest_sp))
print('ET: ', get_sp_size(extra_trees_sp))
print('Grad Boost: ', get_sp_size(gradient_boost_sp))
print('Hist Grad Boost: ', get_sp_size(hist_gradient_boost_sp))
print('SVR: ', get_sp_size(svr_sp))
print('Lasso Lars: ', get_sp_size(lassolars_sp))

from functools import partial
MyGridSearchCV = partial(GridSearchCV, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=4)
MyGridSearchCV_won_jobs = partial(GridSearchCV, cv=3, scoring='neg_mean_squared_error', verbose=4)
MyRandomizedSearchCV = partial(RandomizedSearchCV, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=4)

models = [LinearRegression(n_jobs=-1),
          MyGridSearchCV(Ridge(), param_grid=ridge_sp),
          MyGridSearchCV(MultiTaskLasso(), param_grid=lasso_sp),
          MyGridSearchCV(MultiTaskElasticNet(), param_grid=elasticnet_sp),
          MyGridSearchCV(KernelRidge(), param_grid=kernel_ridge_sp),
          MyGridSearchCV_won_jobs(RandomForestRegressor(n_jobs=-1), param_grid=random_forest_sp),
          MyGridSearchCV_won_jobs(ExtraTreesRegressor(n_jobs=-1), param_grid=extra_trees_sp),
          MyGridSearchCV(MultiOutputRegressor(SVR()), param_grid=svr_sp),
          # MyRandomizedSearchCV(MultiOutputRegressor(GradientBoostingRegressor()), param_distributions=gradient_boost_sp, n_iter=100),
          MyGridSearchCV_won_jobs(MultiOutputRegressor(HistGradientBoostingRegressor(),n_jobs=-1), param_grid=hist_gradient_boost_sp),
          MyGridSearchCV(LassoLars(), param_grid=lassolars_sp)]

idx = int(sys.argv[1])

model = models[idx]
model_path = 'Models/ClassicHPO/20230504-171743-nsamples143401-5days_woqrqs/'
os.makedirs(model_path, exist_ok=True)

print('Fitting Model', model)
model.fit(X_train_sample, Y_train_sample)
dump(model, os.path.join(model_path, f'{idx}.joblib'))
print('---')
