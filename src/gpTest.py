#!/usr/bin/python

# Imports
import pickle
import sys, os
import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pyGPs

data_file='../data/y_alpha_KmLH/y_clark_y_init_normal_t150_yscale_1_alpha_0.77_KmLH_530'
true_y=np.loadtxt(data_file, delimiter=',')
y_d,t_max=true_y.shape
t=np.arange(t_max)

sigma=np.ones((y_d,1))
sigma=np.array([10,5,1,1,1])[:,None]
sigma_factor=0.1
sigma=sigma_factor*true_y.std(axis=1,keepdims=True)
noisy_y=true_y+sigma*np.random.randn(y_d,t_max)

yd=0
t_train=np.arange(50)
t_test=np.arange(50,t_max)

model = pyGPs.GPR()      # specify model (GP regression)
m = pyGPs.mean.Const()
k = pyGPs.cov.Periodic(log_ell=0.0, log_p=3.4, log_sigma=0.0)
model.setPrior(mean=m, kernel=k)
model.covfunc.hyp
model.getPosterior(t_train, noisy_y[yd,t_train]) # fit default model (mean zero & rbf kernel) with data
model.optimize(t_train, noisy_y[yd,t_train])     # optimize hyperparamters (default optimizer: single run minimize)
model.covfunc.hyp
ym, ys2, fm, fs2, lp = model.predict(t_test)
model.plot()

pdb.set_trace()

model = pyGPs.GPR()      # specify model (GP regression)
m = pyGPs.mean.Const()
D = 1 # x \in Real^D
Q = 3 # Number of mixtures
weights=np.ones(Q)
periods=np.linspace(0.01,30,Q)
length_scales=1*np.ones(Q)
hyp = np.array([ np.log(weights), np.log(1/periods), np.log(np.sqrt(length_scales))])
k = pyGPs.cov.SM(Q, hyp.flatten().tolist())
model.setPrior(mean=m, kernel=k)
model.covfunc.hyp
model.getPosterior(t_train, noisy_y[yd,t_train]) # fit default model (mean zero & rbf kernel) with data
model.optimize(t_train, noisy_y[yd,t_train])     # optimize hyperparamters (default optimizer: single run minimize)
model.covfunc.hyp
ym, ys2, fm, fs2, lp = model.predict(t_test)
model.plot()

pdb.set_trace()


model = pyGPs.GPR()      # specify model (GP regression)
m = pyGPs.mean.Const()
D = 1 # x \in Real^D
Q = 3 # Number of mixtures
weights=np.ones(Q)
periods=np.ones(Q)
length_scales=1*np.ones(Q)
hyp = np.array([ np.log(weights), np.log(1/periods), np.log(np.sqrt(length_scales))])
k = pyGPs.cov.SM(Q, hyp.flatten().tolist())
model.setPrior(mean=m, kernel=k)
model.covfunc.hyp
model.getPosterior(t, noisy_y[yd,t]) # fit default model (mean zero & rbf kernel) with data
model.optimize(t, noisy_y[yd,t])     # optimize hyperparamters (default optimizer: single run minimize)
model.covfunc.hyp
ym, ys2, fm, fs2, lp = model.predict(t_test)
model.plot()

