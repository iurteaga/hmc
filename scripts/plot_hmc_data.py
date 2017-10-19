#!/usr/bin/python

# Imports
import pickle
import sys, os
import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import colors
# Plotting colors
my_colors=[colors.cnames['black'], colors.cnames['skyblue'], colors.cnames['cyan'], colors.cnames['blue'], colors.cnames['palegreen'], colors.cnames['lime'], colors.cnames['green'], colors.cnames['yellow'], colors.cnames['orange'], colors.cnames['red'], colors.cnames['purple'], colors.cnames['fuchsia'], colors.cnames['pink'], colors.cnames['saddlebrown'], colors.cnames['chocolate'], colors.cnames['burlywood']]

# Load dde simulation data
data_dir='../data'
data_dir='../data/init_y_perturbations'
data_dir='../data/init_y_scaling'
data_dir='../data/alpha_range'
data_dir='../data/KmLH_range'
for filename in os.listdir(data_dir):
    data_file=data_dir+'/'+filename
    if not os.path.isdir(data_file) and not data_file.endswith('.pdf'):
        print(data_file)
        y=np.loadtxt(data_file, delimiter=',')

        # TODO: figure out true time stamps from data
        t=np.arange(y.shape[1])

        # State plotting
        for yd in np.arange(y.shape[0]):
            plt.plot(t, y[yd,:], my_colors[yd], label='y[{}]'.format(yd))

        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        # Show or save
        plot_save=data_file
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(data_file+'.pdf', format='pdf', bbox_inches='tight')
            plt.close()
