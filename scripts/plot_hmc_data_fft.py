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

# TODO: figure out true sampling rate from data
f_sampling=1.
# Number of FFT points
n_fft=512
f_fft = np.fft.fftshift(np.fft.fftfreq(n_fft))*(n_fft/f_sampling)

# Plotting f range
f_plot=40   # 40 days
f_range=np.arange(-f_plot, f_plot) #
f_range_idx=(f_fft<=f_plot) & (f_fft>=-f_plot)

# Load dde simulation data
data_dir='../data'
data_dir='../data/init_y_perturbations'
data_dir='../data/init_y_scaling'
#data_dir='../data/alpha_range'
#data_dir='../data/KmLH_range'
for filename in os.listdir(data_dir):
    data_file=data_dir+'/'+filename
    if not os.path.isdir(data_file) and not data_file.endswith('.pdf'):
        print(data_file)
        y=np.loadtxt(data_file, delimiter=',')

        # FFT of signal (not shifted yet)
        Y = np.fft.fft(y,n_fft)
        
        # State FFT plotting
        Y_shifted=np.fft.fftshift(Y,axes=1)
        f, axarr = plt.subplots(1,2)
        for yd in np.arange(y.shape[0]):
            # Remember to shift
            axarr[0].plot(f_fft, np.abs(Y_shifted[yd,:]), my_colors[yd], label='y[{}]'.format(yd))
            axarr[1].plot(f_fft, np.angle(Y_shifted[yd,:]), my_colors[yd], label='y[{}]'.format(yd))            

        plt.legend(loc='best')
        plt.xlabel('f')
        plt.grid()
        # Show or save
        plot_save=data_file
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(data_file+'_fft.pdf', format='pdf', bbox_inches='tight')
            plt.close()
            
        # State FFT plotting within range
        f, axarr = plt.subplots(1,2)
        for yd in np.arange(y.shape[0]):
            # Remember to shift
            axarr[0].plot(f_fft[f_range_idx], np.abs(Y_shifted[yd,f_range_idx]), my_colors[yd], label='y[{}]'.format(yd))
            axarr[1].plot(f_fft[f_range_idx], np.angle(Y_shifted[yd,f_range_idx]), my_colors[yd], label='y[{}]'.format(yd))            

        plt.legend(loc='best')
        plt.xlabel('f')
        plt.grid()
        # Show or save
        plot_save=data_file
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(data_file+'_fft_fmax{}.pdf'.format(f_plot), format='pdf', bbox_inches='tight')
            plt.close()
            
        # Max frequencies
        print(np.argsort(np.abs(Y[:,:int(n_fft/2)]), axis=1)[:,-1:0:-1][:,:10])