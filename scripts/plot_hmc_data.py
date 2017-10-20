#!/usr/bin/python

# Imports
import pickle
import sys, os
import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
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
data_dir='../data/y_alpha_KmLH'
results_dir='../results/y_alpha_KmLH'
os.makedirs(results_dir, exist_ok=True)

# Trick
alpha_ranges=np.linspace(0.75, 0.8, 11)
KmLH_ranges=np.linspace(300, 800, 101)
lh_f=np.zeros((alpha_ranges.size, KmLH_ranges.size))

# For all files
for filename in os.listdir(data_dir):
    data_file=data_dir+'/'+filename
    if not os.path.isdir(data_file) and not data_file.endswith('.pdf'):
    
        # State
        #if filename.startswith('x_'):
        # All
        #if filename.startswith('x_') or filename.startswith('y_'):
        # Observations
        observation_labels=['LH', 'FSH', 'E2', 'P4', 'Ih']
        if filename.startswith('y_'):
            print(data_file)
            y=np.loadtxt(data_file, delimiter=',')

            # TODO: figure out true time stamps from data
            t=np.arange(y.shape[1])

            # FFT of signal
            Y = np.fft.fft(y,n_fft)
            Y_shifted=np.fft.fftshift(Y,axes=1)

            # Max frequencies
            #print(np.argsort(np.abs(Y[:,:int(n_fft/2)]), axis=1)[:,-1:0:-1][:,:15])
            # LH
            max_f_lh=np.argsort(np.abs(Y[0,:int(n_fft/2)])**2)[-1:0:-1][:15]
            lh_f[alpha_ranges==float(data_file.split('_')[-3]),KmLH_ranges==float(data_file.split('_')[-1])]=np.amin(max_f_lh[(max_f_lh>20) & (max_f_lh<40)])
            
            # Plotting
            plot_save=results_dir+'/'+filename
            '''
            # State plotting
            for yd in np.arange(y.shape[0]):
                #plt.plot(t, y[yd,:], my_colors[yd], label='y[{}]'.format(yd))
                plt.plot(t, y[yd,:], my_colors[yd], label=observation_labels[yd])

            plt.legend(loc='best')
            plt.xlabel('t')
            plt.grid()
            # Show or save
            if plot_save is None: 
                plt.show()
            else:
                plt.savefig(plot_save+'.pdf', format='pdf', bbox_inches='tight')
                plt.close()
                                
            # State FFT plotting within range
            f, axarr = plt.subplots(1,2)
            for yd in np.arange(y.shape[0]):
                # Remember to shift
                #axarr[0].plot(f_fft[f_range_idx], np.abs(Y_shifted[yd,f_range_idx]), my_colors[yd], label='y[{}]'.format(yd))
                #axarr[1].plot(f_fft[f_range_idx], np.angle(Y_shifted[yd,f_range_idx]), my_colors[yd], label='y[{}]'.format(yd))
                axarr[0].plot(f_fft[f_range_idx], np.abs(Y_shifted[yd,f_range_idx]), my_colors[yd], label=observation_labels[yd])
                axarr[1].plot(f_fft[f_range_idx], np.angle(Y_shifted[yd,f_range_idx]), my_colors[yd], label=observation_labels[yd])                        

            plt.legend(loc='best')
            plt.xlabel('f')
            plt.grid()
            # Show or save
            if plot_save is None: 
                plt.show()
            else:
                plt.savefig(plot_save+'_fft_fmax{}.pdf'.format(f_plot), format='pdf', bbox_inches='tight')
                plt.close()
            
            '''

# Heatmap          
fig, ax = plt.subplots(1)
cmap=ax.pcolormesh(lh_f, cmap='inferno')
fig.colorbar(cmap)
ax.set_yticks(np.arange(alpha_ranges.size))
ax.set_yticklabels(alpha_ranges.astype(str).tolist())
ax.set_xticks(np.arange(KmLH_ranges.size))
ax.set_xticklabels(KmLH_ranges.astype(str).tolist())
plt.ylabel(r'$\alpha$')
plt.xlabel(r'KmLH')
plt.title('Period heatmap')
if plot_save is None: 
    plt.show()
else:
    plt.savefig(results_dir+'/period.pdf', format='pdf', bbox_inches='tight')
    plt.close()
