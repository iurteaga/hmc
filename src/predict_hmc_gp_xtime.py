#!/usr/bin/python

# Imports
import pickle
import sys, os
import argparse
import pdb
import numpy as np
from itertools import combinations
# Gaussian process module by Neumann et al. [24]
import pyGPs

import matplotlib.pyplot as plt
from matplotlib import colors

############################################################
### Source to fit a GP to provided training data and predict over test set
###     Reads true data from provided data_file
###     Processes data and fits GP based on provided script parameters
###     Saves GP object to '../results/data_file/' directory with provided parameters
############################################################

# GP fitting
def fit_GP(f_mean, f_cov, opt_params, x_train, y_train):
    # GP regression model
    model = pyGPs.GPR()
    
    # Set model prior
    model.setPrior(mean=f_mean, kernel=f_cov)

    # Set optimization parameters
    if opt_params:
        model.setOptimizer(opt_params['method'], num_restarts=opt_params['n_restarts'])

    # Fit model and optimize hyperparameters with train data
    try:
        model.optimize(x_train,y_train)
    except:
        print('GP fitting and optimization error: ', sys.exc_info())
        model=None
    
    return model

# Plotting GP prediction
def plot_GP_predict_y(model, t_train, y_train, y, plot_save):  
    # Prediction plotting
    assert y.size==model.fm.size
    plt.figure()
    plt.scatter(t_train, y_train, marker='x', facecolor='g', label='Training points')
    plt.axvspan(t_train.min(), t_train.max(), facecolor='g', alpha=0.3)
    plt.plot(np.arange(y.size), y, 'k', label='True')
    plt.plot(np.arange(model.fm.size), model.fm, 'r', label='Predicted')
    plt.fill_between(np.arange(model.fm.size), (model.fm-np.sqrt(model.fs2))[:,0], (model.fm+np.sqrt(model.fs2))[:,0],alpha=0.4, facecolor='r')
    plt.xlabel('t')
    plt.ylabel(r'$y_t$')
    plt.xlim([0, y.size])
    legend = plt.legend(loc='upper right', ncol=1, shadow=True)
    if plot_save is None:
        plt.show()
    else:
        plt.savefig(plot_save+'prediction.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Plotting GP prediction error
def plot_GP_predict_y_error(model, t_train, y, plot_save):    
    # Squared Error Plotting
    plt.figure()
    plt.axvspan(0, t_train, facecolor='g', alpha=0.3)
    plt.plot(np.arange(y.size), np.power((y-model.fm[:,0]),2), 'r', label='Prediction SrE')
    plt.xlabel('t')
    plt.ylabel(r'$(y_t-\hat{y}_t)^2$')
    plt.xlim([0, y.size])
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    if plot_save is None:
        plt.show()
    else:
        plt.savefig(plot_save+'error.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Plotting GP phase (peak and valley) prediction
def plot_GP_predict_peaks(model, t_train, y, plot_save):        
    # Peak prediction
    true_peaks=(y>y.mean()+y.std())
    est_peaks=(model.fm[:,0]>model.fm[:,0].mean()+model.fm[:,0].std())
    est_peak_dist=np.where(true_peaks)[0][:,None]-np.where(est_peaks)[0][None,:]
    true_positives=((true_peaks.astype(int)-est_peaks.astype(int))==0)[true_peaks].sum()
    false_negatives=((true_peaks.astype(int)-est_peaks.astype(int))==1).sum()
    false_positives=((true_peaks.astype(int)-est_peaks.astype(int))==-1).sum()

    # Peak Plotting
    plt.figure()
    plt.axvspan(0, t_train, facecolor='g', alpha=0.3)
    plt.plot(np.arange(y.size), true_peaks, 'k', label='True Peaks')
    plt.plot(np.arange(model.fm.size), est_peaks, 'r', label='Predicted Peaks ({}/{}) FalsePositives={}'.format(true_positives, false_negatives, false_positives))
    plt.xlabel('t')
    plt.ylabel(r'$y_{peak}$')
    plt.xlim([0, y.size])
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    if plot_save is None:
        plt.show()
    else:
        plt.savefig(plot_save+'peaks.pdf', format='pdf', bbox_inches='tight')
        plt.close()
    
# Main function
def main(data_file, t_init, t_train, sampling_type, sampling_rate, sampling_peak, sigma_factor, gp_cov_f, opt_restarts,R):

    # Load data from file
    true_y=np.loadtxt(data_file, delimiter=',')
    # Forget initial data points
    true_y=true_y[:,t_init:]
    # Determine dimensionalities
    y_d,t_max=true_y.shape
    
    if sampling_peak == 'peak':
        # Determine true peaks
        peak_y, peak_t=np.where(true_y>true_y.mean(axis=1,keepdims=True)+true_y.std(axis=1,keepdims=True))
        # TODO: improve heuristic for period separators
        diff_peak_t=np.diff(peak_t)
        max_diff_peak_t=diff_peak_t[diff_peak_t>1].max()
        p_separators=np.arange(0,t_train+max_diff_peak_t,max_diff_peak_t)
        true_peaks=[]
        for y_idx in np.arange(y_d).astype(int):
            period_peaks=[]
            for p_idx in np.arange(p_separators.size-1):
                period_peaks.append(peak_t[peak_y==y_idx][(p_separators[p_idx]<peak_t[peak_y==y_idx])&(peak_t[peak_y==y_idx]<p_separators[p_idx+1])])
            true_peaks.append(period_peaks)

    # Allocate space for input feature: time
    t=np.arange(t_max)

    # Allocate space for results
    dir_string='../results/{}/{}'.format(data_file.split('/')[-1], '_'.join(gp_cov_f))
    os.makedirs(dir_string, exist_ok=True)

    # GP optimization parameters
    opt_params={'method':'Minimize', 'n_restarts':opt_restarts}

    # GP mean and covariance, with hyperparams
    f_m = pyGPs.mean.Const() # Mean function
    
    # Periodicity kernel
    T=28    # Initial periodicity
    if 'periodic' in gp_cov_f:
        f_k = pyGPs.cov.Periodic(log_p=np.log(T/sampling_rate))

    elif 'periodic_2' in gp_cov_f:
        f_k = pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/2/sampling_rate), log_sigma=0.0)

    elif 'periodic_3' in gp_cov_f:
        f_k = pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/2/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/4/sampling_rate), log_sigma=0.0)
        
    elif 'periodic_4' in gp_cov_f:
        f_k = pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/2/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/4/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/8/sampling_rate), log_sigma=0.0)
        
    # ARD based kernels
    if 'RBFard' in gp_cov_f:
        f_k += pyGPs.cov.RBFard(D=1)
        
    if 'RQard' in gp_cov_f:
        f_k += pyGPs.cov.RQard(D=1)

    # Different realizations
    for r in np.arange(R):
        # Add Gaussian noise to true observations
        y=true_y+sigma_factor*true_y.std(axis=1,keepdims=True)*np.random.randn(y_d,t_max)
        
        # GP list for each output
        fitted_GP=[]
        y_idxs=np.arange(y_d).astype(int)    
        for y_idx in np.arange(y_d).astype(int):
            print('y{}_train{}_{}_rate{}_{}_sigmaf{}_r{}'.format(y_idx+1, t_train, sampling_type, sampling_rate, sampling_peak, sigma_factor,r))
            
            # Sampling type
            if sampling_type == 'uniform':
                t_training=np.arange(0,t_train,sampling_rate, dtype=int)
            elif sampling_type == 'random':
                t_training=np.linspace(0,t_train,int(t_train/sampling_rate), dtype=int)
            else:
                raise ValueError('Sampling type={} not implemented yet'.format(sampling_type))
            # Sampling includes peak
            if sampling_peak == 'peak':
                # Training set contains at least one peak sample
                peak_samples=np.zeros(p_separators.size-1, dtype=int)
                for p_idx in np.arange(p_separators.size-1):
                    peak_samples[p_idx]=int(true_peaks[y_idx][p_idx][np.random.randint(len(true_peaks[y_idx][p_idx]))])
                
                t_training=np.unique(np.concatenate((t_training, peak_samples)))
            
            # Fit to training set
            fitted_GP.append(fit_GP(f_m, f_k, opt_params, t[t_training], y[y_idx,t_training]))
            
            # If model is fitted
            if fitted_GP[y_idx]:
                # Prediction
                fitted_GP[y_idx].predict(t)

                # Plotting
                plot_name=dir_string+'/y{}_train{}_{}_rate{}_{}_sigmaf{}_r{}_'.format(y_idx+1, t_train, sampling_type, sampling_rate, sampling_peak, sigma_factor,r)
                #plot_GP_predict_y(fitted_GP[y_idx], t_training, y[y_idx,t_training], true_y[y_idx,:], plot_name)
                #plot_GP_predict_y_error(fitted_GP[y_idx], t_train, true_y[y_idx,:], plot_name)
                #plot_GP_predict_peaks(fitted_GP[y_idx], t_train, true_y[y_idx,:], plot_name)
                
            else:
                print('ERROR when fitting GP for y{}_train{}_{}_rate{}_{}_sigmaf{}_r{}'.format(y_idx+1, t_train, sampling_type, sampling_rate, sampling_peak, sigma_factor,r))
                
        # Save full GP
        gp_name=dir_string+'/gp_train{}_{}_rate{}_{}_sigmaf{}_r{}'.format(t_train, sampling_type, sampling_rate, sampling_peak, sigma_factor,r)
        with open(gp_name+'.pickle', 'wb') as f:
            pickle.dump(fitted_GP, f)
        
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example: python3 -m pdb predict_hmc_gp_xtime.py -data_file ../data/y_alpha_KmLH/y_clark_y_init_normal_t250_yscale_1_alpha_0.77_KmLH_530 -t_init 100 -t_train 40 -sampling_type uniform -sampling_rate 2 -sampling_peak peak -sigma_factor 0.1 -gp_cov_f periodic -opt_restarts 5 -R 2
    parser = argparse.ArgumentParser(description='Gaussian process for hormonal menstrual cycle prediction')
    parser.add_argument('-data_file', type=str, default=None, help='Data file to process')
    parser.add_argument('-t_init', type=int, default=100, help='Initial time-instants to skip')
    parser.add_argument('-t_train', type=int, default=1, help='Training data to use')
    parser.add_argument('-sampling_type', type=str, default='uniform', help='Uniform or Random sampling')
    parser.add_argument('-sampling_rate', type=int, default=1, help='Data sampling rate')
    parser.add_argument('-sampling_peak', type=str, default='nopeak', help='Whether to force to hae peak samples')
    parser.add_argument('-sigma_factor', type=float, default=0., help='Sigma factor of added noise')
    parser.add_argument('-gp_cov_f', nargs='+', type=str, default=None, help='Type of covariance function to use')
    parser.add_argument('-opt_restarts', type=int, default=10, help='Number of optimization tries')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    
    # Get arguments
    args = parser.parse_args()
    
    # Make sure file exists
    assert os.path.isfile(args.data_file), 'Data file could not be found'

    # Call main function
    main(args.data_file, args.t_init, args.t_train, args.sampling_type, args.sampling_rate, args.sampling_peak, args.sigma_factor, args.gp_cov_f, args.opt_restarts, args.R)

