#!/usr/bin/python

# Imports
import pickle
import sys, os
import argparse
import pdb
import numpy as np
import pyGPs
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib import colors

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
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    if plot_save is None:
        plt.show()
    else:
        plt.savefig(plot_save+'prediction.pdf', format='pdf', bbox_inches='tight')
        plt.close()

def plot_GP_predict_y_sqr(model, t_train, y, plot_save):    
    # Squared Relative Error Plotting
    plt.figure()
    plt.axvspan(0, t_train, facecolor='g', alpha=0.3)
    plt.plot(np.arange(y.size), np.power((y-model.fm[:,0])/y,2), 'r', label='Prediction SrE')
    plt.xlabel('t')
    plt.ylabel(r'$(\frac{(y_t-\hat{y}_t)}{y_t})^2$')
    plt.xlim([0, y.size])
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    if plot_save is None:
        plt.show()
    else:
        plt.savefig(plot_save+'SrE.pdf', format='pdf', bbox_inches='tight')
        plt.close()

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
    
def main(data_file, t_init, t_train, sampling_rate, sigma_factor, d_x, gp_cov_f, opt_restarts):

    # Load data from file
    true_y=np.loadtxt(data_file, delimiter=',')
    # Forget initial data points
    true_y=true_y[:,t_init:]
    # Determine dimensionalities
    y_d,t_max=true_y.shape

    # Allocate space for input features
    x=np.zeros((y_d+1, t_max))

    # Allocate space for results
    dir_string='../results/{}/{}'.format(data_file.split('/')[-1], '_'.join(gp_cov_f))
    os.makedirs(dir_string, exist_ok=True)

    # GP optimization parameters
    opt_params={'method':'Minimize', 'n_restarts':opt_restarts}

    # Input/output info
    x_idxs=np.arange(y_d+1).astype(int)
    y_idxs=np.arange(y_d).astype(int)

    # Add Gaussian noise to true observations
    y=true_y+sigma_factor*true_y.std(axis=1,keepdims=True)*np.random.randn(y_d,t_max)
    # Input
    x[0,:]=np.arange(t_max) # First row is time
    # Rest is data
    x[1:,:]=y

    # GP mean and covariance, with hyperparams
    f_m = pyGPs.mean.Const() # Mean function
    
    # Periodicity kernel
    if 'periodic' in gp_cov_f or 'periodic_2' in gp_cov_f or 'periodic_3' in gp_cov_f or 'periodic_4' in gp_cov_f:
        T=30    # Initial periodicity
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
            f_k += pyGPs.cov.RBFard(D=d_x)
            
        if 'RQard' in gp_cov_f:
            f_k += pyGPs.cov.RQard(D=d_x)
    else:
        # ARD based kernels
        if 'RBFard' in gp_cov_f:
            f_k = pyGPs.cov.RBFard(D=d_x)
            
        if 'RQard' in gp_cov_f:
            f_k = pyGPs.cov.RQard(D=d_x)
            
        if 'SM' in gp_cov_f:
            raise ValueError('SM is not implemented yet, as optimization errors occur')
            '''
            T=30    # Initial periodicity
            Q = 4 # Number of mixtures
            weights=np.ones(Q)
            periods=np.linspace(0.01,T,Q)
            length_scales=1*np.ones(Q)
            hyp = np.array([ np.log(weights), np.log(1/periods), np.log(np.sqrt(length_scales))])
            #f_k = pyGPs.cov.SM(Q, hyp.flatten().tolist())
            f_k = pyGPs.cov.SM(Q, D=d_x)
            '''
            
    # GP input/output
    for y_idx in y_idxs:
        for x_idx in combinations(np.arange(x_idxs.max()+1),d_x):                    
            print('y{}_x{}_train{}_rate{}_sigmaf{}'.format(y_idx+1, ''.join(map(str, x_idx)), t_train, sampling_rate, sigma_factor))
            
            # Fit to training set
            fitted_GP=fit_GP(f_m, f_k, opt_params, x[np.array(x_idx),0:t_train:sampling_rate].T, y[y_idx,0:t_train:sampling_rate])
            
            # If model is fitted
            if fitted_GP:
                # Prediction
                fitted_GP.predict(x[np.array(x_idx),:].T)
                                        
                # Save model
                save_name=dir_string+'/y{}_x{}_train{}_rate{}_sigmaf{}_'.format(y_idx+1, ''.join(map(str, x_idx)), t_train, sampling_rate, sigma_factor)
                with open(save_name+'gp.pickle', 'wb') as f:
                    pickle.dump(fitted_GP, f)

                # Plotting
                plot_GP_predict_y(fitted_GP, np.arange(0,t_train,sampling_rate), y[y_idx,0:t_train:sampling_rate], true_y[y_idx,:], save_name)
                plot_GP_predict_y_sqr(fitted_GP, t_train, true_y[y_idx,:], save_name)
                plot_GP_predict_peaks(fitted_GP, t_train, true_y[y_idx,:], save_name)
                
            else:
                print('ERROR when fitting GP for y{}_x{}_train{}_rate{}_sigmaf{}'.format(y_idx+1, ''.join(map(str, x_idx)), t_train, sampling_rate, sigma_factor))                    
        
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example: python3 -m pdb predict_hmc_gp.py -data_file ../data/y_alpha_KmLH/y_clark_y_init_normal_t250_yscale_1_alpha_0.77_KmLH_530 -t_init 100 -t_train 40 -sampling_rate 2 -sigma_factor 0.1 -d_x 1 -gp_cov_f periodic -opt_restarts 5
    parser = argparse.ArgumentParser(description='Gaussian process for hormonal menstrual cycle prediction')
    parser.add_argument('-data_file', type=str, default=None, help='Data file to process')
    parser.add_argument('-t_init', type=int, default=100, help='Initial time-instants to skip')
    parser.add_argument('-t_train', type=int, default=1, help='Training data to use')
    parser.add_argument('-sampling_rate', type=int, default=1, help='Data sampling rate')
    parser.add_argument('-sigma_factor', type=float, default=0., help='Sigma factor of added noise')
    parser.add_argument('-d_x', type=int, default=1, help='Dimensionality of input')
    parser.add_argument('-gp_cov_f', nargs='+', type=str, default=None, help='Type of covariance function to use')
    parser.add_argument('-opt_restarts', type=int, default=10, help='Number of optimization tries')
    
    # Get arguments
    args = parser.parse_args()
    
    # Make sure file exists
    assert os.path.isfile(args.data_file), 'Data file could not be found'

    # Call main function
    main(args.data_file, args.t_init, args.t_train, args.sampling_rate, args.sigma_factor, args.d_x, args.gp_cov_f, args.opt_restarts)

