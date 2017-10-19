#!/usr/bin/python

# Imports
import pickle
import sys, os
import argparse
import pdb
import numpy as np
import pyGPs

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
    model.optimize(x_train,y_train)
    
    return model
    
def plot_GP_prediction(model, x, y, plot_save):

    # Predict with x
    assert x.size==y.size
    model.predict(x)
    
    # Prediction plotting
    assert y.size==model.fm.size
    plt.figure()
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
        
    # MSE Plotting
    '''
    plt.figure()
    plt.plot(np.arange(y.size), np.power(y-model.fm[:,0],2), 'r', label='Prediction MSE')
    plt.xlabel('t')
    plt.ylabel(r'$(y_t-\hat{y}_t)^2$')
    plt.xlim([0, y.size])
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    if plot_save is None:
        plt.show()
    else:
        plt.savefig(plot_save+'MSE.pdf', format='pdf', bbox_inches='tight')
        plt.close()
    '''
    
def main(data_file, t_init, gp_cov_f):

    # Load data from file
    true_y=np.loadtxt(data_file, delimiter=',')
    # Forget initial data points
    true_y=true_y[:,t_init:]
    # Determine dimensionalities
    y_d,t_max=true_y.shape

    # Allocate space for input features
    x=np.zeros((y_d+1, t_max))
    x[0,:]=np.arange(t_max) # First will be time

    # Allocate space for results
    dir_string='../results/{}/{}'.format(data_file.split('/')[-1], '_'.join(gp_cov_f))
    os.makedirs(dir_string, exist_ok=True)

    # GP optimization parameters
    opt_params={'method':'Minimize', 'n_restarts':100}

    # Evaluate over parameter set
    # Noise
    sigma_factors=np.linspace(0,1,1)
    # Sampling rates
    sampling_rates=np.array([1,2,3,4,5,6,7,10,15], dtype=int)
    # Input/output info
    x_idxs=np.arange(y_d+1).astype(int)
    y_idxs=np.arange(y_d).astype(int)
    # Number of training points
    t_trains=np.array([40,70,100], dtype=int)
    
    for sigma_factor in sigma_factors:
        # Add Gaussian noise to true observations
        y=true_y+sigma_factor*true_y.std(axis=1,keepdims=True)*np.random.randn(y_d,t_max)
        # Add y to x
        x[1:,:]=y
        for t_train in t_trains:
            for sampling_rate in sampling_rates:

                # GP mean and covariance, with hyperparams
                f_m = pyGPs.mean.Const() # Mean function
                
                # Periodicity kernel
                T=30    # Initial periodicity
                if 'periodic' in gp_cov_f:
                    f_k = pyGPs.cov.Periodic(log_p=np.log(T/sampling_rate))

                elif 'periodic_2' in gp_cov_f:
                    f_k = pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/2/sampling_rate), log_sigma=0.0)

                elif 'periodic_3' in gp_cov_f:
                    f_k = pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/2/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/4/sampling_rate), log_sigma=0.0)
                    
                elif 'periodic_4' in gp_cov_f:
                    f_k = pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/2/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/4/sampling_rate), log_sigma=0.0) + pyGPs.cov.Periodic(log_ell=0.0, log_p=np.log(T/8/sampling_rate), log_sigma=0.0)

                else:
                    raise ValueError('At least one periodic kernel is required in gp_cov_f={}'.format(gp_cov_f))
         
                # GP input/output
                for y_idx in y_idxs:
                    for x_idx in x_idxs:

                        # One input to one output
                        if 'RBFard' in gp_cov_f:
                            f_k += pyGPs.cov.RBFard(D=1)
                            
                        if 'RQard' in gp_cov_f:
                            f_k += pyGPs.cov.RQard(D=1)
                    
                        # Fit to training set
                        fitted_GP=fit_GP(f_m, f_k, opt_params, x[x_idx,0:t_train:sampling_rate], y[y_idx,0:t_train:sampling_rate])
                        # Save model
                        save_name=dir_string+'/y{}_x{}_train{}_rate{}_sigmaf{}_'.format(y_idx+1, x_idx, t_train, sampling_rate, sigma_factor)
                        with open(save_name+'gp.pickle', 'wb') as f:
                            pickle.dump(fitted_GP, f)
                        # Plotting
                        plot_GP_prediction(fitted_GP, x[x_idx,:], y[y_idx,:], save_name)
                        
                        '''
                        # Multiple input to one output
                        if 'RBFard' in gp_cov_f:
                            f_newk = pyGPs.cov.RBFard(D=x_idx+1)
                            
                        if 'RQard' in gp_cov_f:
                            f_newk = pyGPs.cov.RQard(D=x_idx+1)
                    
                        # Fit to training set
                        fitted_GP=fit_GP(f_m, f_k+f_newk, opt_params, x[:x_idx+1,0:t_train:sampling_rate].T, y[y_idx,0:t_train:sampling_rate])
                        # Save model
                        save_name=dir_string+'/y{}_x{}_train{}_rate{}_sigmaf{}_'.format(y_idx+1, ''.join(map(str, np.arange(x_idx+1).tolist())), t_train, sampling_rate, sigma_factor)
                        with open(save_name+'gp.pickle', 'wb') as f:
                            pickle.dump(fitted_GP, f)
                        # Plotting
                        plot_GP_prediction(fitted_GP, x[:x_idx+1,:].T, y[y_idx,:], save_name)
                        '''
        
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example: python3 -m pdb predict_hmc_gp.py -data_file ../data/y_alpha_KmLH/y_clark_y_init_normal_t250_yscale_1_alpha_0.77_KmLH_530 -t_init 100 -gp_cov_f periodic
    parser = argparse.ArgumentParser(description='Gaussian process for hormonal menstrual cycle prediction')
    parser.add_argument('-data_file', type=str, default=None, help='Data file to process')
    parser.add_argument('-t_init', type=int, default=100, help='Initial time-instants to skip')
    parser.add_argument('-gp_cov_f', nargs='+', type=str, default=None, help='Type of covariance function to use')

    # Get arguments
    args = parser.parse_args()
    
    # Make sure file exists
    assert os.path.isfile(args.data_file), 'Data file could not be found'

    # Call main function
    main(args.data_file, args.t_init, args.gp_cov_f)

