#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import sys, os, re
import argparse
from itertools import *
import pdb

# Main code
def main(exec_machine, data_file, t_init, sampling_type, sampling_peak, opt_restarts, R):
  
    ########## Possibilities ##########
    # For each of the possible models
    python_scripts=[]
    # Evaluate over parameter set
    # Noise
    sigma_factors=np.array([0,0.001,0.01,0.1])
    
    # Sampling rates
    sampling_rates=np.array([1,2,4,7], dtype=int)
    
    # Number of training points
    # Based on peaks of data file
    true_y=np.loadtxt(data_file, delimiter=',')
    # Forget initial data points
    true_y=true_y[:,t_init:]
    # Determine dimensionalities
    y_d,t_max=true_y.shape
    # TODO: improve heuristic for period separators
    peak_y, peak_t=np.where(true_y>true_y.mean(axis=1,keepdims=True)+true_y.std(axis=1,keepdims=True))
    diff_peak_t=np.diff(peak_t)
    max_diff_peak_t=diff_peak_t[diff_peak_t>1].max()
    t_trains=np.arange(max_diff_peak_t,t_max,max_diff_peak_t)[1:3] # Containing first 3 periods
    
    for t_train in t_trains:
        for sampling_rate in sampling_rates:
            for sigma_factor in sigma_factors:
                # TODO: take this dirty trick out
                #for r in np.arange(R):
                    gp_cov_f='periodic RQard'
                    python_scripts+=['../src/predict_hmc_gp_xtime.py -data_file {} -t_init {} -t_train {} -sampling_type {} -sampling_rate {} -sampling_peak {} -sigma_factor {} -gp_cov_f {} -opt_restarts {} -R {}'.format(data_file, t_init, t_train, sampling_type, sampling_rate, sampling_peak, sigma_factor, gp_cov_f, opt_restarts, R)]
                    #gp_cov_f='periodic_2 RQard'
                    #python_scripts+=['../src/predict_hmc_gp_xtime.py -data_file {} -t_init {} -t_train {} -sampling_type {} -sampling_rate {} -sampling_peak {} -sigma_factor {} -gp_cov_f {} -opt_restarts {} -R {}'.format(data_file, t_init, t_train, sampling_type, sampling_rate, sampling_peak, sigma_factor, gp_cov_f, opt_restarts, R)]
                            
    # Python script
    for (idx, python_script) in enumerate(python_scripts):
        job_name='job_{}_{}_{}'.format(idx, python_script.split()[0].split('/')[-1].split('.')[0], str.replace(''.join(python_script.split('-')[2:]), ' ', '_'))
        # Execute
        print('Executing {}'.format(python_script))

        if exec_machine=='laptop':
            os.system('python3 {}'.format(python_script))
        else:
            print('Only local execution available')
            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example: python3 -m pdb execute_predict_hmc_gp_xtime.py -data_file ../data/y_alpha_KmLH/y_clark_y_init_normal_t250_yscale_1_alpha_0.77_KmLH_580 -t_init 100 -sampling_type uniform -sampling_peak peak -opt_restarts 10 -R 25
    parser = argparse.ArgumentParser(description='Gaussian process for hormonal menstrual cycle prediction')
    parser.add_argument('-exec_machine', type=str, default='laptop', help='Where to run the simulation')
    parser.add_argument('-data_file', type=str, default=None, help='Data file to process')
    parser.add_argument('-t_init', type=int, default=100, help='Initial time-instants to skip')
    parser.add_argument('-sampling_type', type=str, default='uniform', help='Uniform or Random sampling')
    parser.add_argument('-sampling_peak', type=str, default='nopeak', help='Whether to force to hae peak samples')
    parser.add_argument('-opt_restarts', type=int, default=50, help='Number of optimization tries')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    
    # Get arguments
    args = parser.parse_args()
    
    # Make sure file exists
    assert os.path.isfile(args.data_file), 'Data file could not be found'

    # Call main function
    main(args.exec_machine, args.data_file, args.t_init, args.sampling_type, args.sampling_peak, args.opt_restarts, args.R)

