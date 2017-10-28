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

# Plotting colors
my_colors=[colors.cnames['black'], colors.cnames['skyblue'], colors.cnames['cyan'], colors.cnames['blue'], colors.cnames['palegreen'], colors.cnames['lime'], colors.cnames['green'], colors.cnames['yellow'], colors.cnames['orange'], colors.cnames['red'], colors.cnames['purple'], colors.cnames['fuchsia'], colors.cnames['pink'], colors.cnames['saddlebrown'], colors.cnames['chocolate'], colors.cnames['burlywood']]

def main(exec_machine, data_file, t_init, R):

    ########## General conf ##########
    observation_labels=['$LH$', '$FSH$', '$E_2$', '$P_4$', '$I_h$']
    observation_colors=[colors.cnames['skyblue'], colors.cnames['blue'], colors.cnames['orange'], colors.cnames['green'], colors.cnames['red']]

    # Results folders
    results_path='../results'

    ########## Executed options ##########
    # Evaluate over parameter set
    # TODO: is now copied from execute_predict_hmc_gp_xtime.py
    sampling_types=np.array(['uniform', 'random'])
    sampling_peaks=np.array(['nopeak', 'peak'])
    # Noise
    sigma_factors=np.array([0,0.01,0.1])
    # Sampling rates
    sampling_rates=np.array([1,2,4,6,10,15], dtype=int)

    # Number of training points
    # Based on peaks of data file
    true_y=np.loadtxt(data_file, delimiter=',')
    # Forget initial data points
    true_y=true_y[:,t_init:]
    # Determine dimensionalities
    y_d,t_max=true_y.shape

    # PEAKS
    true_peak_idxs=true_y>true_y.mean(axis=1,keepdims=True)+true_y.std(axis=1,keepdims=True)
    peak_y, peak_t=np.where(true_peak_idxs)
    diff_peak_t=np.diff(peak_t)
    max_diff_peak_t=diff_peak_t[diff_peak_t>1].max()
    # TODO: improve heuristic for period separators
    p_separators=np.arange(0,t_max,max_diff_peak_t)
    true_period_peaks=[]
    for y_idx in np.arange(y_d).astype(int):
        period_peaks=[]
        for p_idx in np.arange(p_separators.size-1):
            period_peaks.append(peak_t[peak_y==y_idx][(p_separators[p_idx]<peak_t[peak_y==y_idx])&(peak_t[peak_y==y_idx]<p_separators[p_idx+1])])
        true_period_peaks.append(period_peaks)
    
    # VALLEYS
    true_valley_idxs=true_y<true_y.mean(axis=1,keepdims=True)-true_y.std(axis=1,keepdims=True)
    valley_y, valley_t=np.where(true_valley_idxs)
    true_period_valleys=[]
    for y_idx in np.arange(y_d).astype(int):
        period_valleys=[]
        for p_idx in np.arange(p_separators.size-1):
            period_valleys.append(valley_t[valley_y==y_idx][(p_separators[p_idx]<valley_t[valley_y==y_idx])&(valley_t[valley_y==y_idx]<p_separators[p_idx+1])])
        true_period_valleys.append(period_valleys)
        
    # Train and test times
    t_trains=np.arange(max_diff_peak_t,t_max,max_diff_peak_t)[:3] # Containing first 3 periods
    t_tests=np.arange(max_diff_peak_t,t_max,max_diff_peak_t)[1:5] # Test on next 2 periods

    # Overall results
    overall_results={
        'se': np.NaN*np.ones((sampling_types.size, sampling_peaks.size, t_trains.size, sampling_rates.size, sigma_factors.size, R, y_d, t_max)) ,
        'peak_true_positives':np.NaN*np.ones((sampling_types.size, sampling_peaks.size, t_trains.size, sampling_rates.size, sigma_factors.size, R,  y_d,p_separators.size-1)) ,
        'peak_false_negatives':np.NaN*np.ones((sampling_types.size, sampling_peaks.size, t_trains.size, sampling_rates.size, sigma_factors.size, R,  y_d,p_separators.size-1)) ,
        'peak_false_positives':np.NaN*np.ones((sampling_types.size, sampling_peaks.size, t_trains.size, sampling_rates.size, sigma_factors.size, R,  y_d,p_separators.size-1)) ,
        'peak_est_start_diff':np.NaN*np.ones((sampling_types.size, sampling_peaks.size, t_trains.size, sampling_rates.size, sigma_factors.size, R,  y_d,p_separators.size-1)) ,
        'peak_est_duration_diff':np.NaN*np.ones((sampling_types.size, sampling_peaks.size, t_trains.size, sampling_rates.size, sigma_factors.size, R,  y_d,p_separators.size-1)) ,
        'valley_true_positives':np.NaN*np.ones((sampling_types.size, sampling_peaks.size, t_trains.size, sampling_rates.size, sigma_factors.size, R,  y_d,p_separators.size-1)) ,
        'valley_false_negatives':np.NaN*np.ones((sampling_types.size, sampling_peaks.size, t_trains.size, sampling_rates.size, sigma_factors.size, R,  y_d,p_separators.size-1)) ,
        'valley_false_positives':np.NaN*np.ones((sampling_types.size, sampling_peaks.size, t_trains.size, sampling_rates.size, sigma_factors.size, R,  y_d,p_separators.size-1)) ,
        'valley_est_start_diff':np.NaN*np.ones((sampling_types.size, sampling_peaks.size, t_trains.size, sampling_rates.size, sigma_factors.size, R,  y_d,p_separators.size-1)) ,
        'valley_est_duration_diff':np.NaN*np.ones((sampling_types.size, sampling_peaks.size, t_trains.size, sampling_rates.size, sigma_factors.size, R,  y_d,p_separators.size-1))
        }

    # Results data
    results_data=results_path+'/'+data_file.split('/')[-1]
        
    # For all executed set ups
    for result_name in os.listdir(results_data):
        results_dir=results_data+'/'+result_name

        if os.path.isdir(results_dir):
            # For all results within dir
            for filename in sorted(os.listdir(results_dir)):
                data_file=results_dir+'/'+filename
                if not os.path.isdir(data_file) and data_file.endswith('.pickle'):
                    #print(filename)
                    
                    # Load GP pickle
                    with open(data_file, 'rb') as f:
                        fitted_GP=pickle.load(f)
                    
                    # Identify parameter set
                    splitted_filename=filename.split('_')
                    sampling_type_idx=np.where(sampling_types==splitted_filename[2])[0][0]
                    sampling_peak_idx=np.where(sampling_peaks==splitted_filename[4])[0][0]
                    train_idx=np.where(t_trains==int(splitted_filename[1].split('train')[1]))[0][0]
                    rate_idx=np.where(sampling_rates==int(splitted_filename[3].split('rate')[1]))[0][0]
                    sigmaf_idx=np.where(sigma_factors==float(splitted_filename[5].split('sigmaf')[1]))[0][0]
                    r_idx=int(splitted_filename[6].split('.')[0].split('r')[1])
                    
                    # Process                    
                    for y_idx in np.arange(y_d).astype(int):
                        # If GP was fitted
                        if fitted_GP[y_idx]:    
                            # error over time
                            overall_results['se'][sampling_type_idx, sampling_peak_idx,train_idx, rate_idx, sigmaf_idx, r_idx, y_idx, :]=np.power((true_y[y_idx]-fitted_GP[y_idx].fm[:,0]),2)
                            
                            # peaks
                            est_peak_idxs=(fitted_GP[y_idx].fm[:,0]>fitted_GP[y_idx].fm[:,0].mean()+fitted_GP[y_idx].fm[:,0].std())
                            est_peaks=np.where(est_peak_idxs)[0]
                            # valleys
                            est_valley_idxs=(fitted_GP[y_idx].fm[:,0]<fitted_GP[y_idx].fm[:,0].mean()-fitted_GP[y_idx].fm[:,0].std())
                            est_valleys=np.where(est_valley_idxs)[0]
                            # For each interval
                            for p_idx in np.arange(p_separators.size-1):
                                # PEAKS
                                est_period_peaks=est_peaks[(p_separators[p_idx]<est_peaks)&(est_peaks<p_separators[p_idx+1])]
                                # True/false positives/negatives
                                overall_results['peak_true_positives'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx, p_idx]=((true_peak_idxs[y_idx, p_separators[p_idx]:p_separators[p_idx+1]].astype(int)-est_peak_idxs[p_separators[p_idx]:p_separators[p_idx+1]].astype(int))==0)[true_peak_idxs[y_idx,p_separators[p_idx]:p_separators[p_idx+1]]].sum()
                                overall_results['peak_false_negatives'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx, p_idx]=((true_peak_idxs[y_idx,p_separators[p_idx]:p_separators[p_idx+1]].astype(int)-est_peak_idxs[p_separators[p_idx]:p_separators[p_idx+1]].astype(int))==1).sum()
                                overall_results['peak_false_positives'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx,p_idx]=((true_peak_idxs[y_idx,p_separators[p_idx]:p_separators[p_idx+1]].astype(int)-est_peak_idxs[p_separators[p_idx]:p_separators[p_idx+1]].astype(int))==-1).sum()
                                # Time differences
                                overall_results['peak_est_duration_diff'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx,p_idx]=len(true_period_peaks[y_idx][p_idx])-len(est_period_peaks)
                                if len(est_period_peaks)>0:
                                    overall_results['peak_est_start_diff'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx,p_idx]=true_period_peaks[y_idx][p_idx][0]-est_period_peaks[0]
                                else:
                                    overall_results['peak_est_start_diff'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx, p_idx]=true_period_peaks[y_idx][p_idx][0]
                                    
                                    
                                # VALLEYS
                                if len(true_period_valleys[y_idx][p_idx])>0:
                                    # Only if valley in true
                                    est_period_valleys=est_valleys[(p_separators[p_idx]<est_valleys)&(est_valleys<p_separators[p_idx+1])]
                                    # True/false positives/negatives
                                    overall_results['valley_true_positives'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx, p_idx]=((true_valley_idxs[y_idx, p_separators[p_idx]:p_separators[p_idx+1]].astype(int)-est_valley_idxs[p_separators[p_idx]:p_separators[p_idx+1]].astype(int))==0)[true_valley_idxs[y_idx,p_separators[p_idx]:p_separators[p_idx+1]]].sum()
                                    overall_results['valley_false_negatives'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx, p_idx]=((true_valley_idxs[y_idx,p_separators[p_idx]:p_separators[p_idx+1]].astype(int)-est_valley_idxs[p_separators[p_idx]:p_separators[p_idx+1]].astype(int))==1).sum()
                                    overall_results['valley_false_positives'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx,p_idx]=((true_valley_idxs[y_idx,p_separators[p_idx]:p_separators[p_idx+1]].astype(int)-est_valley_idxs[p_separators[p_idx]:p_separators[p_idx+1]].astype(int))==-1).sum()
                                    # Time differences
                                    overall_results['valley_est_duration_diff'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx,p_idx]=len(true_period_valleys[y_idx][p_idx])-len(est_period_valleys)
                                    if len(est_period_valleys)>0:
                                        overall_results['valley_est_start_diff'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx,p_idx]=true_period_valleys[y_idx][p_idx][0]-est_period_valleys[0]
                                    else:
                                        overall_results['valley_est_start_diff'][sampling_type_idx, sampling_peak_idx, train_idx, rate_idx, sigmaf_idx, r_idx, y_idx, p_idx]=true_period_valleys[y_idx][p_idx][0]
                                    
            # Save overall results
            with open(results_data+'/overall_results_{}_tinit{}.pickle'.format(result_name, t_init), 'wb') as f:
                pickle.dump(overall_results, f)
            
            # Results: per noise
            for y_idx in np.arange(y_d).astype(int):
                for sigma_idx,sigma_factor in enumerate(sigma_factors):
                    for train_idx,t_train in enumerate(t_trains):
                        test_idx=train_idx+1
                        t_test=t_tests[train_idx]
                        for sampling_idx,sampling_rate in enumerate(sampling_rates):
                            for sampling_type_idx, sampling_type in enumerate(sampling_types):
                                for sampling_peak_idx, sampling_peak in enumerate(sampling_peaks):
                                    # MSE
                                    print('{} {} {} y{} sigma_factor{} MSE: {} {} {}'.format(result_name, sampling_type, sampling_peak, y_idx+1,sigma_factor, train_idx, sampling_rate, np.nanmean(overall_results['se'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, t_train:t_test],axis=(0,1)) ))
                            
                    for train_idx,t_train in enumerate(t_trains):
                        test_idx=train_idx+1
                        t_test=t_tests[train_idx]
                        for sampling_idx,sampling_rate in enumerate(sampling_rates):
                            for sampling_type_idx, sampling_type in enumerate(sampling_types):
                                for sampling_peak_idx, sampling_peak in enumerate(sampling_peaks):
                                    # True_positives
                                    print('{} {} {} y{} sigma_factor{} PEAKS: {} {} {}/{} {}'.format(result_name, sampling_type, sampling_peak, y_idx+1,sigma_factor, train_idx, sampling_rate, np.nanmean(overall_results['peak_true_positives'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx]), np.nanmean(overall_results['peak_true_positives'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx]) + np.nanmean(overall_results['peak_false_negatives'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx]) , np.nanmean(overall_results['peak_false_positives'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx]) ))
                            
                    for train_idx,t_train in enumerate(t_trains):
                        test_idx=train_idx+1
                        t_test=t_tests[train_idx]
                        for sampling_idx,sampling_rate in enumerate(sampling_rates):
                            for sampling_type_idx, sampling_type in enumerate(sampling_types):
                                for sampling_peak_idx, sampling_peak in enumerate(sampling_peaks):
                                    # Time differences
                                    print('{} {} {} y{} sigma_factor{} PEAK_DIFFS: {} {} {} {}'.format(result_name, sampling_type, sampling_peak, y_idx+1,sigma_factor, train_idx, sampling_rate, np.nanmean(np.abs(overall_results['peak_est_duration_diff'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx])), np.nanmean(np.abs(overall_results['peak_est_start_diff'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx])) ))

                    if y_idx==2:
                        for train_idx,t_train in enumerate(t_trains):
                            test_idx=train_idx+1
                            t_test=t_tests[train_idx]
                            for sampling_idx,sampling_rate in enumerate(sampling_rates):
                                for sampling_type_idx, sampling_type in enumerate(sampling_types):
                                    for sampling_peak_idx, sampling_peak in enumerate(sampling_peaks):
                                        # True_positives
                                        print('{} {} {} y{} sigma_factor{} VALLEYS: {} {} {}/{} {} '.format(result_name, sampling_type, sampling_peak, y_idx+1,sigma_factor, train_idx, sampling_rate, np.nanmean(overall_results['valley_true_positives'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx]), np.nanmean(overall_results['valley_true_positives'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx]) + np.nanmean(overall_results['valley_false_negatives'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx]) , np.nanmean(overall_results['valley_false_positives'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx]) ))

                        for train_idx,t_train in enumerate(t_trains):
                            test_idx=train_idx+1
                            t_test=t_tests[train_idx]
                            for sampling_idx,sampling_rate in enumerate(sampling_rates):
                                for sampling_type_idx, sampling_type in enumerate(sampling_types):
                                    for sampling_peak_idx, sampling_peak in enumerate(sampling_peaks):
                                        # Time differences
                                        print('{} {} {} y{} sigma_factor{} VALLEY_DIFFS: {} {} {} {}'.format(result_name, sampling_type, sampling_peak, y_idx+1,sigma_factor, train_idx, sampling_rate, np.nanmean(np.abs(overall_results['valley_est_duration_diff'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx])), np.nanmean(np.abs(overall_results['valley_est_start_diff'][sampling_type_idx, sampling_peak_idx, train_idx, sampling_idx, sigma_idx, :, y_idx, test_idx])) ))
                    
                            
            #pdb.set_trace()      
        
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example: python3 -m pdb evaluate_predict_hmc_gp_xtime.py -exec_machine habanero -data_file ../data/y_alpha_KmLH/y_clark_y_init_normal_t500_yscale_1_alpha_0.77_KmLH_580 -t_init 250 -sampling_type uniform -sampling_peak nopeak -R 25
    parser = argparse.ArgumentParser(description='Gaussian process for hormonal menstrual cycle prediction')
    parser.add_argument('-exec_machine', type=str, default='laptop', help='Where to run the simulation')
    parser.add_argument('-data_file', type=str, default=None, help='Data file to process')
    parser.add_argument('-t_init', type=int, default=100, help='Initial time-instants to skip')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    
    # Get arguments
    args = parser.parse_args()
    
    # Make sure file exists
    assert os.path.isfile(args.data_file), 'Data file could not be found'

    # Call main function
    main(args.exec_machine, args.data_file, args.t_init, args.R)
                           

