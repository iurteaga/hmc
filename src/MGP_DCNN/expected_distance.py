import numpy as np
from scipy.signal import resample, find_peaks
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import gpytorch
from scipy.stats import norm
import h5py
import pdb

from MGP import Block_MGP

def resample_data_for_ed(data):
    # Get dimensionalities from data
    nb_individuals, nb_total_time_steps, nb_tasks = data.shape

    # Now we need to process/resample data to have exactly 3 cycles
    preprocessed_data=np.zeros_like(data)
    for i in range(nb_individuals):
        ### scipy.find_peaks doesn't detect the peaks at the extremes, so we add values 
        curve_peaks = np.concatenate((np.array([-100]), data[i,:,0], np.array([-100])))
        idx_peaks,_ = find_peaks(curve_peaks, distance = 20)
        selected_idx_peaks = idx_peaks - 1
        if selected_idx_peaks.size>3:
           idx_peak = selected_idx_peaks[3]
        else:
            raise ValueError('We could not find at least 3 cycles')
        #print('From these found peaks={} we select {}'.format(selected_idx_peaks,idx_peak))
        data_cur = data[i, :idx_peak]
        for j in range(nb_tasks):
            preprocessed_data[i,:,j] = resample(data_cur[:,j],nb_total_time_steps)

        # Plotting
        '''
        plt.plot(data[i, :, 0], 'b', label='True data')
        plt.plot(preprocessed_data[i,:,0], 'r', label='Resampled data')
        plt.axvline(x=idx_peak, color='g')
        legend = plt.legend(loc='upper right', ncol=1, shadow=True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.show()
        '''
        '''
        # Double-checking peaks
        curve_peaks = np.concatenate((np.array([-100]), preprocessed_data[i,:,0], np.array([-100])))
        idx_peaks,_ = find_peaks(curve_peaks, distance = 20)
        resampled_idx_peaks = idx_peaks - 1
        print('Resampled found peaks={}'.format(resampled_idx_peaks))
        '''
        
    return preprocessed_data

def expected_distance(remaining_points, gp_mean, gp_std, y_full, list_selected_points):
    ''' own acquisition function --> return the index of the next point to sample'''

    term1 = (gp_mean - y_full)
    term2 = 1 - 2*norm.cdf((y_full - gp_mean)/gp_std)
    term3 = 2*gp_std*norm.pdf((y_full - gp_mean)/gp_std)
    result_list = term1 * term2 + term3

    # If you want to consider all hormones equally, rescale!
    # scaler = MinMaxScaler()
    # result_list = scaler.fit_transform(result_list)
    # If not, the hormones with highest ED will dominate the selection process
    
    # Sum over hormones
    result_list = np.sum(result_list,axis=-1)
    # If desired, just consider one horome, e.g. FSH
    # result_list = result_list[:,0]

    remaining_points = np.array(remaining_points)
    restricted_list = np.array(result_list[remaining_points])

    # Find the index of the point with highest average ED value
    #index = np.argmax(restricted_list)

    return restricted_list
    
def compute_ed_sampling_list(data, nb_train_time_steps, nb_points_in_sampling_list, mgp_block_indices, mgp_kernel, mgp_learning_rate, mgp_n_iter, mgp_smart_end, mgp_plot=False):
    # Get dimensionalities from data
    nb_individuals, nb_total_time_steps, nb_tasks = data.shape

    # Training/testing data
    x_train = np.arange(nb_train_time_steps)/nb_total_time_steps
    x_test = np.arange(nb_total_time_steps)/nb_total_time_steps
    y_train = data[:, :nb_train_time_steps]
            
    # First and second peaks should be (approximately) at 0 and 35 after resampling, so we include them in the list
    list_selected_points = [0,35]

    # Iterate until all points in sampling list are computed
    for new_point in range(nb_points_in_sampling_list-2):
        print('-------------  NEW POINT {} out of {} -------------'.format(new_point+3, nb_points_in_sampling_list))

        # Array with expected_distance value for each individual and each time point
        expected_distance_values = - 10000 * np.ones((nb_individuals, nb_train_time_steps))
        #note, for the points that we have already sampled, the value will be unchanged                                                                        

        # Unseen remaining points
        remaining_points = list(np.setdiff1d(range(nb_train_time_steps), list_selected_points))

        # Iterate over individuals
        for j in range(nb_individuals):
            print('Point {} for individual {}'.format(new_point,j))
            restart = True
            while restart == True:
                try:
                    # Training data for this individual at already selected points (in pytorch tensor format!)
                    x_train_selected = torch.tensor(x_train[list_selected_points]).float()
                    y_train_selected = torch.tensor(y_train[j,list_selected_points]).float()

                    # Define and train MGP
                    mgp = Block_MGP(mgp_kernel, mgp_learning_rate, mgp_n_iter, mgp_block_indices)
                    mgp.build_and_train_block_models(x_train_selected, y_train_selected, mgp_smart_end)
                    # Plot if desired
                    if mgp_plot:
                        mgp.plot_model(torch.tensor(x_train).float(), torch.tensor(y_train[j]).float(), torch.tensor(x_test).float(), train_filter = this_train_sample_subset)
                    
                    # Predict for this individual
                    gp_means, _, gp_stds = mgp.test_block_model(torch.tensor(x_test).float())

                    # Compute expected_distance values at remaining points for this individual
                    expected_distance_values[j, remaining_points] = expected_distance(remaining_points, np.concatenate(gp_means, axis=-1), np.concatenate(gp_stds, axis=-1), data[j], list_selected_points)
                    restart = False
                except:
                    print('*** Retrying to fit MGP for individual {} with selected points {}'.format(j,list_selected_points))

        # Average expected_distance over individuals
        expected_distance_average = np.mean(expected_distance_values, axis = 0)
        # Select maximumd value
        index_next_point = np.argmax(expected_distance_average)
        # Append to list
        list_selected_points.append(index_next_point)

        print('#################### CURRENT ED LIST with {} points: {}'.format(len(list_selected_points), list_selected_points))

    return np.array(list_selected_points)

# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
