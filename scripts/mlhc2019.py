import torch
import gpytorch
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Integer
from scipy.signal import resample, find_peaks
import sys
import os
import pdb

# New code
sys.path.append('../src/MGP_DCNN')
from data_processing import * # Some auxiliary functions
from MGP import * # MGP models
from TNN import * # TNN models
from RNN  import * # RNN models
from expected_distance  import * # Expected distance implementation

############################### DATA #####################################
# Import the data :
# observation_labels=['LH', 'FSH', 'E2', 'P4', 'Ih']
data_file = '../data/mlhc_2019/realistic_dataset.pickle'
with open(data_file, 'rb') as f:
    y_data=pickle.load(f)

y_data_shape = y_data.shape
print(y_data_shape)

# Random shuffling of individuals (so that each execution is different)
permutation_index = np.random.permutation(np.arange(y_data_shape[0]))
y_data=y_data[permutation_index]

############################### HYPERPARAMETERS #####################################
main_dir='../results/mlhc_2019'
os.makedirs(main_dir, exist_ok=True)

# Random number for each realization
realization=np.random.randn()

################## Data related
# Set all the hyperparameters
nb_input_tasks = y_data_shape[-1]   #number of tasks selected for the input and output of the MGP = input DCNN
nb_output_tasks = y_data_shape[-1]  #number of tasks selected for the output of the DCNN

nb_individuals = 60    #number of individuals
nb_individuals_train = 40  #number of individuals for the training of the DCNN
nb_individuals_val = 10   #number of individuals for the validation of the DCNN
nb_individuals_test = 10  #number of individuals for the test of the DCNN
assert(nb_individuals_train + nb_individuals_val + nb_individuals_test == nb_individuals)

nb_time_steps = 105       #number of time steps
nb_train_time_steps = 70  #time limit for the training (limit between reconstruction and forecast)

# number of posterior samples we want to drawn from each individual MGP distribution
nb_samples_per_id = 1000
nb_samples_per_id = 100

# Execution summary
exec_type='total_time_{}_train_time_{}_nb_train_{}_nb_val_{}_nb_test_{}_nb_samples_per_id_{}'.format(nb_time_steps, nb_train_time_steps, nb_individuals_train, nb_individuals_val, nb_individuals_test, nb_samples_per_id)
if not os.path.exists('{}/{}'.format(main_dir, exec_type)):
    os.makedirs('{}/{}'.format(main_dir,exec_type), exist_ok=True)
if not os.path.exists('{}/{}/trained_models'.format(main_dir, exec_type)):
    os.makedirs('{}/{}/trained_models'.format(main_dir, exec_type), exist_ok=True)

################## Model related
# LSTM parameters
lstm_batch_size = 50
lstm_epochs = 500
lstm_dropout_frac = 0.0

# Parameters for training the MGP
mgp_n_iter = 500               # max of iterations for the training of the MGP
mgp_learning_rate = 0.015  # learning rate for the trainig of the MGP
mgp_time_kernel = gpytorch.kernels.PeriodicKernel(period_length_prior = gpytorch.priors.NormalPrior(0.31,0.1))

# DCNN parameters
dcnn_batch_size=50
dcnn_nb_epochs=500
dcnn_learning_rate=2e-3
dcnn_nb_hidden_layers=5
dcnn_nb_filters=8
dcnn_regularizer_coef=1e-7
dcnn_kernel_size=6
dcnn_dilation_factor=2

# Bayesian Optimization of DCNN
bo_nb_calls = 10
bo_nb_random_starts = 5
# Set potential parameter ranges
bo_learning_rate = Real(low=0.5e-3, high=2e-2, prior='log-uniform', name='learning_rate')
bo_nb_hidden_layers = Integer(low=3, high=6, name='nb_layers')
bo_nb_filters = Integer(low=5, high=12, name='nb_filters')
bo_regularizer_coef = Real(low=1e-8, high=1e-3, prior='log-uniform', name='regularizer_coef')
bo_kernel_size = Integer(low=2, high=9, name='kernel_size')
bo_dilation_factor = Integer(low=1, high=5, name='kernel_size')
bo_parameters_range = [bo_learning_rate,bo_nb_hidden_layers,bo_nb_filters,bo_regularizer_coef, bo_kernel_size, bo_dilation_factor]
bo_default_parameters = [dcnn_learning_rate, dcnn_nb_hidden_layers, dcnn_nb_filters, dcnn_regularizer_coef, dcnn_kernel_size, dcnn_dilation_factor]


############################### DATA PREPROCESSING #####################################
# Data aligned by peaks
y_data = align_data_on_peak(y_data, length=nb_time_steps, column=0)
# Subset of data to consider
y_data = y_data[:nb_individuals, :nb_time_steps]
y_data_shape = y_data.shape

# Resample data for ED
resampled_data = resample_data_for_ed(y_data)
resampled_data_shape=resampled_data.shape

# Scale data per user and hormone
resampled_scaler=StandardScaler()
resampled_data_train=resampled_scaler.fit_transform(resampled_data[:nb_individuals_train].reshape(-1, resampled_data_shape[-1]))
print(resampled_data_train.mean(axis=(0,1)), resampled_data_train.std(axis=(0,1)))
resampled_data = resampled_scaler.transform(resampled_data.reshape(-1,resampled_data_shape[-1])).reshape(resampled_data_shape)
print(resampled_data.mean(axis=(0,1)), resampled_data.std(axis=(0,1)))

y_scaler=StandardScaler()
y_data_train=y_scaler.fit_transform(y_data[:nb_individuals_train].reshape(-1, y_data_shape[-1]))
print(y_data_train.mean(axis=(0,1)), y_data_train.std(axis=(0,1)))
y_data = y_scaler.transform(y_data.reshape(-1,y_data_shape[-1])).reshape(y_data_shape)
print(y_data.mean(axis=(0,1)), y_data.std(axis=(0,1)))

############################### EXECUTION WITH #####################################
# Execution for different number of selected points
nb_peaks_selected = 2     # select the 2 first peaks of the first hormone
# And number of points selected for training
nb_selected_points_all=np.array([10, 15, 25, 35, 50, 70])

# For ED-based model
ed_sampling_list = np.array([])
if ed_sampling_list.size==0:
    # Need to compute list
    # Based on which MGP
    ed_mgp_block_indices =[[0,1],[2,3,4]]
    # Make sure only training data is shared with ED computation
    ed_sampling_list = compute_ed_sampling_list(resampled_data[:nb_individuals_train], nb_train_time_steps, nb_selected_points_all.max(), ed_mgp_block_indices, mgp_time_kernel, mgp_learning_rate, mgp_n_iter, mgp_smart_end=True, mgp_plot=False)

assert(ed_sampling_list.size>nb_selected_points_all.max(), 'Not enought sampling points for list of nb_selected_points')
print('ED sampling list={}'.format(ed_sampling_list))
# This ED sampling list
with open('{}/{}/ed_sampling_list_{}.pickle'.format(main_dir, exec_type, realization), 'wb') as f:
    pickle.dump(ed_sampling_list, f)

# for nb_selected_points in np.array([15]):
for nb_selected_points in nb_selected_points_all:    

    ########## Random sampling
    # Select a set of training points
    train_sampling_type='random_sampling_nb_selected_points_{}_nb_peaks_selected_{}_{}'.format(nb_selected_points, nb_peaks_selected, realization)
    subset_of_points=select_subset_of_points(y_data[:,:nb_train_time_steps], nb_selected_points, nb_peaks_selected, sample_list=None)
    # This random sampling list
    with open('{}/{}/random_sampling_list_{}.pickle'.format(main_dir, exec_type, train_sampling_type), 'wb') as f:
        pickle.dump(subset_of_points, f)
    
    ########### LSTM with random sampling ##################
    lstm_model = RNN_time_prediction(x_length = nb_time_steps, y_length =nb_time_steps, nb_sensors = nb_output_tasks, dropout_ratio = lstm_dropout_frac)
    lstm_callbacks_list = [
        ModelCheckpoint(
            filepath='{}/{}/trained_models/LSTM_{}.h5'.format(main_dir,exec_type, train_sampling_type),
            monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='acc', patience=100)
        ]
    
    # Prepare input data with zeros when no true sample is available
    x_data = np.zeros_like(y_data)
    x_data[np.arange(nb_individuals)[:,None],subset_of_points] = y_data[np.arange(nb_individuals)[:,None],subset_of_points]
    
    x_train, y_train = x_data[:nb_individuals_train], y_data[:nb_individuals_train, :nb_time_steps]
    x_val, y_val = x_data[nb_individuals_train:nb_individuals_train + nb_individuals_val], y_data[nb_individuals_train:nb_individuals_train + nb_individuals_val,:nb_time_steps]
    x_test, y_test = x_data[nb_individuals_train + nb_individuals_val:], y_data[nb_individuals_train + nb_individuals_val:,:nb_time_steps]

    lstm_score = lstm_model.fit(x_train,y_train, batch_size=lstm_batch_size, epochs=lstm_epochs, callbacks=lstm_callbacks_list, validation_data=(x_val, y_val), verbose=1)
    lstm_model.load_weights('{}/{}/trained_models/LSTM_{}.h5'.format(main_dir,exec_type, train_sampling_type))
    lstm_y_predicted = lstm_model.predict(x_data)
    lstm_y_mse=(y_data-lstm_y_predicted)**2
    # Save LSTM MSE info
    with open('{}/{}/LSTM_{}_y.pickle'.format(main_dir,exec_type, train_sampling_type), 'wb') as f:
        pickle.dump(lstm_y_predicted, f)
    with open('{}/{}/LSTM_{}_y_mse.pickle'.format(main_dir,exec_type, train_sampling_type), 'wb') as f:
        pickle.dump(lstm_y_mse, f)
    
    print('LSTM with random sampling with {} points selected for training, TEST LOSS={}'.format(nb_selected_points, lstm_y_mse[nb_individuals_train + nb_individuals_val:].mean()))
    ########################################################
    
    ########### LSTM trained at population level, tested with random sampling ##################
    '''
    lstm_population_model = RNN_time_prediction(x_length = nb_time_steps, y_length =nb_time_steps, nb_sensors = nb_output_tasks, dropout_ratio = lstm_dropout_frac)
    lstm_population_callbacks_list = [
        ModelCheckpoint(
            filepath='{}/{}/trained_models/LSTMpopulation_{}.h5'.format(main_dir,exec_type, train_sampling_type),
            monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='acc', patience=100)
        ]
    
    # Prepare input TEST data with zeros when no true sample is available
    x_data = np.zeros_like(y_data)
    x_data[np.arange(nb_individuals)[:,None],subset_of_points] = y_data[np.arange(nb_individuals)[:,None],subset_of_points]
    
    x_train, y_train = y_data[:nb_individuals_train, :nb_time_steps], y_data[:nb_individuals_train, :nb_time_steps]
    x_val, y_val = y_data[nb_individuals_train:nb_individuals_train + nb_individuals_val,:nb_time_steps], y_data[nb_individuals_train:nb_individuals_train + nb_individuals_val,:nb_time_steps]
    x_test, y_test = x_data[nb_individuals_train + nb_individuals_val:], y_data[nb_individuals_train + nb_individuals_val:,:nb_time_steps]

    lstm_population_score = lstm_population_model.fit(x_train,y_train, batch_size=lstm_batch_size, epochs=lstm_epochs, callbacks=lstm_callbacks_list, validation_data=(x_val, y_val), verbose=1)
    lstm_population_model.load_weights('{}/{}/trained_models/LSTMpopulation_{}.h5'.format(main_dir,exec_type, train_sampling_type))
    lstm_population_y_predicted = lstm_population_model.predict(x_data)
    lstm_population_y_mse=(y_data-lstm_population_y_predicted)**2
    # Save LSTM MSE info
    with open('{}/{}/LSTMpopulation_{}_y.pickle'.format(main_dir,exec_type, train_sampling_type), 'wb') as f:
        pickle.dump(lstm_population_y_predicted, f)
    with open('{}/{}/LSTMpopulation_{}_y_mse.pickle'.format(main_dir,exec_type, train_sampling_type), 'wb') as f:
        pickle.dump(lstm_population_y_mse, f)
    
    print('LSTM population with TEST random sampling with {} points selected for training, TEST LOSS={}'.format(nb_selected_points, lstm_population_y_mse[nb_individuals_train + nb_individuals_val:].mean()))
    '''
    ########################################################
        
    ########### MGPs with random sampling ##################
    all_block_indices = [np.arange(nb_input_tasks).reshape(nb_input_tasks,1).tolist(), # Independent GPs
                        [np.arange(nb_input_tasks).tolist()],  #Full MGP: 
                        [[0,1],[2,3,4]],  # Blockwise MGP
                        ]
    
    #for block_indices in [[[0,1],[2,3,4]]]:
    for block_indices in all_block_indices:

        assert(np.concatenate(block_indices).shape[0] == nb_input_tasks)

        # Train, test split in time!
        train_x, train_y, test_x, test_y, _ = prepare_data_before_GP(y_data,
                                                                    block_indices = block_indices,
                                                                    nb_time_steps = nb_time_steps,
                                                                    nb_train_time_steps = nb_train_time_steps,
                                                                    nb_train_individuals = nb_individuals_train,
                                                                    scaler=None)  # Data is already scaled

        # **Note** that the x values (time) are scaled ! Indeed they are in the range [0,1]    
        # **Note** that the test set is just an extended time period of the train set.
        # We must keep in mind that the test array will feed the DCNN after.
        # We use here 'train' and 'test' for the delimitation in time between the reconstruction and the prediction.
        # It has nothing to do with the individuals used for the training and the test... (Remember that the MGP is trained on the individual level : same behavior for every individual)

        # Train MGP-s per individual, learned model is saved as an h5 file in directory outputGP
        h5_dataset_path, y_predicted = train_Block_MGP_multiple_individuals(train_x, train_y,
                                                                            test_x, test_y,
                                                                            block_indices,
                                                                            kernel=mgp_time_kernel,
                                                                            learning_rate=mgp_learning_rate,
                                                                            n_iter=mgp_n_iter,
                                                                            train_sample_subset = subset_of_points,
                                                                            main_dir=main_dir,
                                                                            exec_type=exec_type,
                                                                            train_sampling_type=train_sampling_type,
                                                                            activate_plot=False,
                                                                            smart_end=True)
        # Evaluate losses
        mgp_y_mse=(test_y.numpy()-y_predicted)**2
        # Save MSE info
        with open('{}/{}/MGP{}blocks_{}_y.pickle'.format(main_dir, exec_type, len(block_indices), train_sampling_type), 'wb') as f:
            pickle.dump(y_predicted, f)
        with open('{}/{}/MGP{}blocks_{}_y_mse.pickle'.format(main_dir, exec_type, len(block_indices), train_sampling_type), 'wb') as f:
            pickle.dump(mgp_y_mse, f)
        
        print('MGP with block indices {} and random sampling with {} points selected for training, TEST LOSS={}'.format(block_indices, nb_selected_points, mgp_y_mse[nb_individuals_train + nb_individuals_val:].mean()))
    ########################################################
        
    ########### MGP+DCNN with random sampling ##################
    # Make sure we use the appropriate MGP
    block_indices=[[0,1],[2,3,4]]
    # and use it to get output samples
    x_train, y_train, x_val, y_val, x_test, y_test = import_and_split_data_train_val_test(output_gp_path = h5_dataset_path,
                                                                                            y_true = y_data,
                                                                                            block_indices= block_indices,
                                                                                            nb_timesteps = nb_time_steps,
                                                                                            nb_tasks = nb_input_tasks,
                                                                                            nb_individuals = nb_individuals,
                                                                                            nb_individuals_train = nb_individuals_train,
                                                                                            nb_individuals_val = nb_individuals_val,
                                                                                            nb_individuals_test = nb_individuals_test,
                                                                                            nb_samples_per_id=nb_samples_per_id,
                                                                                            plot_some_posteriors = False)

    # ** Note that the dataset contains nb_samples_per_id consecutive posteriors for each same individual
    # Build the Neural Network
    network = Time_Neural_Network('CNN', dcnn_batch_size, dcnn_nb_epochs,
                                    x_train=x_train, y_train=y_train,
                                    x_val=x_val, y_val=y_val,
                                    x_test=x_test, y_test=y_test,
                                    main_dir=main_dir,
                                    exec_type=exec_type,
                                    train_sampling_type=train_sampling_type)
    
    network.build_TNN(learning_rate=dcnn_learning_rate, 
                      nb_hidden_layers=dcnn_nb_hidden_layers, 
                      nb_filters=dcnn_nb_filters,
                      regularizer_coef=dcnn_regularizer_coef, 
                      kernel_size=dcnn_kernel_size, 
                      dilation_factor=dcnn_dilation_factor, 
                      display_summary=False)

    # Train with Bayesian Optimization in validation set
    search_result = network.optimization_process(bo_parameters_range,
                                 default_parameters=bo_default_parameters,
                                 nb_calls = bo_nb_calls,
                                 nb_random_starts = bo_nb_random_starts,
                                 plot_conv=False)

    # Pick best network architecture
    index_min_val_loss = np.argmin(search_result.func_vals)
    best_network=network.params_history[index_min_val_loss]
    # Best predictions
    dcnn_predicted=network.y_predicted[index_min_val_loss]
    # Evaluate losses
    dcnn_y_mse=(np.concatenate((y_train,y_val,y_test),axis=0)-dcnn_predicted)**2
    # Since we have nb_samples_per_id
    dcnn_y_mse_reshaped=np.reshape(dcnn_y_mse, (nb_samples_per_id,nb_individuals,nb_time_steps,nb_input_tasks), order='F')
    # Recompute per individual
    dcnn_y_mse=dcnn_y_mse_reshaped.mean(axis=0)
    # Save MSE info
    with open('{}/{}/{}_{}_y.pickle'.format(main_dir, exec_type, network.model_type, network.train_sampling_type), 'wb') as f:
        pickle.dump(dcnn_predicted, f)
    with open('{}/{}/{}_{}_y_mse.pickle'.format(main_dir, exec_type, network.model_type, network.train_sampling_type), 'wb') as f:
        pickle.dump(dcnn_y_mse, f)
    
    print('MGP+DCNN with block indices {} and random sampling with {} points selected for training, TEST LOSS={}'.format(block_indices, nb_selected_points, dcnn_y_mse[(nb_individuals_train + nb_individuals_val):].mean()))
    ########################################################
        
    ########### ED Sampling
    train_sampling_type='ed_sampling_nb_selected_points_{}_nb_peaks_selected_{}_{}'.format(nb_selected_points, nb_peaks_selected, realization)
        
    ########### MGP with ED sampling ##################
    # Pick the MGP we want (the one used when computing the ED sampling points)
    block_indices=[[0,1],[2,3,4]]
    # In resampled_data we have exactly 3 cycles, we can train the CNN with the same commands
    train_x, train_y, test_x, test_y, _ = prepare_data_before_GP(resampled_data,
                                                                block_indices = block_indices,
                                                                nb_time_steps = nb_time_steps,
                                                                nb_train_time_steps = nb_train_time_steps,
                                                                nb_train_individuals = nb_individuals_train,
                                                                scaler=None) # Has been already scaled

    # Train MGP-s per individual (with same MGP structure as used for determining the ED scheme)
    h5_dataset_path, y_predicted = train_Block_MGP_multiple_individuals(train_x, train_y,
                                                                        test_x, test_y,
                                                                        block_indices,
                                                                        kernel=mgp_time_kernel,
                                                                        learning_rate=mgp_learning_rate,
                                                                        n_iter=mgp_n_iter,
                                                                        train_sample_subset = ed_sampling_list[:nb_selected_points],
                                                                        main_dir=main_dir,
                                                                        exec_type=exec_type,
                                                                        train_sampling_type=train_sampling_type,
                                                                        activate_plot=False,
                                                                        smart_end=True)

    # Evaluate losses:
    ed_mgp_y_mse=(test_y.numpy()-y_predicted)**2
    # Save MSE info
    with open('{}/{}/MGP{}blocks_{}_y.pickle'.format(main_dir, exec_type, len(block_indices), train_sampling_type), 'wb') as f:
        pickle.dump(y_predicted, f)
    with open('{}/{}/MGP{}blocks_{}_y_mse.pickle'.format(main_dir, exec_type, len(block_indices), train_sampling_type), 'wb') as f:
        pickle.dump(ed_mgp_y_mse, f)
    
    print('MGP with block indices {} and ED sampling with {} points selected for training, TEST LOSS={}'.format(block_indices, nb_selected_points, ed_mgp_y_mse[nb_individuals_train + nb_individuals_val:].mean()))
    ########################################################
        
    ########### MGP+DCNN with ED sampling ##################
    # Now get output samples from the MGPs
    x_train, y_train, x_val, y_val, x_test, y_test = import_and_split_data_train_val_test(output_gp_path = h5_dataset_path,
                                                                                            y_true = resampled_data,
                                                                                            block_indices= block_indices,
                                                                                            nb_timesteps = nb_time_steps,
                                                                                            nb_tasks = nb_input_tasks,
                                                                                            nb_individuals = nb_individuals,
                                                                                            nb_individuals_train = nb_individuals_train,
                                                                                            nb_individuals_val = nb_individuals_val,
                                                                                            nb_individuals_test = nb_individuals_test,
                                                                                            nb_samples_per_id=nb_samples_per_id,
                                                                                            plot_some_posteriors = False)
    
    # Define the DCNN
    network = Time_Neural_Network('CNN', dcnn_batch_size, dcnn_nb_epochs,
                                    x_train=x_train, y_train=y_train,
                                    x_val=x_val, y_val=y_val,
                                    x_test=x_test, y_test=y_test,
                                    main_dir=main_dir,
                                    exec_type=exec_type,
                                    train_sampling_type=train_sampling_type)
    # And optimize architecture
    search_result = network.optimization_process(bo_parameters_range,
                                 default_parameters=bo_default_parameters,
                                 nb_calls = bo_nb_calls,
                                 nb_random_starts = bo_nb_random_starts,
                                 plot_conv=False)

    # Pick best network architecture
    index_min_val_loss = np.argmin(search_result.func_vals)
    best_network=network.params_history[index_min_val_loss]
    # Best predictions
    ed_dcnn_predicted=network.y_predicted[index_min_val_loss]
    # Evaluate losses
    ed_dcnn_y_mse=(np.concatenate((y_train,y_val,y_test),axis=0)-ed_dcnn_predicted)**2
    # Since we have nb_samples_per_id
    ed_dcnn_y_mse_reshaped=np.reshape(ed_dcnn_y_mse, (nb_samples_per_id,nb_individuals,nb_time_steps,nb_input_tasks), order='F')
    # Recompute per individual
    ed_dcnn_y_mse=ed_dcnn_y_mse_reshaped.mean(axis=0)
    # Save MSE info
    with open('{}/{}/{}_{}_y.pickle'.format(main_dir, exec_type, network.model_type, network.train_sampling_type), 'wb') as f:
        pickle.dump(ed_dcnn_predicted, f)
    with open('{}/{}/{}_{}_y_mse.pickle'.format(main_dir, exec_type, network.model_type, network.train_sampling_type), 'wb') as f:
        pickle.dump(ed_dcnn_y_mse, f)
    
    print('MGP+DCNN with block indices {} and ED sampling with {} points selected for training, TEST LOSS={}'.format(block_indices, nb_selected_points, ed_dcnn_y_mse[(nb_individuals_train + nb_individuals_val):].mean()))
    ########################################################

