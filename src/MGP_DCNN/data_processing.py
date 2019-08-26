import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from scipy.signal import find_peaks
import pdb

def align_data_on_peak(data, length, column=0, plot=False):
    '''
    :param data: a numpy array of size [number_id, number time_step, nb_hormones]
    :param column ; the column on which the peaks have to be aligned (we look at the peak in the first days)
    :return: an array with the same axis but the time series are aligned with the peak of the first hormonal level

    BE CAREFUL : SHAPE OF THE ARRAY HAS CHANGED ! WE SELCT ONLY length points once the data have been aligned ! (still a cube of data)
    '''

    output = []
    for i in range(data.shape[0]):
        index = np.argmax(data[i,:40,column])
        output.append(data[None,i, index:index+length, :])

    output = np.vstack(output)

    return output

def select_subset_of_points(y_data_train, nb_selected_points, nb_peaks_selected, sample_list=None):
    # Returns an array with indexes of the selected points within the training time
    
    # In general
    if sample_list==None:
        # Randomly select
        random_permutation=np.random.permutation(np.arange(y_data_train.shape[1]))
        if nb_selected_points != None:
            if nb_peaks_selected != None:
                # Peaks need to be selected per-individual
                subset_of_points_per_individual=np.ones((y_data_train.shape[0],1))*random_permutation
                for i in range(y_data_train.shape[0]):
                    ### scipy.find_peaks doesn't detect the peaks at the extremes, so we add values 
                    curve_peaks = np.concatenate((np.array([-100]), y_data_train[i, :40, 0], np.array([-100])))
                    idx_peaks,_ = find_peaks(curve_peaks, distance = 20)
                    selected_peak_idx = idx_peaks[:nb_peaks_selected] - 1
                    #print(selected_peak_idx)
                    # Join peaks with non-random            
                    random_not_peak=np.setdiff1d(random_permutation,selected_peak_idx, assume_unique=True)
                    subset_of_points_per_individual[i]=np.concatenate((selected_peak_idx,random_not_peak))
                
                # Keep only nb_selected_points
                subset_of_points = np.sort(subset_of_points_per_individual[:,:nb_selected_points], axis=1)
            elif nb_peaks_selected == None:
                # Fully random
                subset_of_points = np.sort(random_permutation[:nb_selected_points])
            else:
                raise ValueError('Error with NO sample_list and nb_selected_points={}'.format(nb_selected_points))
        elif nb_selected_points == None:
            # Then select all
            subset_of_points=np.arange(y_data_train.shape[1])
        else:
            raise ValueError('Error with NO sample_list and nb_selected_points={}'.format(nb_selected_points))
            
    elif sample_list!=None:
        subset_of_points = np.sort(sample_list[:nb_selected_points])
    else:
        raise ValueError('Error when sample_list={}'.format(sample_list))

    return subset_of_points.astype(int)
    
def prepare_data_before_GP(y_data, block_indices, nb_time_steps, nb_train_time_steps, nb_train_individuals, scaler=None):

    block_indices_flat = np.array([item for sublist in block_indices for item in sublist])
    y_data = y_data[:, :, block_indices_flat]
    nb_gp_tasks = len(block_indices_flat)

    if scaler != None:
        '''  Scale the data  '''
        y_train_data = y_data[:nb_train_individuals]

        shape_before_scaling = y_data.shape
        y_train_data = scaler.fit_transform(y_train_data.reshape(-1, nb_gp_tasks))
        y_data = scaler.transform(y_data.reshape(-1, nb_gp_tasks)).reshape(shape_before_scaling)

    x_train = np.arange(nb_train_time_steps) / nb_time_steps
    x_test = np.arange(nb_time_steps) / nb_time_steps
    y_train = y_data[:, :nb_train_time_steps]
    y_test = y_data[:, :nb_time_steps]

    train_x = torch.tensor(x_train).float()
    train_y = torch.tensor(y_train).float()
    test_x = torch.tensor(x_test).float()
    test_y = torch.tensor(y_test).float()

    return train_x, train_y, test_x, test_y, scaler




def generate_samples_single_id(gp_mean, gp_covar_matrix, num_samples):
    '''
    :param gp_out_mean: mean of the posterior over time and tasks for a single individual
    :param covar_matrix: covariance matrix of the GP over time and tasks for a single individual
    :param num_samples; the amount of samples per individual we want to generate
    :return: the multiples samples drawn from the posteriors distribution in an array of shape [num_samples, time_steps*num_tasks]

    BE CAREFUL : THE OUTPUT IS IN THE FORM OF [[time_1_task_1, time_1_task_2, ..., time_2,task_1, time_2_task_2, ...] * num_samples]
    '''

    gp_mean = np.tile(gp_mean.reshape(-1, 1), (1,num_samples))
    len_covar = gp_covar_matrix.shape[0]
    cholesky_matrix = np.linalg.cholesky(gp_covar_matrix + 8e-4*np.eye(len_covar))  #we add this quantity to be sure that we have a positive semi definite matrix
    posterior_samples = gp_mean + np.dot(cholesky_matrix, np.random.normal(size=(len_covar, num_samples)))
    return posterior_samples


def reorganize_samples_single_id(samples_array, nb_tasks):
    '''
    :param samples_array: array generated by the generate_samples_single_id() function
    BE CAREFUL INPUT TYPE [[time_1_task_1_sample_1, time_1_task_1_sample_2], [time_1_task_2_sample_1, time_1_task_2_sample_2],
                                                 [time_2,task_1_sample_1, time_2_task_1_sample_2], ....]
    :param nb_tasks = number of samples per individual
    return : the samples array but yet of shape [number of samples, number of time_steps, number of hormones]

    '''

    assert(samples_array.shape[0]%nb_tasks == 0)

    if (samples_array.shape[0]%nb_tasks != 0):
        raise ValueError('Invalid input shape')

    out = samples_array.reshape(samples_array.shape[0]//nb_tasks, nb_tasks, samples_array.shape[-1])
    out = np.swapaxes(out, 0, -1)
    out = np.swapaxes(out, 1, -1)
    return out


def generate_posterior_samples(h5_name, nb_samples_per_id):
    '''
    :param h5_data_path: take as input a h5 file with many datasets inside( y_data, y_mean, y_covar_1, y_covar_2, y_std, x_train, filters)
     = results of the MGP on all the women
    '''

    print('\n ---- GENERATE MULTIPLE SAMPLES FROM POSTERIOR DISTRIBUTION ----\n')

    with h5py.File('output_MGP/'+h5_name, 'r') as data:
        mean = data['mean_array'][:]
        covar_matrix = data['covar_matrix_array'][:]

    output = np.empty(shape=(mean.shape[0], nb_samples_per_id) + (mean.shape[1:]))
    (nb_individuals, nb_time_steps, nb_tasks) = mean.shape

    for i in range(nb_individuals):
        print('Individual %d/%d'%(i+1,nb_individuals))
        samples = generate_samples_single_id(mean[i], covar_matrix[i], nb_samples_per_id)
        samples = reorganize_samples_single_id(samples, nb_tasks=nb_tasks)
        output[i] = samples

    print(output.shape)
    return output

def generate_samples_single_id_with_covar_matrix_2(gp_out_mean, covar_matrix, num_samples):
    return np.random.multivariate_normal(gp_out_mean, covar_matrix, num_samples)


def change_representation_covariance_matrix(covar_matrix, num_tasks):

    ''' FROM [[time_1_task_1, time_1_task_2, time_2_task_1, time_1_task_2, ...]
        TO [[time_1_task_1, time_2_task_1,...],[time_1_task_2, time_1_task_2]]

    ... or the inverse (works both ways)
    '''
    len_covar = covar_matrix.shape[0]
    new_index = np.array([np.arange(i, len_covar, num_tasks) for i in range(num_tasks)]).reshape(-1)
    new_index = np.array([[i, j] for i in new_index for j in new_index]).reshape(len_covar, len_covar, 2)
    a = np.zeros((len_covar, len_covar))
    for i in range(len_covar):
        for j in range(len_covar):
            a[i, j] = covar_matrix[new_index[i, j, 0], new_index[i, j, 1]]
    return a

def generate_samples_single_id_with_covar_matrix(gp_out_mean, covar_matrix, num_samples):
    '''
    :param gp_out_mean: mean of the posterior over the all x range
    :param covar_matrix: covariance matrix of the GP over the all x range
    :param num_tasks:
    :param num_samples; the amount of samples per id we want
    :return: an array of shape [num_samples, time_steps*num_tasks]

    BE CAREFUL : THE OUTPUT IS OF THE FORM [[time_1_task_1, time_1_task_2, time_2,task_1, time_2_task_2, ...] * num_samples]
    '''
    gp_mean = np.tile(gp_out_mean.reshape(-1, 1), (1,num_samples))
    len_covar = covar_matrix.shape[0]
    cholesky_matrix = np.linalg.cholesky(covar_matrix + 9e-4*np.eye(len_covar))  #we add this quantity to be sure to have a positive semi definite matrix
    posterior_samples = gp_mean + np.dot(cholesky_matrix, np.random.normal(size=(len_covar, num_samples)))
    return posterior_samples


def from_samples_vector_to_final_array(vector, num_tasks):
    '''
    :param vector: BE CAREFUL INPUT TYPE [[time_1_task_1_sample_1, time_1_task_1_sample_2], [time_1_task_2_sample_1, time_1_task_2_sample_2],
                                         [time_2,task_1_sample_1, time_2_task_1_sample_2], ....]

    :param num_tasks = number of hormonal levels for one GP
    return : an array of shape [number of samples, number of time_steps, number of hormones]

    BE CAREFUL : THE FORM OF THE INPUT
    '''

    assert(vector.shape[0]%num_tasks == 0)
    out = vector.reshape(vector.shape[0]//num_tasks, num_tasks, vector.shape[-1])
    out = np.swapaxes(out, 0, -1)
    out = np.swapaxes(out, 1, -1)
    return out



def import_and_split_data_train_val_test(output_gp_path, y_true, block_indices, nb_timesteps, nb_tasks,
                                         nb_individuals, nb_individuals_train, nb_individuals_val, nb_individuals_test,
                                         nb_samples_per_id=1, plot_some_posteriors= False):

    y_gp_mean = []
    y_gp_covar_matrix = []

    ########################################   IMPORT DATA ##########################################

    with h5py.File(output_gp_path, 'r') as data:

        for i in range(len(block_indices)):
            y_mean_cur = data['mean_block_%d' % i][:]
            y_covar_cur = data['covar_block_%d' % i][:]

            if len(block_indices[i]) == 1:
                y_mean_cur = y_mean_cur[..., None]

            y_gp_mean.append(y_mean_cur)
            y_gp_covar_matrix.append(y_covar_cur)

    #################################################################################################



    ############          GENERATE MULTIPLE SAMPLES for each MGP distribtion AND SPLIT       ###########

    '''  Build the index to split the data  '''
    #new_index_individuals = np.random.permutation(np.arange(nb_individuals))
    new_index_individuals = np.arange(nb_individuals)   # NO reshuflling
    index_individuals_train = new_index_individuals[:nb_individuals_train]
    index_individuals_val = new_index_individuals[nb_individuals_train: nb_individuals_train + nb_individuals_val]
    index_individuals_test = new_index_individuals[nb_individuals_train + nb_individuals_val: nb_individuals_train + nb_individuals_val + nb_individuals_test]


    if nb_samples_per_id > 1 :  # if we draw multiple sample from the posterior distribution

        if nb_samples_per_id == 0:
            raise ValueError(' The number of samples (argument) has to be greater than 0 for the data augmentation')

        y_out_of_gp = np.zeros(shape=(nb_individuals, nb_samples_per_id, nb_timesteps, nb_tasks))

        for i in range(nb_individuals):
            for j in range(len(block_indices)):
                samples_cur = generate_samples_single_id_with_covar_matrix_2(y_gp_mean[j][i].reshape(-1),
                                                                             y_gp_covar_matrix[j][i],
                                                                             num_samples=nb_samples_per_id).reshape(-1,nb_timesteps,len(block_indices[j]))
                y_out_of_gp[i, :, :, np.array(block_indices[j])] = np.swapaxes(np.swapaxes(samples_cur,0,-1),1,2)


        if plot_some_posteriors==True:
            id = random.randint(0,nb_individuals-1)
            fig,ax = plt.subplots(nb_tasks, figsize=(15,9))
            for j in range(nb_tasks):
                for i in range(nb_samples_per_id//7):
                    ax[j].plot(y_out_of_gp[id, i, :, j])
            plt.show()


        '''  Split the data and Shuffle the training data   '''
        '''
        x_train = y_out_of_gp[index_individuals_train].reshape(-1, nb_timesteps, nb_tasks)
        shuffle_index_train = np.random.permutation(np.arange(nb_individuals_train * nb_samples_per_id))
        x_train = x_train[shuffle_index_train]
        y_train = np.tile(y_true[None, index_individuals_train], (nb_samples_per_id, 1, 1, 1))
        y_train = np.swapaxes(y_train, 0, 1).reshape(-1, nb_timesteps, nb_tasks)
        y_train = y_train[shuffle_index_train]
        '''
        
        '''  Split the data'''

        x_train = y_out_of_gp[index_individuals_train].reshape(-1, nb_timesteps, nb_tasks)
        y_train = np.tile(y_true[None, index_individuals_train], (nb_samples_per_id, 1, 1, 1))
        y_train = np.swapaxes(y_train, 0, 1).reshape(-1, nb_timesteps, nb_tasks)

        x_val = y_out_of_gp[index_individuals_val].reshape(-1, nb_timesteps,nb_tasks)
        y_val = np.tile(y_true[None, index_individuals_val], (nb_samples_per_id, 1, 1, 1))
        y_val = np.swapaxes(y_val, 0, 1).reshape(-1, nb_timesteps, nb_tasks)

        x_test = y_out_of_gp[index_individuals_test].reshape(-1, nb_timesteps, nb_tasks)
        y_test = np.tile(y_true[None, index_individuals_test], (nb_samples_per_id, 1, 1, 1))
        y_test = np.swapaxes(y_test, 0, 1).reshape(-1, nb_timesteps, nb_tasks)

    #########################           JUST TAKES THE MEAN AND SPLIT       ################################

    if nb_samples_per_id == 1:

        y_out_of_gp = np.empty(shape=(nb_individuals, nb_timesteps, nb_tasks))

        for i in range(nb_individuals):
            for j in range(len(block_indices)):
                print(np.swapaxes(y_gp_mean[j][i], 0, -1).shape)
                y_out_of_gp[i, :, block_indices[j]] = np.swapaxes(y_gp_mean[j][i], 0, -1)

        x_train = y_out_of_gp[index_individuals_train]
        y_train = y_true[index_individuals_train]
        x_val = y_out_of_gp[index_individuals_val]
        y_val = y_true[index_individuals_val]
        x_test = y_out_of_gp[index_individuals_test]
        y_test = y_true[index_individuals_test]


    return x_train, y_train, x_val, y_val, x_test, y_test

# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
