import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import os
import pdb

from MGP_subclasses import *
from data_processing import *

class Block_MGP():

    def __init__(self, kernel, learning_rate, n_training_iter, block_indices):

        self.kernel = kernel
        self.learning_rate = learning_rate
        self.n_training_iter = n_training_iter

        self.block_indices = block_indices
        self.number_of_block = len(block_indices)
        self.total_nb_tasks = len([item for sublist in self.block_indices for item in sublist])

        self.model = []
        self.likelihood = []
        self.loss_list = []


    def build_and_train_single_model(self, x_train, y_train, block_number=0, smart_end = False):
        '''
        :param x_train: array size nb_timesteps *1, represents time
        :param y_train: array size nb_timesteps * nb_tasks
        :param block_number: the number of the block, starts from 0
        :return: modifies the attributes model and likelihood according to the training data
        '''


        nb_tasks = y_train.shape[-1]
        if nb_tasks == 1:
            self.likelihood.append(gpytorch.likelihoods.GaussianLikelihood())
            y_train = y_train[:,0]
            self.model.append(Single_task_GP_model(x_train, y_train, self.likelihood[block_number], self.kernel))

        if nb_tasks>1: #if no model has been ever trained, create a model
            self.likelihood.append(gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nb_tasks))
            self.model.append(Multitask_GP_Model(x_train, y_train, self.likelihood[block_number], nb_tasks, self.kernel))

        self.model[block_number].train()
        self.likelihood[block_number].train()
        optimizer = torch.optim.Adam([{'params': self.model[block_number].parameters()}, ], lr=self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood[block_number], self.model[block_number])
        loss_list_cur = []

        plot_frequency = self.n_training_iter // 10

        if smart_end:
            loss_hist = 0

        for i in range(self.n_training_iter):
            optimizer.zero_grad()
            output = self.model[block_number](x_train)
            loss = -mll(output, y_train)

            if i>120 and smart_end:
                min_loss_variation = np.min(np.array(loss_list_cur[1:30])-np.array(loss_list_cur[0:29]))
                if loss - loss_hist > - min_loss_variation :
                    break
                else:
                    loss.backward()
                    optimizer.step()
                    if i % plot_frequency == 0:
                        print('Iter %d/%d - Loss: %.3f' % (i + 1, self.n_training_iter, loss.item()))
                    loss_list_cur.append(loss.item())

            else:
                loss.backward()
                optimizer.step()
                if i % plot_frequency == 0:
                    print('Iter %d/%d - Loss: %.3f' % (i + 1, self.n_training_iter, loss.item()))
                loss_list_cur.append(loss.item())

            loss_hist = loss.item()

        self.loss_list.append(loss_list_cur)


    def build_and_train_block_models(self, x_train, y_train, smart_end = False):
        '''
        :param x_train: array size nb_timesteps *1, represents time
        :param y_train: array size nb_timesteps * nb_tasks
        :return: train the multiple MGP, one for each block
        '''
        for i in range(self.number_of_block):
            print('### BLOCK %d ###'%i)
            self.build_and_train_single_model(x_train, y_train[:,self.block_indices[i]], i, smart_end)


    def test_block_model(self, x_test):
        '''
        :param x_test: array size nb_timesteps_test * 1, represents time
        :return:  test_mean_list : the mean of the posterior MGPs
                  test_covar_matrix_list : the psoetrior covariance matrices
                  test_std :the standard deviation of the MGPs
        BE CAREFUL : the outputs are list, each block has then its own mean /coavriances arrays
        '''

        test_mean_list = []
        test_covar_matrix_list = []
        test_std = []

        for i in range(self.number_of_block):

            self.model[i].eval()
            self.likelihood[i].eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_observed_pred = self.likelihood[i](self.model[i](x_test))
                test_mean= test_observed_pred.mean.detach().numpy()
                test_covar_matrix = self.model[i].return_covar_matrix(x_test).detach().numpy()

            test_mean_list.append(test_mean)
            test_covar_matrix_list.append(test_covar_matrix)

            test_lower, test_upper = test_observed_pred.confidence_region() #95% confidence interval
            test_lower, test_upper = test_lower.detach().numpy(), test_upper.detach().numpy()
            test_std.append((test_upper - test_lower) / 2*1.96) # 95% confidence interval to std

        return test_mean_list, test_covar_matrix_list, test_std


    def plot_model(self, x_train, y_train, x_test, train_filter):
        '''
        :param x_train: array size nb_timesteps * 1, represents time
        :param y_train: array size nb_timesteps * nb_tasks
        :param x_test: array size nb_timesteps_test * 1, represents time
        :param train_filter : indices of the selected points for the training
        :return: a plot of the losses, the covariance matrices and the regression for each block
        '''

        test_mean_list, test_covar_matrix_list, test_std_deviation = self.test_block_model(x_test)

        fig = plt.figure(figsize=(18.5,9))
        gs = GridSpec(2, max(self.total_nb_tasks, 2*self.number_of_block))
        iter = 0

        for j in range(self.number_of_block):

            if len(self.block_indices[j])==1: #Single GP
                ax = fig.add_subplot(gs[0, iter])
                ax.plot(x_test.detach().numpy(), test_mean_list[j])
                ax.fill_between(x_test, test_mean_list[j] + test_std_deviation[j],
                                test_mean_list[j] - test_std_deviation[j], alpha=0.3)
                ax.set_title('Block %d Level %d' % (j, self.block_indices[j][0]))
                ax.plot(x_train.detach().numpy(), y_train.detach().numpy()[:, self.block_indices[j]],color='tomato')
                ax.plot(x_train.detach().numpy()[train_filter],y_train.detach().numpy()[train_filter, self.block_indices[j][0]], 'k*', color='red')
                iter = iter + 1
                ax.axvline(x_train.shape[0]/x_test.shape[0], color='green')

            else: #MGP
                for i in range(len(self.block_indices[j])):
                    ax = fig.add_subplot(gs[0, iter])
                    ax.plot(x_test.detach().numpy(), test_mean_list[j][:,i])
                    ax.fill_between(x_test, test_mean_list[j][:,i] + test_std_deviation[j][:,i], test_mean_list[j][:,i] - test_std_deviation[j][:,i], alpha=0.3)
                    ax.set_title('Block %d Level %d'%(j,self.block_indices[j][i]))
                    ax.plot(x_train.detach().numpy(), y_train.detach().numpy()[:, self.block_indices[j][i]], color='tomato')
                    ax.plot(x_train.detach().numpy()[train_filter], y_train.detach().numpy()[train_filter, self.block_indices[j][i]], 'k*', color='red')
                    ax.axvline(x_train.shape[0]/x_test.shape[0], color='green')
                    iter=iter+1


        for j in range(self.number_of_block):
            nb_tasks = len(self.block_indices[j])

            if nb_tasks ==1: #single GP
                ax1 = fig.add_subplot(gs[1, 2*j])
                ax1.imshow(test_covar_matrix_list[j])
                ax1.set_title('Block %d Covar Matrix' % j)
            if nb_tasks > 1 :  # multi GP
                ax1 = fig.add_subplot(gs[1, 2*j])
                matrix = change_representation_covariance_matrix(test_covar_matrix_list[j], nb_tasks)
                ax1.imshow(matrix)
                ax1.set_title('Block %d Covar Matrix' % j)

            ax2 = fig.add_subplot(gs[1, 2*j+1])
            ax2.plot(self.loss_list[j])
            ax2.set_title('Block %d Loss' % j)

        plt.show()

def train_Block_MGP_multiple_individuals(x_train, y_train, x_test, y_test, block_indices,
                                            kernel, learning_rate, n_iter,
                                            train_sample_subset = np.array([]), main_dir='unknown_dir', exec_type='unknown_exec', train_sampling_type = 'unknown_sampling',
                                            activate_plot=False, smart_end = False):
    '''
    :param x_train: array size nb_timesteps_test * 1, represents time
    :param y_train: array size nb_individuals * nb_timesteps_test * number_tasks
    :param block_indices: list of lists of indices (ex: [[0,1],[2,3],[4]]
    :param x_test: array size nb_timesteps_test * 1, represents time
    :param y_test: array size nb_individuals * nb_timesteps_test * number_tasks
    :param save_h5: boolean, to save the test values in a h5 file or not
    :param activate_plot: to plot for each individual the resulted regressions, losses...
    :return: train Block MGP for multiple individuals
    :return: predicted values (of same size as y_test) at x_test

    BE CAREFUL : x_train and x_test must be the same for all the individuals...
    '''

    flat_block_indices = [item for sublist in block_indices for item in sublist]
    y_predicted = np.nan*np.ones(y_test.numpy().shape)

    a = []
    for i in range(len(block_indices)):
        a.append([])
        for j in range(len(block_indices[i])):
            a[i].append(flat_block_indices.index(block_indices[i][j]))
    block_indices = a

    if len(x_train.shape)>1:
        raise ValueError('Wrong dimensions for the input X_train, x_train should be a 1D Vector')
    if len(x_test.shape)>1:
        raise ValueError('Wrong dimensions for the input X_test, x_test should be a 1D Vector')
    if x_train.shape[0] != y_train.shape[1]:
        raise ValueError('Number of time steps is different for x_train and y_train')


    flat_indices = [item for sublist in block_indices for item in sublist]
    nb_individuals, _, nb_tasks = y_train.shape

    if max(flat_indices) > nb_tasks:
        raise ValueError('One of the block indices is higher than the number of tasks in Y_train')

    list_means = []
    list_covariance_matrix = []

    for i in range(nb_individuals):
        # Training subset?
        if len(train_sample_subset.shape)==0:
            this_train_sample_subset = np.arange(x_train.shape[0])
        elif len(train_sample_subset.shape)==1:
            this_train_sample_subset=train_sample_subset
        elif len(train_sample_subset.shape)==2:
            this_train_sample_subset=train_sample_subset[i]
        else:
            raise ValueError('Error with train_sample_subset.shape={}'.format(train_sample_subset.shape))


        # Just use subset for training
        x_train_cur = x_train[this_train_sample_subset]
        y_train_cur = y_train[i, this_train_sample_subset]

        print('###########      INDIVIDUAL %d    ###########'%i)
        # Define and train
        mgp = Block_MGP(kernel, learning_rate, n_iter, block_indices)
        mgp.build_and_train_block_models(x_train_cur, y_train_cur, smart_end)
        # Plot if desired
        if activate_plot:
            mgp.plot_model(x_train, y_train[i], x_test, train_filter = this_train_sample_subset)
        
        # Predict for this individual
        test_mean_list, test_covar_matrix_list, _ = mgp.test_block_model(x_test)
        list_means.append(test_mean_list)
        list_covariance_matrix.append(test_covar_matrix_list)
        # Keep predicted mean
        for k in range(len(block_indices)):
            y_predicted[i,:,block_indices[k]]=test_mean_list[k].T

    # Save dataset
    h5_dataset_path='{}/{}/trained_models/MGP{}blocks_{}.h5'.format(main_dir, exec_type, len(block_indices), train_sampling_type)
    h5_dataset = h5py.File(h5_dataset_path, 'w')
    # Per block
    for i in range(len(block_indices)):
        cur_mean = np.array([list_means[j][i] for j in range(y_train.shape[0])])
        cur_covariance = np.array([list_covariance_matrix[j][i] for j in range(y_train.shape[0])])

        h5_dataset.create_dataset('mean_block_%d'%i, data=cur_mean)
        h5_dataset.create_dataset('covar_block_%d'%i, data=cur_covariance)
    # All predictions
    h5_dataset.create_dataset('y_predicted', data=y_predicted)
    h5_dataset.close()

    return h5_dataset_path, y_predicted

# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
