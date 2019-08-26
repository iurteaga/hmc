import random
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import os
from skopt.plots import plot_convergence
from skopt import gp_minimize
import matplotlib.pyplot as plt
import h5py
import pdb

from TNN_subclasses import *

class Time_Neural_Network():

    def __init__(self, model_type, batch_size, nb_epochs,
                        x_train, y_train, x_val, y_val, x_test, y_test,
                        main_dir='unknown_dir', exec_type='unknown_exec', train_sampling_type='unknown_sampling'):

        ''' ################       FIXED FOR THE OPTIMIZATION       #####################'''

        self.nb_time_steps = x_train.shape[1]
        self.nb_tasks_input = x_train.shape[-1]
        self.nb_tasks_output = y_train.shape[-1]

        self.model_type = model_type
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        self.main_dir=main_dir
        self.exec_type=exec_type
        self.train_sampling_type=train_sampling_type
        
        ''' ################       VARIABLES OF THE OPTIMIZATION       #####################'''
        self.learning_rate = None
        self.nb_layers = None
        self.nb_filters = None
        self.regularizer_coef = None
        self.kernel_size = None
        self.dilation_factor = None
        
        # To keep track of parameters and predictions as we optimize
        self.params_history = []
        self.y_predicted = []


    def build_TNN(self, learning_rate, nb_hidden_layers, nb_filters, regularizer_coef=1e-7, kernel_size=0, dilation_factor=1, display_summary=False):

        '''
        :param learning_rate:
        :param nb_layers:
        :param nb_filters: BE CAREFUL : in the case of CNN : of nb of filters
                                        in the case of RNN : = hidden dimension = dimension of state between two units
        :param kernel_size:
        :param regularizer_coef:
        :return: BUILD network
        '''

        self.learning_rate = learning_rate
        self.nb_layers = nb_hidden_layers
        self.nb_filters = nb_filters
        self.regularizer_coef = regularizer_coef
        self.kernel_size = kernel_size
        self.dilation_factor = dilation_factor

        ###########################                   BUILD MODEL                     #############################

        if self.model_type == 'RNN':
            self.model = RNN_time_prediction(nb_time_steps = self.nb_time_steps,
                                        nb_tasks_input = self.nb_tasks_input,
                                        nb_tasks_output= self.nb_tasks_output,
                                        regularizer_coef = regularizer_coef,
                                        nb_layers = nb_hidden_layers,
                                        nb_hidden_dimension = nb_filters,
                                        learning_rate=learning_rate,)


        if self.model_type == 'CNN':
            if kernel_size == 0:
                raise ValueError('CNN model needs a kernel_size different from 0')

            self.model = CNN_time_prediction(xy_length = self.nb_time_steps,
                                        nb_tasks_input = self.nb_tasks_input,
                                        nb_tasks_output= self.nb_tasks_output,
                                        nb_hidden_layers=nb_hidden_layers,
                                        learning_rate=learning_rate,
                                        number_of_filters=nb_filters,
                                        kernel_size=kernel_size,
                                        regularizer_coef=regularizer_coef,
                                        dilation_factor= dilation_factor)

        if display_summary:
            self.model.summary()



    def train_validate_TNN(self, regression_plot=False):

        if self.learning_rate == None:
            raise ValueError('You have to build a network before training it ---> call obj.build_TNN() ')

        callbacks_list = [
            ModelCheckpoint(
                filepath='{}/{}/trained_models/{}_{}.h5'.format(self.main_dir, self.exec_type, self.model_type, self.train_sampling_type),
                monitor='val_loss', save_best_only=True),
            EarlyStopping(monitor='acc', patience=5)]

        # Fit the model
        t1 = time.time()
        x_train = self.x_train
        y_train = self.y_train
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.nb_epochs,
                       callbacks=callbacks_list,
                       verbose=0,
                       validation_data=(self.x_val, self.y_val))

        self.training_time = time.time() - t1

        self.model.load_weights(
            '{}/{}/trained_models/{}_{}.h5'.format(self.main_dir, self.exec_type, self.model_type, self.train_sampling_type)
            )
        
        # Keep list of used params
        self.params_history.append([self.learning_rate, self.nb_layers, self.nb_filters, self.kernel_size, self.regularizer_coef])
        # Predict
        y_predicted_train = self.model.predict(self.x_train)
        y_predicted_val = self.model.predict(self.x_val)
        y_predicted_test = self.model.predict(self.x_test)
        # Keep predictions            
        self.y_predicted.append(np.concatenate((y_predicted_train,y_predicted_val,y_predicted_test), axis=0))
            
        # Evaluate
        train_loss = self.compute_loss(y_predicted_train, self.y_train)
        val_loss = self.compute_loss(y_predicted_val, self.y_val)
        test_loss = self.compute_loss(y_predicted_test, self.y_test)
        print('MGP+TNN: \t Train loss={} ({})\n\t\t Validation loss={} ({})\n\t\t Test loss={} ({})'.format(train_loss[0], train_loss[1], val_loss[0], val_loss[1], test_loss[0],test_loss[1]))
        print('Time', self.training_time, 'Learning_rate ', self.learning_rate, ' nb_layers ', self.nb_layers, ' nb_filters ', self.nb_filters, ' kernel_size ', self.kernel_size, ' regularizer_coef ', self.regularizer_coef, ' dilation_factor ', self.dilation_factor)

        # Plot if desired
        if regression_plot == True:
            self.output_plots(self.x_train, self.y_train, y_predicted_train, self.x_test, self.y_test, y_predicted_test, val_loss[0])

        # Return all losses
        return [train_loss, val_loss, test_loss]

    def build_and_train(self, values):

        learning_rate = values[0]
        nb_hidden_layers = values[1]
        nb_filters = values[2]
        regularizer_coef = values[3]
        kernel_size= values[4]
        dilation_factor = values[5]

        # Build
        self.build_TNN(learning_rate, nb_hidden_layers, nb_filters, regularizer_coef, kernel_size, dilation_factor)
        # Train
        losses = self.train_validate_TNN(regression_plot=False)
        # Return validation loss (needed for bayesian optimization)
        return losses[1][0]


    def optimization_process(self, range_parameters, default_parameters, nb_calls, nb_random_starts, plot_conv=False):

        search_result = gp_minimize(func=self.build_and_train,
                                    dimensions=range_parameters,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=nb_calls,
                                    n_random_starts=nb_random_starts,
                                    x0=default_parameters)
        if plot_conv==True:
            plot_convergence(search_result)
            plt.show()

        return search_result


    def output_plots(self, x_train, y_train, output_train, x_test, y_test, output_test, score):

        if not os.path.exists('{}/plots'.format(self.main_dir)):
            os.makedirs('{}/plots'.format(self.main_dir))

        fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(20, 10))
        j = 0
        while j<5:
            id = random.randint(0, x_test.shape[0] - 1)

            if np.max(y_test[id,:,0]) > 0.4:
                for i in range(5):
                    axes[i, j].set_title('H_%d Id_%d' % (i, id))
                    axes[i, j].plot(output_test[id, :, i], 'orange', label='prediction RNN')
                    axes[i, j].plot(x_test[id, :, i], label='prediction GP')
                    axes[i, j].plot(y_test[id, :, i], label='truth')
                    axes[i, j].legend()
                fig.suptitle('Test')

                j = j + 1

            else:
                pass

        plt.savefig('{}/plots/{}_LOSS_{}_TEST_{}.pdf'.format(self.main_dir,self.model_type, score * 10000, self.train_sampling_type))
        plt.close(fig)
        plt.close()

        # fig, axes = plt.subplots(ncols=5, nrows=5, figsize=(20, 10))
        # for j in range(5):
        #     id = random.randint(0, x_train.shape[0] - 1)
        #     for i in range(5):
        #         axes[i, j].set_title('H_%d Id_%d' % (i, id))
        #         axes[i, j].plot(output_train[id, :, i], 'orange', label='prediction RNN')
        #         axes[i, j].plot(x_train[id, :, i], label='prediction GP')
        #         axes[i, j].plot(y_train[id, :, i], label='truth')
        #         axes[i, j].legend()
        # fig.suptitle('Train')
        # plt.savefig('plots/%s_LOSS_%d_TRAIN_%s.pdf' %
        #             (self.model_type, score * 10000, self.train_sampling_type))
        # plt.close()


    def compute_loss(self, target, prediction):
        assert(target.shape == prediction.shape)
        squared_error=(target - prediction)**2
        return np.mean(squared_error), np.std(squared_error)

# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
