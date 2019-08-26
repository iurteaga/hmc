import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import h5py

class Multitask_GP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, gpytorch_kernel):
        super(Multitask_GP_Model, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(),num_tasks=num_tasks)
        self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch_kernel, num_tasks=num_tasks, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def return_covar_matrix(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(self.mean_module(x), self.covar_module(x)).covariance_matrix



class Single_task_GP_model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, gpytorch_kernel):
        super(Single_task_GP_model, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def return_covar_matrix(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x)).covariance_matrix

# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
