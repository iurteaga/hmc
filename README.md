# Hormonal menstrual cycle project

Repository for the work on hormonal menstrual cycle: experiments with mechanistic models and machine learning.

## References
[1] Towards Personalized Modeling of the Female Hormonal Cycle: Experiments with Mechanistic Models and Gaussian Processes, Iñigo Urteaga, David J. Albers, Marija Vlajic Wheeler, Anna Druet, Hans Raffauf, Noémie Elhadad, Accepted at NIPS 2017 Workshop on Machine Learning for Health (https://ml4health.github.io/2017/)

[2] Multi-Task Gaussian Processes and Dilated Convolutional Networks for Reconstruction of Reproductive Hormonal Dynamics, Iñigo Urteaga, Tristan Bertin, Theresa M. Hardy, David J. Albers, Noémie Elhadad, Presented at Machine Learning for Healtcare (MLHC) 2019 (https://arxiv.org/abs/1908.10226)

## Directories

### src

Directory where the algorithms for simulation and prediction of hmc data are implemented.

./src/ contains the GP based prediction code presented in [1].

./src/MGP_DCNN/ contains the MGP+DCNN python implementation developed in collaboration with https://github.com/TristanBertin and presented in [2].

### scripts

Python code for evaluation and plotting of hmc algorithms.

./scripts/ contains python code to execute prediction, evaluation and plotting of results presented in [1].

./scripts/mlhc_2019.py contains python code to replicate results presented in [2].

./scripts/jupyter/ contains example python notebooks with code illustrating the MGP+DCNN approach presented in [2].

### data

Directory where the mechanistic model saves the generated hormonal data

- data/examples: some examples of simulated hormonal data for specific parameterizations.

- data/mlhc_2019: example simulated hormonal data used in [2].
