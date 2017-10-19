#!/bin/sh

# Directives
#PBS -N simulate_dde_alpha
#PBS -W group_list=yetidsi
#PBS -M iu2153@columbia.edu
#PBS -m bea
# Wall time set to max in batch queues
#PBS -l walltime=120:00:00
#PBS -V

# Output and error directories
#PBS -o localhost:simulate_dde_alpha.out
#PBS -e localhost:simulate_dde_alpha.log

# Start
echo "Starting script $PBS_JOBNAME at $(date)"

# Load matlab module
module load matlab/2016b

# Params
model="'clark'"
params_file="'../src/input/clark_params'"
y0_file="'../src/input/clark_y_init_normal'"
options_file="'../src/input/options_file'"
t_max=150
n_t=150
# Run script
matlab -nosplash -nodisplay -nodesktop -r "../src/simulate_dde_alpha($model, $params_file, $y0_file, $options_file, $t_max, $n_t)" > simulate_dde_alpha

# Finished
echo "Done at $(date)"

# END
