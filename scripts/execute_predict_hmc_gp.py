#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import sys, os, re
import argparse
from itertools import *

# Main code
def main(exec_machine, data_file, t_init):
  
    ########## Possibilities ##########
    # For each of the possible models
    python_scripts=[]
    
    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic'.format(data_file, t_init)]
    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic_2'.format(data_file, t_init)]
    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic_3'.format(data_file, t_init)]
    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic_4'.format(data_file, t_init)]

    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic RBFard'.format(data_file, t_init)]
    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic_2 RBFard'.format(data_file, t_init)]
    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic_3 RBFard'.format(data_file, t_init)]
    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic_4 RBFard'.format(data_file, t_init)]

    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic RQard'.format(data_file, t_init)]
    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic_2 RQard'.format(data_file, t_init)]
    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic_3 RQard'.format(data_file, t_init)]
    python_scripts+=['../src/predict_hmc_gp.py -data_file {} -t_init {} -gp_cov_f periodic_4 RQard'.format(data_file, t_init)]
        
    # Python script
    for (idx, python_script) in enumerate(python_scripts):
        job_name='job_{}_{}_{}'.format(idx, python_script.split()[0].split('.')[0], python_script.split()[-1])
        # Execute
        print('Executing {}'.format(python_script))
        if exec_machine=='laptop':
            os.system('python3 {}'.format(python_script))
        elif exec_machine=='habanero' or exec_machine=='yeti':
            # Script folders
            os.makedirs('{}/{}'.format(os.getcwd(), exec_machine), exist_ok=True)
            # Load template job script
            with open('./template_job_{}.sh'.format(exec_machine)) as template:
                # Read template
                template_data=template.read()
                # Open new job file to write
                with open('./{}/{}.sh'.format(exec_machine, job_name), 'w') as new_job:
                    # Update job name
                    new_job_data=re.sub('template_job',job_name,template_data)
                    # Update output name
                    new_job_data=re.sub('template_output','{}/{}/{}'.format(os.getcwd(), exec_machine, job_name),new_job_data)
                    # Update python script
                    new_job_data=re.sub('python_job','python -u {}'.format(python_script),new_job_data)
                    # Write to file and close
                    new_job.write(new_job_data)
                    new_job.close()
                # Execute new script
                if exec_machine=='habanero':
                    os.system('sbatch ./{}/{}.sh'.format(exec_machine, job_name))
                if exec_machine=='yeti':
                    os.system('qsub ./{}/{}.sh'.format(exec_machine, job_name))
            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example: python3 -m pdb execute_predict_hmc_gp.py -exec_machine habanero -data_file ../data/y_alpha_KmLH/y_clark_y_init_normal_t250_yscale_1_alpha_0.77_KmLH_560 -t_init 100
    parser = argparse.ArgumentParser(description='Gaussian process for hormonal menstrual cycle prediction')
    parser.add_argument('-exec_machine', type=str, default='laptop', help='Where to run the simulation')
    parser.add_argument('-data_file', type=str, default=None, help='Data file to process')
    parser.add_argument('-t_init', type=int, default=100, help='Initial time-instants to skip')

    # Get arguments
    args = parser.parse_args()
    
    # Make sure file exists
    assert os.path.isfile(args.data_file), 'Data file could not be found'

    # Call main function
    main(args.exec_machine, args.data_file, args.t_init)

