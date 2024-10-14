#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=test_double_ldm.err       # standard error file
#SBATCH --output=test_double_ldm.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name


python -m echocardiography.diffusion.tools.test_cond_ldm\
        --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/'\
        --trial trial_2\
        --experiment cond_ldm_1\
        --epoch 100\
        --guide_w 0.6

