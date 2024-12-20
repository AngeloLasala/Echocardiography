#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=18:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=relative_cond_ldm.err       # standard error file
#SBATCH --output=relative_cond_ldm.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

python -m echocardiography.diffusion.tools.train_cond_ldm --data eco_class_relative_cond\
          --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco'\
          --trial trial_2
 