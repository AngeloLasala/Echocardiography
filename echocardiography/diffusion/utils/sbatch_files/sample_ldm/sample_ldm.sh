#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=sample_ldm.err       # standard error file
#SBATCH --output=sample_ldm.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name


for epoch in 20 40 60 80 100 120 150; do
        python -m echocardiography.diffusion.tools.sample_ldm --data eco_image_cond_all_batch\
                --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/'\
                --trial trial_2\
                --experiment ldm_1\
                --epoch $epoch
    done