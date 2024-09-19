#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=sample_cond_ldm_2.err       # standard error file
#SBATCH --output=sample_cond_ldm_2.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

for w in 0.2 ; do
    for epoch in 120; do
            python -m echocardiography.diffusion.tools.sample_cond_ldm --data eco_image_cond_all_batch\
                    --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/'\
                    --trial trial_2\
                    --experiment cond_ldm_1\
                    --epoch $epoch\
                    --guide_w $w
     done
done