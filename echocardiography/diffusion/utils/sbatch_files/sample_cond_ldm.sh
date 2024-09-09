#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=8:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=cond_ldm.err       # standard error file
#SBATCH --output=cond_ldm.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

for i in 20
do
    python -m echocardiography.diffusion.tools.sample_cond_ldm --data eco_image_cond\
            --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/'\
            --trial trial_1\
            --experiment cond_ldm_1\
            --epoch $i\
            --guide_w 0.1
done