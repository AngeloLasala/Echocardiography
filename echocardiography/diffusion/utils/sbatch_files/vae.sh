#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=40:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=vae.err       # standard error file
#SBATCH --output=vae.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

python -m echocardiography.diffusion.tools.train_vae --data eco_image_cond  --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion'
 