#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=1:30:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=fid.err       # standard error file
#SBATCH --output=fid.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

for w in 0.0 0.2 0.4 0.8; do
    python -m echocardiography.diffusion.evaluation.fid\
                    --par_dir '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco'\
                    --trial trial_2\
                    --experiment cond_ldm_6\
                    --guide_w $w
    done