#!/bin/bash
#SBATCH --nodes=1                  # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=2:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=eval_2.err       # standard error file
#SBATCH --output=eval_2.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

python -m echocardiography.diffusion.evaluation.read_test --par_dir "/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco"\
                    --trial trial_2\
                    --experiment cond_ldm_1\
                    --guide_w 0.6\
                    --epoch 100 