#!/bin/bash
#SBATCH --nodes=1                  # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=10:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=eval_1.err       # standard error file
#SBATCH --output=eval_1.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

for trial in trial_3 trial_4; do
    for epoch in 20 40 60 80 100 120 150; do
        python -m echocardiography.diffusion.evaluation.hypertropy_eval\
                --par_dir '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco'\
                --par_dir_regression '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/regression'\
                --trial_regression trial_3\
                --trial $trial\
                --experiment cond_ldm_3\
                --guide_w -1.0\
                --epoch $epoch
    done
done

 