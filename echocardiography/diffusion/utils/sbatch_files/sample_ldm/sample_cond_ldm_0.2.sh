#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=relative_0_2.err       # standard error file
#SBATCH --output=relative__0_2.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

for trial in trial_2; do
    for w in 0.2 0.6 1.0; do
        for epoch in 120 150; do
                python -m echocardiography.diffusion.tools.sample_cond_ldm\
                        --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/'\
                        --trial $trial\
                        --experiment cond_ldm_6\
                        --epoch $epoch\
                        --guide_w $w\

         done
    done
done
