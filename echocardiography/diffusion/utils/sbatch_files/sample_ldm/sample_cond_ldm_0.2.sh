#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=keys_0_2.err       # standard error file
#SBATCH --output=keys__0_2.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

for trial in trial_2; do
    for w in 0.0 0.2 ; do
        for epoch in 20 40 60 80 100 120 150; do
                python -m echocardiography.diffusion.tools.sample_cond_ldm\
                        --save_folder '/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/'\
                        --trial $trial\
                        --experiment cond_ldm_4\
                        --epoch $epoch\
                        --guide_w $w\

         done
    done
done
