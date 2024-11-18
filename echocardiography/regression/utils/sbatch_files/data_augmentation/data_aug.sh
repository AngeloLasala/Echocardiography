#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=3:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=data_aug.err       # standard error file
#SBATCH --output=data_aug.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

for p in 0.2 0.4 0.6 0.8 1.0; do

    python -m echocardiography.regression.data_augmentation --data_path "/leonardo_work/IscrC_Med-LMGM/Angelo/echo_data/regression/DATA_h" --save_dir "/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/regression" \
            --target heatmaps \
            --model unet_res_skip \
            --input_channels 1 \
            --size 240 320 \
            --epochs 100 \
            --batch_size 8 \
            --lr 0.001 \
            --weight_decay 0.0 \
            --threshold_wloss 0.0\
            --par_dir_generate "/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/diffusion/eco"\
            --trial_generate trial_2\
            --experiment_generate cond_ldm_1\
            --guide_w_generate 0.6\
            --epoch_generate 100\
            --percentace $p\
    
done