#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=cross_val.err       # standard error file
#SBATCH --output=cross_val.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

for p in 0.0; do

    python -m echocardiography.regression.cross_validation_aug --data_path "/leonardo_work/IscrC_Med-LMGM/Angelo/echo_data/regression/DATA_h" --save_dir "/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/regression" \
            --target heatmaps \
            --model unet_res_skip \
            --input_channels 1 \
            --size 240 320 \
            --epochs 20 \
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