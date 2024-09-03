#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=2:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=unet_res_skip_8.err       # standard error file
#SBATCH --output=unet_res_skip_8.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

python -m echocardiography.regression.train --data_path "/leonardo_work/IscrC_Med-LMGM/Angelo/echo_data/regression/DATA_h" --save_dir "/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/regression" \
        --target heatmaps \
        --model unet_res_skip \
        --input_channels 1 \
        --size 240 320 \
        --epochs 100 \
        --batch_size 8 \
        --lr 0.001 \
        --weight_decay 0.0 \
        --threshold_wloss 0.0