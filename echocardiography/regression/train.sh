#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=10:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=error_file.err       # standard error file
#SBATCH --output=file_print.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

python -m echocardiography.regression.train --data_path "/leonardo_work/IscrC_Med-LMGM/Angelo/echo_data/regression/DATA_h" --save_dir "/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/regression" \
        --target heatmaps \
        --model unet_up \
        --input_channels 1 \
        --size 240 320 \
        --epochs 5 \
        --batch_size=16 \
        --lr 0.001 \
        --weight_decay 0.0 \
        --threshold_wloss 0.0