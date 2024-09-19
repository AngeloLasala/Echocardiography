#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=40:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=error_file.err       # standard error file
#SBATCH --output=file_print.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

for i in 1 2 3 4 5 6 7 8 9
do
   python -m echocardiography.regression.test --data_path "/leonardo_work/IscrC_Med-LMGM/Angelo/echo_data/regression/DATA_h" --model_path "/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/regression/" \
        --trial trial_$i \
        --split val \
        --method_center max_value

    python -m echocardiography.regression.test --data_path "/leonardo_work/IscrC_Med-LMGM/Angelo/echo_data/regression/DATA_h" --model_path "/leonardo_work/IscrC_Med-LMGM/Angelo/trained_model/regression/" \
            --trial trial_$i \
            --split val \
            --method_center ellipses
done

