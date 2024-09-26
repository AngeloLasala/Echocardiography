#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=4     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=24:00:00                 # time limits: 1 hour
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=ddp.err       # standard error file
#SBATCH --output=ddp.out      # standard output file
#SBATCH --account=IscrC_Med-LMGM     # account name

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_PORT=29500

python -m echocardiography.diffusion.utils.dist_utils
 