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

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=11111

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT


# python -m echocardiography.diffusion.utils.dist_utils
torchrun /leonardo_work/IscrC_Med-LMGM/Angelo/Echocardiography/echocardiography/diffusion/utils/dist_utils.py 