"""
Distributed Data parallel utilities
"""
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn

import torchvision.models as models

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data_utils

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def get_resources(verbose=True):

    # rank = 0
    # local_rank = 0
    # world_size = 1

    if os.environ.get("RANK"):
        # launched with torchrun (python -m torch.distributed.run)
        rank = int(os.getenv("RANK"))
        local_rank = int(os.getenv("LOCAL_RANK"))
        world_size = int(os.getenv("WORLD_SIZE"))
        local_size = int(os.getenv("LOCAL_WORLD_SIZE"))
        if verbose and rank == 0:
            print("launch with torchrun")

    elif os.environ.get("OMPI_COMMAND"):
        # launched with mpirun
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
        if verbose and rank == 0:
            print("launch with mpirun")

    else:
        # launched with srun (SLURM)
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])
        local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        if verbose and rank == 0:
            print("launch with srun")

    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])

    return rank, local_rank, world_size, local_size, num_workers



# dist.init_process_group("nccl", rank=rank, world_size=world_size)



if __name__ == '__main__':  

    # Initialize the distributed training environment
    # setup_dist()
    rank, local_rank, world_size, local_size, num_workers = get_resources()
    print(rank, local_rank, world_size, local_size, num_workers)

   