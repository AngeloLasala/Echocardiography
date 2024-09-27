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

# def setup_dist():
#     """
#     Setup a distributed process group.
#     """
#     if dist.is_initialized():
#         return

#     comm = MPI.COMM_WORLD
#     backend = "gloo" if not th.cuda.is_available() else "nccl"

#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)

#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)
#     dist.init_process_group(backend=backend, init_method="env://")


# def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")



if __name__ == '__main__':  

    # Initialize the distributed training environment
    # setup_dist()
    rank, local_rank, world_size, local_size, num_workers = get_resources()

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0:
        # print(f"Dataset path .............. {dataset_path}")
        # print(f"Num. Images ............... {len(dataset)}")
        # print(f"Batch Size ................ {batch_size}")
        # print(f"Num. Batches .............. {len(loader)}")
        print(f"Num. processors ........... {world_size}")
        print(f"Num. GPUs per node ........ {local_size}")
        print(f"Num. Workers .............. {num_workers}")
        print(f"Device ....................", "GPU" if torch.cuda.is_available() else "CPU")
        print(f"GPU device name ........... {torch.cuda.get_device_name(0)}")
        print(f"Num. of available GPUs .... {torch.cuda.device_count()}")
        print(f"Num. of processes ......... {world_size}")

    device = torch.device("cuda:{}".format(local_rank))
    # each process exclusively works on a single GPU
    torch.cuda.set_device(local_rank)
    print(device)
   