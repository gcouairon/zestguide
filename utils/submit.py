# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import submitit
import os
import numpy as np
from pathlib import Path

def submit(f, 
           *args, 
           output='generated/default', 
           slurm_partition='learnlab',
           ngpus=2,
           **kwargs):
    #function to parallelize f
    logs_folder = Path(output).joinpath('sblogs')
    logs_folder.mkdir(exist_ok=True)
    aex = submitit.AutoExecutor(folder=str(logs_folder))

    aex.update_parameters(timeout_min=60*24, 
                          slurm_partition=slurm_partition, 
                          nodes=1, 
                          gpus_per_node=1, 
                          slurm_array_parallelism=ngpus)

    with aex.batch():
        jobs = [aex.submit(f, *args, **kwargs) for _ in range(ngpus)]
    
    print(f'{ngpus} jobs successfully scheduled !')
