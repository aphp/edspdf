from typing import List, Optional, Union

import torch

from ..registry import registry
from .base import Accelerator


@registry.accelerator.register("multiprocessing")
class MultiprocessingAccelerator(Accelerator):
    """
    Deprecated: Use `docs.map_pipeline(model).set_processing(...)` instead
    """

    def __init__(
        self,
        batch_size: int,
        num_cpu_workers: Optional[int] = None,
        num_gpu_workers: Optional[int] = None,
        gpu_pipe_names: Optional[List[str]] = None,
        gpu_worker_devices: Optional[List[Union[torch.device, str]]] = None,
        cpu_worker_devices: Optional[List[Union[torch.device, str]]] = None,
    ):
        self.batch_size = batch_size
        self.num_gpu_workers: Optional[int] = num_gpu_workers
        self.num_cpu_workers = num_cpu_workers
        self.gpu_pipe_names = gpu_pipe_names
        self.gpu_worker_devices = gpu_worker_devices
        self.cpu_worker_devices = cpu_worker_devices
