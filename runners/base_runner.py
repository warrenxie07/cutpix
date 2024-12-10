import torch
from pathlib import Path
from abc import abstractmethod
import torch.distributed as dist


class BaseRunner(object):
    def __init__(self, loader, model, logger, model_path, rank):
        self.loader = loader
        self.model = model
        self.logger = logger
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        # self.load()

    def log(self, string, log_type):
        if self.rank == 0:
            if log_type == "info":
                logger = self.logger.info
            else:
                logger = self.logger.debug
            logger(string)

    @abstractmethod
    def load(self):
        pass


@torch.no_grad()
def reduce_tensor(tensor, mean=False):
    try:
        world_size = dist.get_world_size()
    except AssertionError:
        world_size = 1
    if world_size < 2:
        return tensor
    temp = tensor.clone()
    dist.all_reduce(temp)
    if dist.get_rank() == 0 and mean:
        temp /= world_size
    return temp


@torch.no_grad()
def gather_tensor(tensor):
    try:
        world_size = dist.get_world_size()
    except AssertionError:
        world_size = 1
    if world_size < 2:
        return tensor
    temp = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(temp, tensor)
    return temp
