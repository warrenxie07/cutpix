from ast import Raise
from typing import Tuple, Union
from torch.utils.data import sampler
from pathlib import Path
from torch.utils import data
from itertools import chain

import torch.distributed as dist

from loader.datasets import TrainingDataset, TestDataset
from loader.mixer import (CutMixDataset, StackCutMixDataset,
                          MixUpDataset, AugMixDataset, PixMixDataset, YocoPixMixDataset)


class DataLoader(object):
    def __init__(
        self,
        root: str,
        image_size: Tuple[int],
        batch_size: int,
        num_cores: int,
        dim: int,
        mix: str = "none",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_cores = num_cores
        self.dim = dim
        self.mix = mix
        self.seed = seed
        self.args_training = dict()

    def load(self, dataset) -> data.DataLoader:
        _f = {
            "train": lambda: self._train(),
            "val": lambda: self._val(),
            "test": lambda: self._test(),
        }
        try:
            loader = _f[dataset]()
        except KeyError:
            raise "Dataset should be one of [train, val, test]."
        return loader

    def _train(self):
        paths = list(Path(f"{self.root}/train").rglob("*.mat"))
        loader_dict = {
            "none": TrainingDataset,
            "cutmix": CutMixDataset,
            "cutmix_stack": StackCutMixDataset,
            "mixup": MixUpDataset,
            "augmix": AugMixDataset,
            "pixmix": PixMixDataset,
            "cutpix": CutPixMixDataset,
        }
        dataset = loader_dict[self.mix](
            paths, self.image_size, self.dim, **self.args_training
        )
        sampler = data.DistributedSampler(
            dataset, shuffle=True, seed=self.seed)
        loader = data.DataLoader(
            dataset,
            self.batch_size,
            sampler=sampler,
            num_workers=self.num_cores,
            pin_memory=True,
        )
        return loader

    def _val(self):
        paths = list(Path(f"{self.root}/valid").rglob("*.mat"))
        dataset = TestDataset(paths, self.image_size, self.dim)
        sampler = data.DistributedSampler(dataset, shuffle=False)
        loader = data.DataLoader(
            dataset,
            self.batch_size,
            sampler=sampler,
            num_workers=self.num_cores,
            pin_memory=True,
        )
        return loader

    def _test(self):
        paths = list(Path(f"{self.root}/test").rglob("*.mat"))
        dataset = TestDataset(paths, self.image_size, self.dim)
        sampler = data.DistributedSampler(dataset, shuffle=False)
        loader = data.DataLoader(
            dataset,
            self.batch_size,
            sampler=sampler,
            num_workers=self.num_cores,
            pin_memory=True,
        )
        return loader

    def set_train_args(self, **args_training):
        self.args_training.update(args_training)


if __name__ == "__main__":
    loader = DataLoader("fold.yaml", 4, 4, 0)
    dataset = loader._train()

    for d, p, _ in dataset:
        print(d.shape, p)
