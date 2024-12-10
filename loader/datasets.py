from scipy.io import loadmat
import torch
import random
import numpy as np
from numpy import ndarray
from pathlib import Path
from abc import abstractmethod
from typing import List, Tuple, Union
from einops import rearrange
from scipy.ndimage import zoom

from utils.augmentation import RandAugment, flip, norm, axial_rotate, transpose


class _Dataset(object):
    def __init__(
        self,
        paths: List[str],
        image_size: Tuple[int] = (64, 128, 128),
        dim: int = 3,
        augmentor: Union[RandAugment, None] = None,
    ) -> None:
        super().__init__()
        labels = sorted(set([path.parts[-2] for path in paths]))
        indices = list(range(len(labels)))
        self.label_dict = {name: index for name, index in zip(labels, indices)}
        self.image_size = image_size
        self.paths = paths
        self.dim = dim
        self.aug_fn = augmentor or (lambda x: x)

    def _get_class_index(self, path: str) -> int:
        name = Path(path).parts[-2]
        return self.label_dict[name]

    @abstractmethod
    def _preprocess_image(self, image: ndarray) -> ndarray:
        pass

    def _load_image(self, path: str) -> ndarray:
        image = loadmat(path)["data"].astype("float32")
        return rearrange(norm(image), "h w d -> d h w")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.paths[idx]
        image = self._load_image(path)
        image_cropped = self._preprocess_image(image)
        image_augmented = zoom(self.aug_fn(image_cropped), zoom=(1, 1.5, 1.5))
        label = self._get_class_index(path)
        if self.dim == 3:
            image_augmented = image_augmented[None]
        return torch.from_numpy(image_augmented), label

    def __len__(self) -> int:
        return len(self.paths)


class TrainingDataset(_Dataset):
    def _generate_offset_from_normal_dist(self, x, x_ref, denom_var=8):
        diff_x = x - x_ref
        x_offset = np.random.normal(loc=diff_x // 2, scale=diff_x / denom_var)
        return min(max(0, int(x_offset)), diff_x)

    def _preprocess_image(self, image: ndarray) -> ndarray:
        "random cropping"
        d, h, w = image.shape
        d_ref, h_ref, w_ref = self.image_size

        d_offset = self._generate_offset_from_normal_dist(d, d_ref)
        h_offset = self._generate_offset_from_normal_dist(h, h_ref)
        w_offset = self._generate_offset_from_normal_dist(w, w_ref)

        image_cropped = image[
            d_offset: d_offset + d_ref,
            h_offset: h_offset + h_ref,
            w_offset: w_offset + w_ref,
        ]

        image_flipped = flip(image_cropped, random.randint(0, 3))
        image_transposed = transpose(image_flipped, random.randint(0, 1))
        return image_transposed
        # image_rotated = axial_rotate(image_flipped, random.randint(0, 4))
        # return image_rotated


class TestDataset(_Dataset):
    def _preprocess_image(self, image: ndarray) -> ndarray:
        "center cropping"
        d, h, w = image.shape
        d_ref, h_ref, w_ref = self.image_size
        return image[
            d // 2 - d_ref // 2: d // 2 + d_ref // 2,
            h // 2 - h_ref // 2: h // 2 + h_ref // 2,
            w // 2 - w_ref // 2: w // 2 + w_ref // 2,
        ]
