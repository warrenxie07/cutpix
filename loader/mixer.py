from pathlib import Path
import numpy as np
from loader.datasets import TrainingDataset
import torch
import numpy as np
from typing import Tuple
from scipy.ndimage import zoom
from utils.aug import augmentations_all, augmentations, mixings
from PIL import Image


class CutMixDataset(TrainingDataset):
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.paths[idx]
        image = self._load_image(path)
        label = self._get_class_index(path)
        label_one_hot = np.zeros(len(self.label_dict))
        label_one_hot[label] = 1

        if np.random.rand() > 0.5:
            path2 = self.paths[np.random.randint(len(self))]
            image2 = self._load_image(path2)
            label2 = self._get_class_index(path2)
            label2_one_hot = np.zeros(len(self.label_dict))
            label2_one_hot[label2] = 1

            image, lam = self.mix(image, image2)
            label_one_hot = label_one_hot * lam + label2_one_hot * (1 - lam)

        image_cropped = self._preprocess_image(image)
        image_augmented = zoom(self.aug_fn(image_cropped), zoom=(1, 1.5, 1.5))
        if self.dim == 3:
            image_augmented = image_augmented[None]
        return torch.from_numpy(image_augmented), torch.from_numpy(label_one_hot)

    def mix(self, img1, img2):
        lam = np.random.uniform(0.5, 1)
        size = img1.shape[1]
        size_img1 = int(size * lam)
        offset_h_img1 = np.random.randint(0, size - size_img1)
        offset_w_img1 = np.random.randint(0, size - size_img1)
        img1_crop = img1[:, offset_h_img1:offset_h_img1 +
                         size_img1, offset_w_img1:offset_w_img1 + size_img1]

        img2 = zoom(img2, (1, 0.5, 0.5))
        size = img2.shape[1]
        size_img2 = int(size * (1 - lam))
        offset_h_img2 = min(max(int(np.random.normal(
            loc=size // 2, scale=2) - (size_img2 // 2)), 0), size_img2)
        offset_w_img2 = min(max(int(np.random.normal(
            loc=size // 2, scale=2) - (size_img2 // 2)), 0), size_img2)
        # offset_h_img2 = self._generate_offset_from_normal_dist(size, size_img2, (size - size_img2) / 2)
        # offset_w_img2 = self._generate_offset_from_normal_dist(size, size_img2, (size - size_img2) / 2)
        img2_crop = img2[:, offset_h_img2:offset_h_img2 +
                         size_img2, offset_w_img2:offset_w_img2 + size_img2]

        offset_h = np.random.randint(0, size_img1 - size_img2)
        offset_w = np.random.randint(0, size_img1 - size_img2)
        img1_crop[:, offset_h:offset_h + size_img2,
                  offset_w:offset_w + size_img2] = img2_crop
        return img1_crop, lam


class StackCutMixDataset(CutMixDataset):
    def mix(self, img1, img2):
        d, h, w = img1.shape
        d_ref, h_ref, w_ref = self.image_size

        offset_z = self._generate_offset_from_normal_dist(d, d_ref, 30)
        offset_h = self._generate_offset_from_normal_dist(h, h_ref, 8)
        offset_w = self._generate_offset_from_normal_dist(w, w_ref, 8)

        img1_crop = img1[
            offset_z: offset_z + d_ref,
            offset_h: offset_h + h_ref,
            offset_w: offset_w + w_ref,
        ]

        offset_z = self._generate_offset_from_normal_dist(d, d_ref, 30)
        offset_h = self._generate_offset_from_normal_dist(h, h_ref, 8)
        offset_w = self._generate_offset_from_normal_dist(w, w_ref, 8)

        img2_crop = img2[
            offset_z: offset_z + d_ref,
            offset_h: offset_h + h_ref,
            offset_w: offset_w + w_ref,
        ]

        lam = np.random.rand()
        offset_z = int(lam * d_ref)
        img1_crop[offset_z:] = img2_crop[offset_z:]
        return img1_crop, lam


class MixUpDataset(CutMixDataset):
    def mix(self, img1, img2):
        offset_z = self._generate_offset_from_normal_dist(64, 22, 30)
        offset_h = self._generate_offset_from_normal_dist(128, 64, 8)
        offset_w = self._generate_offset_from_normal_dist(128, 64, 8)

        img1_crop = img1[
            offset_z: offset_z + 22,
            offset_h: offset_h + 64,
            offset_w: offset_w + 64,
        ]

        offset_z = self._generate_offset_from_normal_dist(64, 22, 30)
        offset_h = self._generate_offset_from_normal_dist(128, 64, 8)
        offset_w = self._generate_offset_from_normal_dist(128, 64, 8)

        img2_crop = img2[
            offset_z: offset_z + 22,
            offset_h: offset_h + 64,
            offset_w: offset_w + 64,
        ]

        lam = np.random.rand()
        return img1_crop * lam + img2_crop * (1 - lam), lam


class AugMixDataset(TrainingDataset):
    def __init__(self, paths, image_size, dim, augmentor):
        super().__init__(paths, image_size, dim, augmentor)
        self.aug_fn = self._aug
        self.severity = 3
        self.width = 3
        self.depth = -1
        self.alpha = 1.

    def _aug(self, image):
        ws = np.float32(
            np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))

        mix = np.zeros_like(image)
        for i in range(self.width):
            image_aug = image.copy()
            d = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for _ in range(d):
                op = np.random.choice(augmentations)
                image_aug = op(image_aug, self.severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * image_aug

        mixed = (1 - m) * image + m * mix
        return mixed

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor | int]:
        path = self.paths[idx]
        image = self._load_image(path)
        image_cropped = self._preprocess_image(image)
        image_augmented = zoom(self.aug_fn(image_cropped), zoom=(1, 1.5, 1.5))

        label = self._get_class_index(path)
        if self.dim == 3:
            image_augmented = image_augmented[None]
        return torch.from_numpy(image_augmented), label


class PixMixDataset(TrainingDataset):
    def __init__(self, paths, image_size, dim, augmentor):
        super().__init__(paths, image_size, dim, augmentor)
        self.aug_fn = self._aug
        self.severity = 3
        self.mixing_list = list(
            Path('fractals_and_fvis/fractals').rglob('*.jpg'))

    def _aug(self, image):
        op = np.random.choice(augmentations_all)
        image_aug = op(image, self.severity)
        return image_aug

    def _pixmix(self, volume, mixing_volume):
        if np.random.rand() > 0.5:
            mixed = self.aug_fn(volume).clip(0, 1)
        else:
            mixed = volume

        for _ in range(np.random.randint(1, 4)):
            if np.random.rand() > 0.5:
                aug_volume = self.aug_fn(volume)
            else:
                aug_volume = mixing_volume
            mixed_op = np.random.choice(mixings)
            mixed = mixed_op(mixed, aug_volume, 3)
            mixed = mixed.clip(0, 1)
        return mixed

    def _load_mixings(self):
        path = np.random.choice(self.mixing_list)
        image = Image.open(str(path)).convert('L')
        image_resize = image.resize(
            size=(self.image_size[1] + 10, self.image_size[2] + 10))

        x_start = np.random.randint(0, 10)
        y_start = np.random.randint(0, 10)
        x_end = x_start + self.image_size[1]
        y_end = y_start + self.image_size[2]
        image_crop = image_resize.crop((x_start, y_start, x_end, y_end))
        image_volume = np.stack([np.array(image_crop) / 255
                                for _ in range(self.image_size[0])])
        return image_volume

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor | int]:
        path = self.paths[idx]
        image = self._load_image(path)
        image_cropped = self._preprocess_image(image)
        mixing_volume = self._load_mixings()
        mixed = self._pixmix(image_cropped, mixing_volume)
        image_augmented = zoom(mixed, zoom=(1, 1.5, 1.5))

        label = self._get_class_index(path)
        if self.dim == 3:
            image_augmented = image_augmented[None]
        return torch.from_numpy(image_augmented).float(), label


class CutPixMixDataset(PixMixDataset):
    def _pixmix(self, volume, mixing_volume):
        if np.random.rand() > 0.5:
            mixed = self.aug_fn(volume).clip(0, 1)
        else:
            mixed = volume

        for _ in range(np.random.randint(1, 4)):
            if np.random.rand() > 0.5:
                aug_volume = self.aug_fn(volume)
            else:
                aug_volume = mixing_volume
            mixed_op = np.random.choice(mixings)
            mixed = mixed_op(mixed, aug_volume, 3)
            mixed = mixed.clip(0, 1)

        if np.random.rand() > 0.5:
            return mixed

        cut_index = np.random.randint(0, 7)
        z, y, x = self.image_size
        match cut_index:
            case 0:
                mixed[:z//2] = volume[:z//2]
            case 1:
                mixed[z//2:] = volume[z//2:]
            case 2:
                mixed[:, :y//2] = volume[:, :y//2]
            case 3:
                mixed[:, y//2:] = volume[:, y//2:]
            case 4:
                mixed[:, :, :x//2] = volume[:, :, :x//2]
            case 5:
                mixed[:, :, x//2:] = volume[:, :, x//2:]
        return mixed
