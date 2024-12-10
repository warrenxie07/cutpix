from utils import corruption
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
from einops import rearrange, reduce, repeat
from multiprocessing import Pool
from scipy.ndimage import zoom, affine_transform, rotate
import cv2
from functools import partial

IMAGE_SIZE = 64
Z_STACK = 22


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def _translate(img, x, y, z):
    matrix = np.eye(4)
    matrix[:3, 3] = [z, y, x]
    translated_image = affine_transform(
        img, matrix[:3, :3], offset=matrix[:3, 3])
    return translated_image


def _shear(img, shear_factor, axis_pair):
    matrix = np.eye(4)
    matrix[axis_pair[0], axis_pair[1]] = shear_factor
    translated_image = affine_transform(img, matrix[:3, :3])
    return translated_image


def rotate_xy(img, level):
    level = int_parameter(sample_level(level), 30)
    if np.random.rand() > 0.5:
        level = -level
    return rotate(img, level, (1, 2), order=1, reshape=False)


def translate_x(img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE // 3)
    if np.random.rand() > 0.5:
        level = -level
    return _translate(img, level, 0, 0)


def translate_y(img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE // 3)
    if np.random.rand() > 0.5:
        level = -level
    return _translate(img, 0, level, 0)


def translate_z(img, level):
    level = int_parameter(sample_level(level), Z_STACK // 3)
    if np.random.rand() > 0.5:
        level = -level
    return _translate(img, 0, 0, level)


def shear_x(img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.rand() > 0.5:
        level = -level
    return _shear(img, level, (0, 1))


def shear_y(img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.rand() > 0.5:
        level = -level
    return _shear(img, level, (1, 0))


def shear_z(img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.rand() > 0.5:
        level = -level
    return _shear(img, level, (2, 0))


def sharpness(img, level):
    level = float_parameter(sample_level(level), 1.0)
    return corruption.sharpness(img, level)


def zoom_blur(img, level):
    level = int_parameter(sample_level(level), 10)
    return corruption.zoom_blur(img, level)


augmentations = [
    rotate_xy,
    translate_x,
    translate_y,
    translate_z,
    shear_x,
    shear_y,
    shear_z,
]

augmentations_all = [
    rotate_xy,
    translate_x,
    translate_y,
    translate_z,
    shear_x,
    shear_y,
    shear_z,
    sharpness,
    zoom_blur
]


def get_ab(beta):
    if np.random.random() < 0.5:
        a = np.float32(np.random.beta(beta, 1))
        b = np.float32(np.random.beta(1, beta))
    else:
        a = 1 + np.float32(np.random.beta(1, beta))
        b = -np.float32(np.random.beta(1, beta))
    return a, b


def add(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    out = (out + 1) / 2
    return out


def multiply(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2, img2 * 2
    out = (img1 ** a) * (img2.clip(1e-37) ** b)
    out = out / 2
    return out


mixings = [add, multiply]
