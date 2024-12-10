from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat


def plot(img, figsize=(10, 10), cmap="turbo", vmin=0, vmax=0.7):
    img = img[20:-20, 20:-20, 20:-20]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img[12], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_aspect(1.)

    divider = make_axes_locatable(ax)
    ax_yz = divider.append_axes("left", 1.7, pad=0.01, sharey=ax)
    ax_xz = divider.append_axes("bottom", 1.7, pad=0.01, sharex=ax)
    # ax_cb = divider.append_axes("right", 0.5, pad=0.15)

    ax_yz.imshow(img.max(2).T, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xz.imshow(img.max(1), cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xz.axis("off")
    ax_yz.axis("off")
    # plt.colorbar(im, cax=ax_cb)
    plt.show()


def plot_main(img, figsize=(11, 11), cmap="turbo", vmin=0, vmax=0.7):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img[11], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    ax.set_aspect(1.)

    divider = make_axes_locatable(ax)
    ax_yz = divider.append_axes("left", 1.7, pad=0.01, sharey=ax)
    ax_xz = divider.append_axes("bottom", 1.7, pad=0.01, sharex=ax)
    # ax_cb = divider.append_axes("right", 0.5, pad=0.15)

    ax_yz.imshow(img.max(2).T, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xz.imshow(img.max(1)[::-1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax_xz.axis("off")
    ax_yz.axis("off")
    # plt.colorbar(im, cax=ax_cb)
    plt.show()


def load_mat(path: Path) -> None:
    matData = loadmat(path)
    return matData["data"]


def norm(x):
    MAX = 1.4200
    MIN = 1.3370
    return (x.clip(MIN, MAX) - MIN) / (MAX - MIN)
