"""
relief.viz
==========
Display helpers for manual inspection. Not used by the pipeline; handy when
looking at results in a notebook or a standalone script.
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


def _prep(img):
    """Prepare any array for imshow: BGR->RGB, uint8->float 0..1."""
    a = np.asarray(img).astype("float32")
    if a.max() > 1.5:
        a = a / 255.0
    if a.ndim == 3:
        a = a[:, :, ::-1]          # BGR -> RGB
    return a


def show_images(images, titles=None, ncols=2, width=15, cmap="gray", suptitle=None):
    """Show a set of images in a grid."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    h, w = images[0].shape[:2]
    fig_h = width / ncols * (h / w) * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, fig_h))
    axes = [axes] if n == 1 else list(np.atleast_1d(axes).ravel())

    for ax, img, i in zip(axes, images, range(n)):
        ax.imshow(_prep(img), cmap=cmap, vmin=0, vmax=1)
        if titles:
            ax.set_title(titles[i], fontsize=11)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig


def show_comparison(bgr, fusion, cavity, width=15):
    """Three-panel comparison: original, fusion, cavity."""
    return show_images([bgr, fusion, cavity], ["Original", "Fusion", "Cavity"],
                       ncols=3, width=width)


def show_crop(images, box, titles=None, ncols=None, width=15):
    """
    Compare one region at 1:1 zoom.

    box : (x, y, w, h) in full-image pixel coordinates.
    """
    x, y, w, h = box
    crops = [im[y:y + h, x:x + w] for im in images]
    return show_images(crops, titles, ncols or len(crops), width)


def show_histogram(images, labels, bins=200, width=10):
    """Value distribution of each image. Useful for checking whether the
    full contrast range is used or values pile up at one end."""
    fig, ax = plt.subplots(figsize=(width, 3.5))
    for img, label in zip(images, labels):
        a = np.asarray(img).astype("float32")
        a = a / 255.0 if a.max() > 1.5 else a
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        ax.hist(a.ravel(), bins=bins, histtype="step", label=label, density=True)
    ax.set_xlim(0, 1)
    ax.set_xlabel("pixel value")
    ax.legend()
    fig.tight_layout()
    return fig
