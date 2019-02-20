from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy


def write_image(img, centroids, filename, color=None, thickness=None):
    # process args
    if color is None:
        color = (0, 0, 255)
    if thickness is None:
        thickness = -3
    # make markup image
    img = copy.deepcopy(img)
    for i, c in enumerate(centroids):
        cv2.circle(img, (c["x"], c["y"]), 3, color, thickness=thickness)
    cv2.imwrite(filename, img)


def write_histogram(data, filename, title=None, xlabel=None, ylabel=None, xlim=None, bins=None, colormap=None, figsize=None):
    # process args
    if title is None:
        title = "Grain Area Distribution"
    if xlabel is None:
        xlabel = "Area ($\mu m^2$)"
    if ylabel is None:
        ylabel = "Count"
    if xlim is None:
        xlim = (min(data), max(data))
    if bins is None:
        bins = 25
    if colormap is None:
        colormap = plt.cm.viridis
    if figsize is None:
        figsize = (6, 8)
    # make plot
    _, ax = plt.subplots(tight_layout=True, figsize=figsize)
    n, _, patches = ax.hist(data, bins=bins)
    fractions = n/n.max()
    _color_normalization(fractions, patches, colormap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_xlim(xlim)
    plt.savefig(filename)


def _color_normalization(fractions, patches, colormap):
    norm = colors.Normalize(fractions.min(), fractions.max())
    for f, p in zip(fractions, patches):
        color = colormap(norm(f))
        p.set_facecolor(color)
