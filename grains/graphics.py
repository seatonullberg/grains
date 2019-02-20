from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy


def write_centroids_image(img, centroids, filename, color=None, thickness=None, radius=None):
    """Writes an image with circles superimposed on the centroids of the original image.
    
    Args:
        img:       image array
        centroids: OpenCV centroids
        filename:  output filename for the modified image
        color:     BGR tuple for circle color
        thickness: circle thickness (< 0 for solid or > 0 for hollow)
        radius:    circle radius
    """
    
    # process args
    if color is None:
        color = (0, 0, 255)
    if thickness is None:
        thickness = -3
    if radius is None:
        radius = 3
    # make markup image
    img = copy.deepcopy(img)
    for i, c in enumerate(centroids):
        cv2.circle(img, (c["x"], c["y"]), radius, color, thickness=thickness)
    cv2.imwrite(filename, img)


def write_histogram(data, filename, title=None, xlabel=None, ylabel=None, xlim=None, bins=None, colormap=None, figsize=None):
    """Writes a histogram plot.
    
    Args:
        data:     1D array
        filename: output filename
        title:    passed to plt.title()
        xlabel:   passed to plt.xlabel()
        ylabel:   passed to plt.ylabel()
        xlim:     passed to ax.set_xlim()
        * no ylim - let matplotlib determine this
        bins:     passed to ax.hist()
        colormap: a plt.cm used to style the histogram bars
        figsize:  passed to plt.subplots()
    """
    
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
    """Assigns colors to histogram bars.
    
    Args:
        fractions: 0-1 value used to map a color intensity to a bar height
        patches:   list of histogram bars (? unsure of the exact type ?)
        colormap:  plt.cm used to assign colors
    """
    
    norm = colors.Normalize(fractions.min(), fractions.max())
    for f, p in zip(fractions, patches):
        color = colormap(norm(f))
        p.set_facecolor(color)
