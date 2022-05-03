"""jupyter_utils

Utilities for jupyter notebook visualizations


"""
import os 

import numpy as np
from matplotlib import pyplot as plt

import ipywidgets as widgets
from IPython.display import display


def imshow3(image, **kwargs): 

    """Interactive imshow for 3D images using IPython widgets."""
    try:
        figsize = kwargs.pop('figsize')
    except KeyError:
        figsize = plt.rcParams['figure.figsize']
    try:
        s = kwargs.pop('s')
    except KeyError:
        s = len(image) / 2
    try:
        vmin = kwargs.pop('vmin')
    except KeyError:
        vmin = image.min()
    try:
        vmax = kwargs.pop('vmax')
    except KeyError:
        vmax = image.max()
    try:
        mask = kwargs.pop('mask')
    except:
        mask = None
    cmap = kwargs.pop('cmap', 'gray')

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={
                              'frameon': False, 'xticks': [], 'yticks': []})
    im = ax.imshow(image[s], vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    if mask is not None:
        maskim = ax.imshow(mask[s], alpha=.4,
                           cmap='magma', vmin=0, vmax=1, **kwargs)
    plt.close()

    def imshow3(Slice, Minimum, Maximum):
        im.set_array(image[Slice])
        im.set_clim(vmin=Minimum, vmax=Maximum)
        if mask is not None:
            maskim.set_array(mask[Slice])
            maskim.set_clim(vmin=0, vmax=1)
        display(fig)

    widgets.interact(imshow3,
                     Slice=widgets.IntSlider(
                         value=s, min=0, max=len(image) - 1),
                     Minimum=widgets.FloatSlider(value=vmin, min=image.min(
                     ), max=image.max(), step=(image.max() - image.min()) / 100),
                     Maximum=widgets.FloatSlider(value=vmax, min=image.min(), max=image.max(), step=(image.max() - image.min()) / 100))
