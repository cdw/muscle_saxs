#!/usr/bin/env python
# encoding: utf-8
"""
support.py - support functions used in many parts of the other modules 

Created by Dave Williams on 2015-02-19
"""

import os, sys
import warnings
import numpy as np
from PIL import Image


def image_as_numpy(filename):
    """Load an image and convert it to a numpy array of floats."""
    return np.array(Image.open(filename), dtype=np.float) 

def define_circle_points(center, radius):
    """Create a list of pixel locations to look at on a circle.
    Note that this isn't aliased etc.
    """
    res = np.pi/radius # set resolution to avoid double counting a pixel
    x = center[0] + np.round(radius * np.cos(np.arange(-np.pi, np.pi, res)))
    y = center[1] + np.round(radius * np.sin(np.arange(-np.pi, np.pi, res)))
    return x, y

def yank_circle_pixels(img, center, radius):
    """Return a list of pixel values in a circle from the passed image."""
    x, y = _define_circle_points(center, radius) 
    ## Filter out out-of-bounds points
    yx = zip(y, x)  # yx b/c row,column
    y_max, x_max = img.shape
    inbounds = lambda yx: 0 <= yx[0] <= y_max and 0 <= yx[1] <= x_max
    yx_inbounds = filter(inbounds, yx)
    if len(yx) != len(yx_inbounds):
        warnings.warn("Circle is clipped by image limits.")
    ## Find pix
    pix = [img[yx] for yx in yx_inbounds]
    return pix


