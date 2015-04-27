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


## Image loading and extraction
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

def evaluate_to_img(func, size):
    """Evaluate a function at every grid point in x and y.
    Takes:
        func - f(x,y) which returns a single value
        size - (row, col)
    Returns:
        eval - grid of results
    """
    x_ind = np.arange(0, size[0])
    y_ind = np.arange(0, size[1])
    eval = func(x_ind[:, None], y_ind[None, :])
    return eval


## Geometry
def pt_to_pt_angle(pt1, pt2):
    """What the angle made by the pt to pt vector via arctan2. Pts are y,x."""
    y1, x1 = pt1
    y2, x2 = pt2
    # Find angles
    if x1 < x2:
        return np.arctan2(y2-y1, x2-x1)
    else:
        return np.arctan2(y1-y2, x1-x2)


## Probabilistic distributions
def pearson(size, center, H, K, M):
    """Return an image with a symmetrical Pearson VII distribution.
    This is defined as:
        pear = H * (1 + K**2 * dist**2 / M)**-M
    Where the distance is from the passed center.
    Takes:
        size - (row, col) size of the image to generate
        center - the (row, col) center of the peak
        H - the height of the distribution (should be >0)
        K - controls the spread (should be >0)
        M - controls the rate of decay of the tails (should be >0)
    Returns:
        img - the generated distribution image
    """
    dist = lambda x,y: np.hypot((center[0]-x), (center[1]-y))
    sq = lambda x: np.power(x,2)
    pear = lambda x,y: H * np.power(1 + (sq(K) * sq(dist(x,y))) / M, -M)
    img = evaluate_to_img(pear, size)
    return img

def double_exponential(size, center, a, b, c, d, e):
    """Return a background resulting from a double exponential function.
    This is defined as:
    back = a + b*exp(-dist*c) + d*exp(-dist*e)
    Where the distance is from the passed center.
    Takes:
        size - (row, col) size of the background image to generate
        center - the (row,col) center of the background
        a - the amplitude of the background far from the center
        b,c,d,e - coefficients of the two exponentials 
    Returns:
        img - the generated background image
    """
    dist = lambda x,y: np.hypot((center[0]-x), (center[1]-y))
    exp = lambda x,y,j,k: j * np.exp(-dist(x,y)*k)
    back = lambda x,y: a + exp(x,y,b,c) + exp(x,y,d,e)
    img = evaluate_to_img(back, size)
    return img

def double_exponential_1d(x, a, b, c, d, e): 
    """Return a 1D double exponential function at passed x values"""
    return a + b*np.exp(-x*c) + d*np.exp(-x*e)
