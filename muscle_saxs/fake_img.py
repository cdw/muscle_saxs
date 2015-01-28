#!/usr/bin/env python
# encoding: utf-8
"""
xray_background.py - remove background from small angle x-ray scattering imgs

Created by Dave Williams on 2014-10-09
"""

import os, sys
import warnings
import numpy as np
from PIL import Image

def blank(size):
    """Given a (row, col) size, create a blank numpy array"""
    return np.zeros(size)

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


def background(size, center, a, b, c, d, e):
    """Return a background resulting from a double exponential function.
    This is defined as:
    back = a + exp((dist-b)/c) + exp((dist-d)/e)
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
    exp = lambda x,y,j,k: np.exp( (dist(x,y)-j) / k)
    back = lambda x,y: a + exp(x,y,b,c) + exp(x,y,d,e)
    img = evaluate_to_img(back, size)
    return img
