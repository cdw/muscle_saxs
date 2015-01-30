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
import cv2

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
    exp = lambda x,y,j,k: j * np.exp(-dist(x,y)*k)
    back = lambda x,y: a + exp(x,y,b,c) + exp(x,y,d,e)
    img = evaluate_to_img(back, size)
    return img

def pearson(size, center, H, K, M):
    """Return an image with a symmetrical Pearson VII distribution.
    This is defined as:
        pear = H * (1 + K**2 * dist**2 / M)**-M
    Where the distance is from the passed center.
    Takes:
        size - (row, col) size of the image to generate
        center - the (row, col) center of the background image
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

def masking(size, center, radius):
    """Create a circular masking image, with 0s in the masked region.. 
    The intended use is as `img*masking(img.shape, center, radius)`
    Takes:
        size - (row, col) size of the masked image
        center - the (row, col) center of the masked region
        radius - the diameter of the masked region in pixels
    Gives:
        img - the image after masking
    """
    # NOTE: this is not an anti-aliased circle, despite telling opencv to do
    # it that way, this should later be fixed to be such using Wu's method.
    centercv = (center[1], center[0]) # cv2 likes col, row
    mask = np.ones(size)
    cv2.circle(mask, centercv, radius, 0, -1, cv2.cv.CV_AA)
    # img to work on
               #np.transpose(center), # center in cv2 is col,row
               #radius,               # masking radius
               #0,                    # value to draw in circle
               #-1,                   # negative value indicates infill
               #cv2.cv.CV_AA)         # says to anti-alias, but doesn't
    return mask


