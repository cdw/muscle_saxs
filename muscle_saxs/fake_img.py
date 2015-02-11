#!/usr/bin/env python
# encoding: utf-8
"""
fake_img.py - pass parameters and generate a fake x-ray diffraction 
              image from them 

Created by Dave Williams on 2015-01-21
"""

import os, sys
import warnings
import numpy as np
from scipy.misc import factorial
from PIL import Image
import cv2

## Support/utility functions
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

## Image part generation
def background(size, center, a, b, c, d, e):
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
    radius = int(radius)
    centercv = (int(center[1]), int(center[0])) # cv2 likes col, row
    mask = np.ones(size)
    cv2.circle(mask, centercv, radius, 0, -1, cv2.cv.CV_AA)
    # img to work on
               #np.transpose(center), # center in cv2 is col,row
               #radius,               # masking radius
               #0,                    # value to draw in circle
               #-1,                   # negative value indicates infill
               #cv2.cv.CV_AA)         # says to anti-alias, but doesn't
    return mask

## Whole image generation and residuals
def fake_img(size, mask_center, mask_rad, 
             diff_center, back_a, back_b, back_c, back_d, back_e, 
             d10_spacing, d10_angle, d10_height, d10_spread, d10_decay,
             d20_spacing, d20_height, d20_spread, d20_decay):
    """Generate a fake image, for comparison with a real one.
    """
    # Background first
    img = background(size, diff_center, 
                     back_a, back_b, back_c, back_d, back_e)
    # Now the d_10 peaks
    row_delta = lambda ang, space: np.sin(np.radians(ang)) * 0.5 * space
    col_delta = lambda ang, space: np.cos(np.radians(ang)) * 0.5 * space
    d10_row_delta = row_delta(d10_angle, d10_spacing)
    d10_col_delta = col_delta(d10_angle, d10_spacing)
    d10_center_r = (diff_center[0] + d10_row_delta, 
                    diff_center[1] + d10_col_delta)
    d10_center_l = (diff_center[0] - d10_row_delta, 
                    diff_center[1] - d10_col_delta)
    d10_r = pearson(size, d10_center_r, d10_height, d10_spread, d10_decay)
    d10_l = pearson(size, d10_center_l, d10_height, d10_spread, d10_decay)
    # Now the d_20 peaks
    d20_row_delta = row_delta(d10_angle, d20_spacing)
    d20_col_delta = col_delta(d10_angle, d20_spacing)
    d20_center_r = (diff_center[0] + d20_row_delta, 
                    diff_center[1] + d20_col_delta)
    d20_center_l = (diff_center[0] - d20_row_delta, 
                    diff_center[1] - d20_col_delta)
    d20_r = pearson(size, d20_center_r, d20_height, d20_spread, d20_decay)
    d20_l = pearson(size, d20_center_l, d20_height, d20_spread, d20_decay)
    # Now combine and mask
    img = img + d10_r + d10_l + d20_r + d20_l
    img *= masking(size, mask_center, mask_rad)
    return img

def img_diff(real_img, *args):
    return np.sum(np.abs(np.subtract(real_img, fake_img(*args))))

def no_tuples_img_diff((mask_center_row, mask_center_col, mask_rad,
                       diff_center_row, diff_center_col, back_a, back_b,
                       back_c, back_d, back_e, d10_spacing, d10_angle,
                       d10_height, d10_spread, d10_decay, d20_spacing,
                       d20_height, d20_spread, d20_decay), size_row,
                       size_col, real_img):
    return img_diff(real_img, 
                    (size_row, size_col), 
                    (mask_center_row, mask_center_col), mask_rad,
                    (diff_center_row, diff_center_col), 
                    back_a, back_b, back_c, back_d, back_e,
                    d10_spacing, d10_angle, d10_height, 
                    d10_spread, d10_decay, 
                    d20_spacing, d20_height, d20_spread, d20_decay)

## Probabilistic image matching
def pixel_difference_log_prob(model, data):
    """The probability of a data value given an underlying model.
    Note: Assumes pixel data counting is a Poisson process.
    Takes:
        model: the idealized model we assume underlies a phenomenon
        data: the recorded pixel data value
    Returns:
        prob: log of the probability of the difference being observed
    """
    # Poisson process in form
    # P(R|L) = e**-L * (L**R / R!)
    # Where we are looking for the likelihood of the data given the model
    # Taking log thereof:
    # Log(P(R|L)) = Log(e**-L * (L**R / R!))
    # Log(P(R|L)) = -L + R*Log(L) - Log(R!)
    if type(data)==int or data.dtype==int:
        data = data.astype(int)
    d, m = data, model
    log_rf = d*np.log(d) - d + np.log(d*(1+4*d*(1+2*d)))/6 + np.log(np.pi)/2
    prob = -m + d*np.log(m) - log_rf
    # Take cases where data==0, assign prob to be zero
    prob[np.nonzero(np.isnan(prob))] = 0
    # Return the negative log likelihood
    prob = -1 * prob
    return prob



