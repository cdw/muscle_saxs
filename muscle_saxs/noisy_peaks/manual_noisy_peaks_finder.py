#!/usr/bin/env python
# encoding: utf-8
"""
manual_noisy_peak_finder.py - find diffraction peaks in noisy images

NOTE: just experimental at this date

Created by Dave Williams on 2015-01-14
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage, stats, optimize

FN = 'f8_d8.68_e100.png' 
x, y = 295, 98

def load(fn):
    return  np.array(Image.open(open(fn)), dtype=np.float) 

def blur(img):
    return ndimage.filters.gaussian_filter(img, 1, mode='nearest')

def redraw(img):
    pim.set_array(img)
    pim.set_interpolation('nearest')
    pim.set
    plt.draw()

def gaussian(mu_x, mu_y, sig_x, sig_y, amp, back, size):
    """Return a bivariate Gaussian distribution.
    
    Takes:
        mu_x - center of dist(ribution), x
        mu_y - center of dist, y
        sig_x - spread of dist, x
        sig_y - spread of dist, y
        amp - height of center of dist
        back - background level
        size - size of returned matrix, (size,size), both odd or coerced 
               to be such
    Gives:
        gauss_dist - matrix of gaussian distributed values, centered around
                     the (mu_x, mu_y) point
    """
    ## What size will the returned matrix be?
    size_x, size_y = int(size[0]), int(size[1])
    if not size_x%2==1:
        warnings.warn("X size of Gaussian isn't an odd int; lowering.")
        size_x = np.floor(size_x)-1
        assert size_x%2==1
    if not size_y%2==1:
        warnings.warn("Y size of Gaussian isn't an odd int; lowering.")
        size_y = np.floor(size_y)-1
        assert size_y%2==1
    # Create the matrix to feed the distribution
    in_mat = np.dstack(np.meshgrid(
        np.arange(mu_x-0.5*size_x+0.5, mu_x+0.5*size_x+0.5), 
        np.arange(mu_y-0.5*size_y+0.5, mu_y+0.5*size_y+0.5)))
    # Create the possibly asymmetric distribution
    var = stats.multivariate_normal(
        mean = [mu_x, mu_y], 
        cov = [[sig_x,0],[0,sig_y]])
    # Find the amplitude correction
    amp_cor = (amp - back) / var.pdf([mu_x, mu_y])
    # Calculate the actual distribution
    gauss_dist = amp_cor * var.pdf(in_mat) + back
    assert gauss_dist.shape[0] == gauss_dist.shape[1]
    return gauss_dist

def residual(img, distribution, center):
    """Find a residual for the image portion covered by the distribution.
    We assume the distribution isn't hanging off the edge of the image.
    
    Takes:
        img - an image in numpy form
        distribution - an array that we want to subtract from the image
        center - where on the image the distribution is centered, (x,y)
    Gives:
        residual - the absolute difference between the two
    """
    # Distribution boundaries
    ds = distribution.shape
    d_x = (center[0] - 0.5*(ds[0]-1) , center[0] + 0.5*(ds[0]-1))
    d_y = (center[1] - 0.5*(ds[1]-1) , center[1] + 0.5*(ds[1]-1))
    # Check size
    is_y, is_x = img.shape
    assert 1 <= d_x[0]
    assert d_x[1] <= is_x
    assert 1 <= d_y[0]
    assert d_y[1] <= is_y
    #  Slice and subtract
    img_slice = img[d_y[0]-1:d_y[1], d_x[0]-1:d_x[1]]
    return np.sum(np.abs(img_slice - distribution))

def optimize(center, img):
    """Find the center of the gaussian given a starting location"""
    ## Define parameters and initial guesses
    dist_size = 11 # distribution x/y size in pixels
    peak_search = dist_size # possible peak distance from passed center
    c_x, c_y = center
    sig_x, sig_y = 5, 5
    sig_max = 40
    ## Find the amplitude and background
    amp = np.max(img[c_y-peak_search/2 : c_y+peak_search/2, 
                     c_x-peak_search/2 : c_x+peak_search/2])
    back = np.min(img[c_y-peak_search/2 : c_y+peak_search/2, 
                      c_x-peak_search/2 : c_x+peak_search/2])
    ## Create the penalty function
    def func(vals, amp, back, size, img):
        mu_x, mu_y, sig_x, sig_y = vals
        dist = gaussian(mu_x, mu_y, sig_x, sig_y, amp, back, size)
        assert dist.shape[0] == dist.shape[1]
        return residual(img, dist, (mu_x, mu_y))
    ## Find the centers
    opt_result = optimize.minimize(func, (c_x, c_y, sig_x, sig_y),
        args = (amp, back, (dist_size, dist_size), img),
        method = 'L-BFGS-B',
        bounds = ((c_x-peak_search/2, c_x+peak_search/2),
                  (c_y-peak_search/2, c_y+peak_search/2),
                  (1, sig_max), (1, sig_max)))

def img_to_feature_array(img):
    output = []
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if not img[y,x] == 0:
                output.append([x,y])
    return np.array(output)

