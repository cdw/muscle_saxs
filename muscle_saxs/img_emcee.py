#!/usr/bin/env python
# encoding: utf-8
"""
img_emcee.py - snippits of code to test emcee with our x-ray images

Created by Dave Williams on 2015-02-02
"""

# System imports
import emcee
import numpy as np
import matplotlib.pyplot as plt
# Local imports
import fake_img as fimg
import xray_background as back
import peak_finder as peakf
import support 

def gather_guess(img):
    """From an image, gather the initial guesses to feed MCMC optimization.
    Takes:
        img: the image of interest
    Returns:
        p0: an array of parameters to jiggle for initial walker positions
    """
    block_center, block_rad = back.find_blocked_region(img, plot=True)
    unorg_peaks = peakf.peaks_from_image(img, plot=True)
    success, thetas, peaks = peakf.optimize_thetas(block_center, unorg_peaks,
                                                  plot=True, pimg=img)
    

def img_prior():
    """Need one"""
    return None

def img_likelihood():
    """Need one"""
    return None

def img_probability():
    """Need one"""
    return None

def peak_probability(crcchkm, real_peak):
    """Log prob that a generated peak comes from the data's distribution. 
    
    Args:
        cxcyhkm (tuple): the parameters that define our model peak, in the
            form of (center_row, center_col, H, K, M) where the latter 3 are
            described under support.pearson
        real_peak (array): the data we are comparing to
    Returns:
        ln_prob (float): the log(prob) that the data came from a distribution
            centered on the model, treating each pixel as a Poisson process
    """
    ## Generate model image
    cen_row, cen_col, H, K, M = crcchkm
    size_row, size_col = real_peak.shape
    model = support.pearson((size_row, size_col), (cen_row, cen_col), H, K, M)
    ## Enforce limits
    limits = (cen_row < 0,
              cen_row > size_row, 
              cen_col < 0, 
              cen_col > size_col,
              H <= 0, 
              K <= 0, 
              M <= 0)
    if any(limits):
        return -np.inf
    ## Compare model to data
    prob = fimg.pixel_difference_log_prob(model, real_peak)
    ## Return sum
    return np.sum(prob)


## Test if run directly
def main():
    samplefn = 'sampleimg1.tif'
    data = support.image_as_numpy(samplefn)

if __name__ == '__main__':
	main()

