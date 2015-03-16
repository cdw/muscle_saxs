#!/usr/bin/env python
# encoding: utf-8
"""
img_emcee.py - snippits of code to test emcee with our x-ray images

Created by Dave Williams on 2015-02-02
"""

# System imports
import emcee
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
    
    
 p0 = np.array((
        57,   # mask center row
        251,  # mask center col
        15,   # mask radius
        57,   # diffraction center row
        251,  # diffraction center col
        14,   # background a
        220,  # background b
        0.11, # background c
        3.15, # background d
        0.36, # background e
        85,   # d10 spacing
        15,   # d10 angle
        1054, # d10 height
        0.3,  # d10 spread
        2.3,  # d10 decay
        169,  # d20 spacing
        267,  # d20 height
        4,    # d20 spread
        0.3)) # d20 decay


def img_prior():
    """Need one"""
    return None

def img_likelihood():
    """Need one"""
    return None

def img_probability():
    """Need one"""
    return None


## Test if run directly
def main():
    samplefn = 'sampleimg1.tif'
    data = support.image_as_numpy(samplefn)

if __name__ == '__main__':
	main()

