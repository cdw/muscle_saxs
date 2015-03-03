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
import support


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
    return support.double_exponential(size, center, a, b, c, d, e)

def pearson(size, center, H, K, M):
    """Create a Pearson VII peak from the support functions.
    Takes:
        size - (row, col) size of the image to generate
        center - the (row, col) center of the background image
        H - the height of the distribution (should be >0)
        K - controls the spread (should be >0)
        M - controls the rate of decay of the tails (should be >0)
    Returns:
        img - the generated distribution image
    """
    return support.pearson(size, center, H, K, M)

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
    
    Poisson process in form
        P(R|L) = e**-L * (L**R / R!)
    Where we are looking for the likelihood of the data given the model
    Taking log thereof:
        Log(P(R|L)) = Log(e**-L * (L**R / R!))
        Log(P(R|L)) = -L + R*Log(L) - Log(R!)
    
    Takes:
        model: the idealized model we assume underlies a phenomenon
        data: the recorded pixel data value
    Returns:
        prob: log of the probability of the difference being observed
    """
    ## make sure we're dealing with floats/ints
    if not type(data)==int and not data.dtype==int:
        data = data.astype(int)
    if type(model)==int or model.dtype==int:
        model = model.astype(float)
    d, m = data, model
    # Take the cases where model=0 and set to model=1e-100 as 
    # Poisson processes are constrained to have L>0
    m[np.nonzero(m==0)] = 1e-100
    # Find the probabilities of the pixel values
    log_rf = d*np.log(d) - d + np.log(d*(1+4*d*(1+2*d)))/6 + np.log(np.pi)/2
    log_rf[np.nonzero(data==0)] = np.log(np.math.factorial(0)) #approx breaks
    prob = -m + d*np.log(m) - log_rf
    # Return the negative log likelihood
    #prob = -prob
    return prob

def lnprob(p, size_row, size_col, real_img):
    (mask_center_row, mask_center_col, mask_rad,
     diff_center_row, diff_center_col, 
     back_a, back_b, back_c, back_d, back_e, 
     d10_spacing, d10_angle, d10_height, d10_spread, d10_decay, 
     d20_spacing, d20_height, d20_spread, d20_decay) = p
    model = fake_img((size_row, size_col), 
        (mask_center_row, mask_center_col), mask_rad, 
        (diff_center_row, diff_center_col), 
        back_a, back_b, back_c, back_d, back_e, 
        d10_spacing, d10_angle, d10_height, d10_spread, d10_decay,
        d20_spacing, d20_height, d20_spread, d20_decay) 
    data = real_img
    #prob = np.sum(pixel_difference_log_prob(model, data))
    prob = np.sum(pixel_difference_log_prob(model, data))
    return prob


## Test if run directly
def main():
    # Set up a sample run
    fn = 'sampleimg1.tif'
    import emcee
    data = support.image_as_numpy(fn)
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
    nwalkers = 100
    p0s = np.array([p0*np.random.uniform(.8,1.2) for i in range(nwalkers)])
    # Run a sample run
    sampler = emcee.EnsembleSampler(100, len(p0), lnprob, 
                                    args = [195, 487, data])
    walker_out = sampler.run_mcmc(p0s, 10)
    # Plot the parameter distributions
    ndim= len(p0)
    for i in range(ndim):
        plt.figure()
        plt.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
        plt.title("Dimension {0:d}".format(i))

    plt.show()

if __name__ == '__main__':
	main()
## Test Run

