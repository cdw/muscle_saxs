#!/usr/bin/env python
# encoding: utf-8
"""
xray_background.py - remove background from small angle x-ray scattering imgs

Created by Dave Williams on 2014-10-09
"""

# System imports
import numpy as np
from scipy import optimize
import cv2
import matplotlib.pyplot as plt


## Find the background profile and fit it

def background_collapse(center, img, thetas, plot=False):
    """Collapse the image background, ignoring the peak regions.
    Good ideas to be had here: http://goo.gl/2xEApw
    Takes:
        center: x,y center of blocked image
        img: from which background is extracted
        thetas: angles of lines we wish to miss
        plot: if we should plot the exclusion regions and profile
    Gives:
        background: profile of background
        background_dists: pixel distances of background from center
    """
    ## With shifting, find the masking region
    mask = np.ones((img.shape[0], img.shape[1]*2), dtype=np.float)
    m_center = (int(round(center[0] + img.shape[1])), int(round(center[1])))
    m_thetas = np.concatenate(([t+np.pi for t in thetas], thetas))
    m_thetas = np.round(np.degrees(m_thetas)).astype(np.int)
    theta_pm = 12 # amount to block on either side
    m_angles = [(t-theta_pm, t+theta_pm) for t in m_thetas] # angles to block
    m_axes = (img.shape[1], img.shape[1]) # should always fill screen
    for angle in m_angles:
        cv2.ellipse(mask, m_center, m_axes, 180, angle[0], angle[1], 0, -1)
    mask = mask[:,img.shape[1]:]
    # Construct a radial distance img
    row, col = np.indices(img.shape)
    r = np.sqrt((col-center[0])**2 + (row-center[1])**2)
    # Coerce into ints for bincount
    r = r.astype(np.int)
    img = img.astype(np.int)
    img = img*mask
    # Do the counting
    flat_count = np.bincount(r.ravel(), img.ravel())
    rad_occurances = np.bincount(r.ravel()) 
    radial_profile = flat_count/rad_occurances
    # Kill the blocked region
    highest_ind = radial_profile.argmax()
    background = radial_profile[highest_ind:]
    background_dists = np.arange(highest_ind,len(radial_profile))
    # Plot if passed
    if plot is True:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[6,6])
        ax1.scatter(center[0], center[1], color='m')
        ax1.imshow(mask*img) 
        ax2.plot(background, linewidth=3)
        ax1.set_title("Masked image for profiling")
        ax2.set_title("Resulting radial profile")
        plt.draw()
        plt.tight_layout()
        plt.show()
    return background, background_dists

def fit_double_exp(trace_y, trace_x, plot=False):
    """Fit a double exponential function to the passed trace.
    Ignore the region to the left of the peak. 
    Takes:
        trace: a nx1 data trace
    Gives:
        vals: optimized parameters for a double exp
    """
    # A residual function to test how good our fits are
    dexp = double_exponential_1d 
    diff = lambda i, j: np.sum(np.abs(np.subtract(i,j)))
    resi = lambda g: diff(dexp(trace_x, g[0], g[1], g[2], g[3], g[4]), trace_y)
    # Guess some values then optimize
    guess = [1.0, 1000.0, 0.01, 5000.0, 0.1]
    opt_res = optimize.minimize(resi, guess, jac=False, bounds = (
        (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)))
    success = opt_res['success']
    vals = opt_res['x']
    # Plot if desired
    if plot is True:
        fig, ax = plt.subplots(figsize=[6,3])
        plt.plot(trace_x, dexp(trace_x, *zip(vals)), 'c', linewidth=3)
        plt.plot(trace_x, trace_y, 'r', linewidth=3)
        ax.set_title("Real (r) and fitted (c) values")
        plt.draw()
        plt.tight_layout()
        plt.show()
    return vals


## Test if run directly
def main():
    import support
    import peak_finder
    SAMPLEFILE = 'sampleimg1.tif'
    data = support.image_as_numpy(SAMPLEFILE) # load
    block = find_blocked_region(data, True) # find blocker
    unorg_peaks = peak_finder.peaks_from_image(data, block, plot=True)
    success, thetas, peaks = peak_finder.optimize_thetas(block[0], unorg_peaks)
    back, back_dists = background_collapse(block[0], data, thetas, True)
    back_params = fit_double_exp(back, back_dists, True)

if __name__ == '__main__':
	main()

