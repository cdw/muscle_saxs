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
# Local module imports
import support
import fake_img


## Find the background profile and fit it

def background_collapse(center, img, peaks, plot=False):
    """Collapse the image background, ignoring the peak regions.
    Good ideas to be had here: http://goo.gl/2xEApw
    
    Args:
        center: x,y center of blocked image
        img: from which background is extracted
        peaks: row,col locations of peaks; don't want these in the background
        plot: if we should plot the exclusion regions and profile (True/False)
              or a list of two axes to plot onto
    Gives:
        background: profile of background
        background_dists: pixel distances of background from center
    """
    #import ipdb; ipdb.set_trace()
    ## Find peak angles
    cx, cy = center
    #m_thetas = [support.pt_to_pt_angle((cy, cx), pt) for pt in peaks]
    m_thetas = [np.arctan2(pt[0] - cy, pt[1] - cx) for pt in peaks]
    ## With shifting, find the masking region
    mask = np.ones((img.shape[0], img.shape[1]*2), dtype=np.float)
    m_center = (int(round(center[0] + img.shape[1])), int(round(center[1])))
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
    if plot is not False:
        if plot is True:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[6,6])
        else:
            ax1, ax2 = plot
        ax1.scatter(center[0], center[1], color='m')
        ax1.imshow(mask*img) 
        colors = list(np.tile(
            ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '.25', '.5', '.75'], 5))
        for peak in peaks:
            c = colors.pop(0)
            ax1.scatter(peak[1], peak[0], c=c, s=40)
        ax2.plot(background, linewidth=3)
        ax1.set_title("Masked image for profiling")
        ax2.set_title("Resulting radial profile")
        plt.draw()
        plt.tight_layout()
        plt.show()
    return background_dists, background

def _fit_double_exp(trace_y, trace_x, plot=False):
    """Fit a double exponential function to the passed trace.
    Ignore the region to the left of the peak. 
    Takes:
        trace_y: a nx1 data trace
        trace_x: the x indices that go with trace_y
        plot: whether or not to plot the fit
    Gives:
        vals: optimized parameters for a double exp
    """
    # A residual function to test how good our fits are
    dexp = support.double_exponential_1d 
    diff = lambda i, j: np.sum(np.abs(np.subtract(i,j)))
    resi = lambda g: diff(dexp(trace_x, g[0], g[1], g[2], g[3], g[4]), trace_y)
    # Guess some values then optimize
    guess = [1.0, 1000.0, 0.01, 5000.0, 0.1]
    opt_res = optimize.minimize(resi, guess, jac=False, bounds = ( (0, np.inf), (0, np.inf), (0, 1), (0, np.inf), (0, 1)))
    success = opt_res['success']
    vals = opt_res['x']
    # Plot if desired
    if plot is not False:
        if plot is True:
            fig, ax = plt.subplots(figsize=[6,3])
        else:
            ax = plot
        plt.plot(trace_x, dexp(trace_x, *zip(vals)), 'c', linewidth=3)
        plt.plot(trace_x, trace_y, 'r', linewidth=3)
        ax.set_title("Real (r) and fitted (c) values")
        plt.draw()
        plt.tight_layout()
        plt.show()
    return vals


## Generate a fake background and subtract it from a passed image

def _fake_background(size, mask_center, mask_rad, diff_center, back_vals):
    """Generate a fake background image from the passed (fitted) values.
    
    Args:
        size (tuple): (row, col) size of image to generate
        mask_center: the center of the masked region
        mask_rad: the radius of the masked region
        diff_center: the center of the diffraction (and background) pattern
        back_vals (iter): the (a,b,c,d,e) values of the double exponential
    
    Returns:
        img: the fake background image
    """
    # Flips and unpacks
    a, b, c, d, e = back_vals
    mask_center = (mask_center[1], mask_center[0])
    diff_center = (diff_center[1], diff_center[0])
    exp_img = fake_img.background(size, diff_center, a, b, c, d, e)
    mask_img = fake_img.masking(size, mask_center, mask_rad)
    return exp_img*mask_img

def find_and_remove_background(mask_cen, mask_rad, diff_cen, img, peaks,
                               plot=False):
    """Fit/subtract the background of an image and the peaks of its angles.
    
    Args:
        mask_cen: the center of the masking region
        mask_rad: the radius of the masking region
        diff_cen: the center of the diffraction pattern
        img: the image whose background we're interested in
        peaks: the peaks we want to exclude (at least one)
        plot: to plot the masks and fit or not (True/False) or a list of 
              three axes to plot onto
        
    Returns:
        img: img-background, to best of abilities
    """
    # Plot set up
    if plot is not False:
        if plot is True:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[6,9])
            ax12 = (ax1, ax2)
        else:
            ax12 = plot[0], plot[1]
            ax3 = plot[2]
    else:
        ax12, ax3 = False, False
    size = img.shape
    back_x, back_y = background_collapse(diff_cen, img, peaks, plot=ax12)
    fits = _fit_double_exp(back_y, back_x, plot=ax3)
    fake_back_img = _fake_background(size, mask_cen, mask_rad, diff_cen, fits)
    return img-fake_back_img


## Use peaks to find info about image
def find_diffraction_center(pairs, which_list='longest'):
    """ Find the diffraction center based off of pairs of points.
    By default, use the longest list of pairs.
    Takes:
        pairs: lists of point pairs, output of extract_pairs
        which_list: "longest" or index location in pairs
    Gives:
        center: row,col center of the diffraction image
    """
    # Which pair list to use
    if which_list == 'longest':
        which_list = np.argmax(map(len, pairs))
    # Find mean middle point
    mid = lambda pair: np.add(np.subtract(pair[0], pair[1])/2.0, pair[1])
    center = np.mean([mid(p) for p in pairs[which_list]], 0)
    return center



## Test if run directly
def main():
    import support
    import peak_finder
    SAMPLEFILE = 'sampleimg1.tif'
    data = support.image_as_numpy(SAMPLEFILE) # load
    block = find_blocked_region(data, True) # find blocker
    unorg_peaks = peak_finder.peaks_from_image(data, block, plot=True)
    success, thetas, peaks = peak_finder.optimize_thetas(block[0], unorg_peaks)
    back_dists, back = background_collapse(block[0], data, thetas, True)
    back_params = _fit_double_exp(back, back_dists, True)

if __name__ == '__main__':
	main()

