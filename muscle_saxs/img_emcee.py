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
import xray_background as xbkg
import blocked_region
import peak_finder as peakf
import fiber_lines as flines
import support 

def gather_guess(img):
    """From an image, gather the initial guesses to feed MCMC optimization.
    Takes:
        img: the image of interest
    Returns:
        p0: an array of parameters to jiggle for initial walker positions
    """
    # Parameter settings, may need to jiggle/pass directly in future
    roi_size = 8
    peak_range = (4,12)
    # Process peaks out of image
    center, radius = blocked_region.find_blocked_region(img, plot=True)
    unorg_peaks = peakf.peaks_from_image(img, (center, radius),
                                         peak_range=peak_range, plot=False)
    pairs = peakf.extract_pairs(center, unorg_peaks, plot=True, pimg=img)
    d10 = pairs[0]
    # Find diffraction lines and subtract background
    success, thetas, clus_peaks = flines.optimize_thetas(center, unorg_peaks)
    theta = thetas[np.argmax(clus_peaks)]
    img_no_bkg = xbkg.find_and_remove_background(center, radius, center, 
                                                 img, [theta])
    rois = [peakf.img_roi(ploc, img_no_bkg, roi_size) for ploc in d10]
    peak_fits = [peakf.fit_peak((roi_size, roi_size), r) for r in rois]
    return zip(d10, peak_fits, rois)

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

