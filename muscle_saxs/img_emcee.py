#!/usr/bin/env python
# encoding: utf-8
"""
img_emcee.py - snippits of code to test emcee with our x-ray images

Created by Dave Williams on 2015-02-02
"""

# System imports
import emcee
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
# Local imports
import fake_img as fimg
import xray_background as xbkg
import blocked_region
import peak_finder as peakf
import fiber_lines as flines
import support 

def gather_guess(img, show_plots=False):
    """From an image, gather the initial guesses to feed MCMC optimization.
    Takes:
        img: the image of interest
        show_plots: whether to show intermediate plots or pass an axis to use
    Returns:
        p0: an array of parameters to jiggle for initial walker positions
    """
    # Parameter settings, may need to jiggle/pass directly in future
    roi_size = 8
    peak_range = (4,12)
    # Allow plotting to sequential or individual axes
    if show_plots is False:
        plt1, plt2, plt3 = False, False, False
    elif show_plots is True:
        plt1, plt2, plt3 = True, True, True
    elif not hasattr(show_plots, '__iter__'):
        plt1, plt2, plt3 = show_plots, show_plots, show_plots # single axis
    else:
        plt1, plt2, plt3 = show_plots # list of axes
    # Process peaks out of image
    block = blocked_region.find_blocked_region(img, plot=False)
    center, radius = block
    unorg_peaks = peakf.peaks_from_image(img, block, peak_range=peak_range, 
                                         plot=plt1)
    pairs = peakf.extract_pairs(block, unorg_peaks, plot=plt2, pimg=img)
    d10 = peakf.extract_d10(pairs, horizontal=True, plot=plt3, pimg=img)
    # Find diffraction lines and subtract background
    img_no_bkg = xbkg.find_and_remove_background(center, radius, center, 
                                                 img, unorg_peaks)
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

def sampler(roi, peak_fit, walkers=200, burn_steps=100, run_steps=1000):
    """Run a sampler ensemble on the passed region of interest.
    
    Args:
        roi (array): selected region of interest containing a peak
        peak_fit (tuple): starting peak parameters (H, K, M). Samplers' 
            initial locations will be jiggled about this position (and the
            center of the ROI).
        walkers (int): the number of walkers to explore the space (200)
        burn_steps (int): number of steps to take in burn_in settling period
        run_steps (int): number of steps to run for parameter exploration
    Returns:
        chain (array): sampler chain
    """
    initial_pos = [roi.shape[0]/2+1, roi.shape[1]/2+1]
    initial_pos.extend(peak_fit)
    perturbing_stds = [0.4, 0.4, 5, 0.1, 0.1] 
    p0 = emcee.utils.sample_ball(initial_pos, perturbing_stds, walkers)
    sampler = emcee.EnsembleSampler(walkers, 
                                    len(initial_pos), 
                                    peak_probability, 
                                    args = [roi]) 
    burn_in = sampler.run_mcmc(p0, burn_steps)
    post_burn_pos = burn_in[0]
    sampler.reset()
    sampler.run_mcmc(post_burn_pos, run_steps)
    return sampler

def histograms(sampler):
    """Plot histograms of sample distributions."""
    fig, (a1, a2, a3, a4, a5) = plt.subplots(5, 1, figsize=(4,6))
    a1.set_ylabel("Row")
    a2.set_ylabel("Col")
    a3.set_ylabel("H")
    a4.set_ylabel("K")
    a5.set_ylabel("M")
    a1.hist(sampler.flatchain[:,0], 100, normed=True, histtype='step')
    a2.hist(sampler.flatchain[:,1], 100, normed=True, histtype='step')
    a3.hist(sampler.flatchain[:,2], 100, normed=True, histtype='step')
    a4.hist(sampler.flatchain[:,3], 100, normed=True, histtype='step')
    a5.hist(sampler.flatchain[:,4], 100, normed=True, histtype='step')
    plt.draw()
    plt.tight_layout()
    plt.draw()
    return

def location_max_likelyhood(sampler):
    """Row and col of max likelihood"""
    return sampler.flatchain.mean(0)[:2]

def location_interval(sampler, interval=[16,84]):
    """Find the row, col location and its surrounding confidence interval.

    Args:
        sampler: from which to draw the locations
        interval (list): the limits on the confidence interval
    Returns:
        intervals (list): [(row, row_plus, row_minus), 
                           (col, col_plus, col_minus)]
    """
    lo, hi = interval
    percentiles = zip(*np.percentile(sampler.flatchain, [lo, 50, hi], axis=0))  
    plus_minus = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), percentiles)
    return plus_minus[:2]

## Test if run directly
def main():
    samplefn = 'sampleimg1.tif'
    data = support.image_as_numpy(samplefn)

if __name__ == '__main__':
	main()

