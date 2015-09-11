#!/usr/bin/env python
# encoding: utf-8
"""
muscle_saxs.py - provide top level interface to sax imaging guesses

Created by Dave Williams on 2015-09-11
"""

# System imports
import numpy as np
import matplotlib.pyplot as plt
# Local imports
import img_emcee
from support import image_as_numpy as load_image

def get_segment(img, interactive=False):
    """
    Get the segmentation based guesses for an np image

    Args:
        img (array): the image to get guesses for
        interactive (boolean): default to False, whether to display and allow
            evaluation of diffraction peak localization
    Returns:
        guess_dict (dict): dictionary of guessed parameters
    """
    if interactive: # Set up figure
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=[10,4])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.subplots_adjust(0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    else:
        ax = False
    # The meat of the matter
    guess = img_emcee.gather_guess(img, [False, False, ax])
    ((l_cen, l_fit, l_roi), (r_cen, r_fit, r_roi)) = guess
    # Allow feedback if desired
    if interactive:
        goodbad = raw_input("Good(g), bad(b) or quit(q):")
    else:
        goodbad = np.NaN
    guess_dict =  {'goodbad': goodbad,
                   'l_center': l_cen,
                   'l_fit': l_fit,
                   'l_roi': l_roi,
                   'r_center': r_cen,
                   'r_fit': r_fit,
                   'r_roi': r_roi}
    return guess_dict

def get_mcmc(guess):
    """
    Get the mcmc parameter estimations based on segmentation guesses
    
    Args:
        guess (dict): dictionary of guessed parameters
    Returns:
        sampler_n_locs (dict): samplers and location estimates
    """
    lsampler = img_emcee.sampler(guess['l_roi'], guess['l_fit'],
                               run_steps = 1000)
    rsampler = img_emcee.sampler(guess['r_roi'], guess['r_fit'],
                               run_steps = 1000)
    lloc = img_emcee.location_max_likelyhood(lsampler)
    rloc = img_emcee.location_max_likelyhood(rsampler)
    sampler_n_locs = {'lsampler': lsampler, 
                      'rsampler': rsampler, 
                      'lloc': lloc,
                      'rloc': rloc, 
                      'input': guess} 
    return sampler_n_locs

def get_both(fn=None, img=None, out_fn=None, interactive=False):
    """
    Get both segment guesses and mcmc for an image, optionally save the result. 
    
    Args:
        fn (str): full or relative filename from which to load image
        img (array): pass image as array in liu of fn
        out_fn (str): optional output file to save results to
        interactive (bool): default False, whether to display and allow
            evaluation of diffraction peak localization
    Returns:
        mcmc (dict): guesses and mcmc results, all packed up
    """
    if fn is not None:
        img = load_image(fn) # Load if fn passed
    guess = get_segment(img, interactive)
    mcmc = get_mcmc(guess)
    if fn is not None:
        mcmc['fn'] = fn
    if out_fn is not None:
        pkl.dump(mcmc, open(out_fn, 'wb'))
    return mcmc


