#!/usr/bin/env python
# encoding: utf-8
"""
test_script.py - make sure that our analysis works

Created by Dave Williams on 2015-03-10
"""

# System imports
import support
import time
import os
import emcee
import numpy as np
import matplotlib.pyplot as plt
# Local imports
import peak_finder as peakf
import blocked_region
import fiber_lines
import xray_background as xbkg
import fake_img as fimg
import img_emcee as imemcee
import support

# Test of d10 locating
img_dir = '/Users/dave/Desktop/argonne_2013/img_averages/non_precessive_means/'
fns = os.listdir(img_dir)
fns = filter(lambda s:s.endswith('.tif'), fns)
plt.ion()


# Trad bits
fn = fns[15]
img = support.image_as_numpy(img_dir+fn)
center, radius = blocked_region.find_blocked_region(img, False)
points = peakf.peaks_from_image(img, (center, radius), plot=False)
success, thetas, clus_peaks = fiber_lines.optimize_thetas(center, points, plot=False, pimg=img)
theta = thetas[np.argmax(clus_peaks)]
pairs = peakf.extract_pairs(center, points, True, img)
d10 = pairs[0]
img_no_bkg = xbkg.find_and_remove_background(center, radius, center, img, [theta])
roi_size = 8
rois = [peakf.img_roi(ploc, img_no_bkg, roi_size) for ploc in d10]
peak_fits = [peakf.fit_peak((roi_size, roi_size), r) for r in rois]
## Emcee bits
nwalkers = 200
initial_pos = [[r.shape[0]/2+1, r.shape[1]/2+1, f[0], f[1], f[2]] 
               for r,f in zip(rois, peak_fits)]
shakeitup = lambda p0: np.array([np.multiply(p0, np.random.uniform(.8, 1.2, len(p0))) for i in range(nwalkers)]) 
p0s = [shakeitup(pos) for pos in initial_pos]
samplers = [emcee.EnsembleSampler(nwalkers, len(p0), 
                                  imemcee.peak_probability, args = [roi]) 
            for p0,roi in zip(initial_pos, rois)]
burn_ins = [sampler.run_mcmc(p0, 100) for sampler, p0 in zip(samplers, p0s)]
[sampler.reset() for sampler in samplers]
#mcmc_runs = [sampler.run_mcmc(p0, 1000) for sampler, p0 in zip(samplers, p0s)]

## Plot one
sampler = samplers[0]
sampler.run_mcmc(p0s[0], 1000)
param_num = len(initial_pos[0])
# interlude to print confidence intervals
ndims = len(initial_pos[0])
samples = sampler.chain.reshape((-1, ndims))
cy_mcmc, cx_mcmc, h_mcmc, k_mcmc, m_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

fig, axes = plt.subplots(5,1)
for i in range(param_num):
    plt.sca(axes[i])
    samples = sampler.flatchain[:, i]
    weights = np.ones_like(samples)/len(samples)
    plt.hist(samples, 100, weights=weights, color='k', histtype='step')
axes[0].set_title('Peak center (y)')
axes[1].set_title('Peak center (x)')
axes[2].set_title('Peak height (photons)')
axes[3].set_title('Peak spread')
axes[4].set_title('Peak decay')


fig, axes = plt.subplots(3,2,figsize=[12,8])
for fn in fns:
    img = support.image_as_numpy(img_dir+fn)
    #ax = axes[0][0]
    #ax.imshow(img)
    #ax.set_title('Raw image')
    block = blocked_region.find_blocked_region(img, 
                                       plot=False)
    max_points = peakf.peaks_from_image(img, block, plot=False)
    success, thetas, clus_peaks = fiber_lines.optimize_thetas(block[0],
                                  max_points, plot=False, pimg=img)
    pairs = peakf.extract_pairs(block[0], max_points, True, img)
    time.sleep(0.2)
    #d10 = peakf.extractd10(pairs, plot=axes[2][1], pimg=img)
    #plt.tight_layout()
    #plt.draw()
    time.sleep(0.2)
    # Plotting
    plt.clf()
