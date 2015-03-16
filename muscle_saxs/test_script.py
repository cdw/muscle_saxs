#!/usr/bin/env python
# encoding: utf-8
"""
test_script.py - make sure that our analysis works

Created by Dave Williams on 2015-03-10
"""

import support
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import peak_finder as peakf
import blocked_region
import fiber_lines
import xray_backgroung as xbkg
import support

# Test of d10 locating
img_dir = '/Users/dave/Desktop/argonne_2013/img_averages/non_precessive_means/'
fns = os.listdir(img_dir)
fns = filter(lambda s:s.endswith('.tif'), fns)
plt.ion()


fn = fns[15]
img = support.image_as_numpy(img_dir+fn)
center, radius = blocked_region.find_blocked_region(img, False)
points = peakf.peaks_from_image(img, (center, radius), plot=False)
success, thetas, clus_peaks = fiber_lines.optimize_thetas(center, points, plot=False, pimg=img)
theta = thetas[np.argmax(clus_peaks)]
pairs = peakf.extract_pairs(center, points, True, img)
d10 = pairs[0]
img_no_bkg = xbkg.find_and_remove_background(center, radius, center, img, [theta])
rois = [peakf.img_roi(ploc, img_no_bkg, 8) for ploc in d10]




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
