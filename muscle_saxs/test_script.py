#!/usr/bin/env python
# encoding: utf-8
"""
test_script.py - make sure that our analysis works

Created by Dave Williams on 2015-03-10
"""

import support
import time
import os
import matplotlib.pyplot as plt
import peak_finder as peakf
import blocked_region
import support

# Test of d10 locating
img_dir = '/Users/dave/Desktop/argonne_2013/img_averages/non_precessive_means/'
fns = os.listdir(img_dir)
fns = filter(lambda s:s.endswith('.tif'), fns)
plt.ion()


fn = fns[10]
img = support.image_as_numpy(img_dir+fn)
center, radius = blocked_region.find_blocked_region(img, False)
points = peakf.peaks_from_image(img, (center, radius), plot=False)
success, thetas, clus_peaks = peakf.optimize_thetas(center,
                              points, plot=False, pimg=img)
pairs = peakf.extract_pairs(center, points, True, img)





fig, axes = plt.subplots(3,2,figsize=[12,8])
for fn in fns:
    img = support.image_as_numpy(img_dir+fn)
    #ax = axes[0][0]
    #ax.imshow(img)
    #ax.set_title('Raw image')
    block = blocked_region.find_blocked_region(img, 
                                       plot=False)
    max_points = peakf.peaks_from_image(img, block, plot=False)
    success, thetas, clus_peaks = peakf.optimize_thetas(block[0],
                                  max_points, plot=False, pimg=img)
    pairs = peakf.extract_pairs(block[0], max_points, True, img)
    time.sleep(0.2)
    #d10 = peakf.extractd10(pairs, plot=axes[2][1], pimg=img)
    #plt.tight_layout()
    #plt.draw()
    time.sleep(0.2)
    # Plotting
    plt.clf()
