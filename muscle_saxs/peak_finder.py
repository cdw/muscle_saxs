#!/usr/bin/env python
# encoding: utf-8
"""
peak_finder.py - locate and identify the peaks in an image

Algorithm description: 
    - smooth image
    - mask out regions below a threshold, close to edge, close to blocker
    - find peaks in non-masked region
    - construct lines with angles that pass through blocked region center
    - classify peaks by distances to lines
    - residual is the sum of distance residuals to line from each point
    - minimize residual by altering lines' angles
    - classify peaks by distance to optimized lines
    - sort peaks into pairs on either side of center

Created by Dave Williams on 2015-02-19
"""

# System imports
import copy
import warnings
import cv2
import numpy as np
from scipy import ndimage, optimize
import matplotlib.pyplot as plt
# Local package imports
import blocked_region
import support


## Build up the peak locations
def peaks_from_image(img, block, smooth=1, mask_percent=80,
                     peak_range=False, plot = False):
    """Return peak locations from an img.
    Takes:
        img: image to find peaks in
        block: ((center_col, center_row), center_rad) of the blocked region
        smooth: the (odd) number of pixels to smooth over
        mask_percent: the brightness cutoff for peak locations (%)
        peak_range: (min, max) number of peaks to find, if passed then jiggle
            the mask_percent until the requisite number of peaks are found
            and return those, if unable to, return found peaks. This supersedes
            the mask_percent if passed (False)
        plot: whether or not to plot the points, or axis to use
    Gives:
        peaks: peak locations matching 
    """
    # Smooth the image and find local maxima
    smoothed = ndimage.filters.gaussian_filter(img, smooth)
    all_maxes = ndimage.filters.maximum_filter(smoothed, size=(3,3))==smoothed
    # Mask image regions that we don't want peaks in
    # Central blocked region first
    blocked_region = np.ones_like(img, dtype=np.int8)
    (b_x, b_y), b_r = block
    extra_around_block = 10 # pix number to add to blocked radius
    cv2.circle(blocked_region, (int(round(b_x)), int(round(b_y))), 
               int(round(b_r+extra_around_block)), 0, -1)
    # And around the edges
    edge_region = np.ones_like(img, dtype=np.int8)
    s_y, s_x = edge_region.shape
    edge_dist = 5 # pixels around edge to mask
    cv2.rectangle(edge_region, (0,0), (s_x, edge_dist), 0, -1)
    cv2.rectangle(edge_region, (0,0), (edge_dist, s_y), 0, -1)
    cv2.rectangle(edge_region, (s_x,0), (s_x - edge_dist, s_y), 0, -1)
    cv2.rectangle(edge_region, (0,s_y), (s_x, s_y-edge_dist), 0, -1)
    def masked_area(percent):
        """Give a masked image from a percent and some pre-blocked areas"""
        above_background = img>np.percentile(img, percent)
        masked = above_background * blocked_region * edge_region
        return masked
    # Find the peaks that aren't masked
    def masked_peaks(percent):
        """Return peaks in the masked region above cutoff percentage"""
        masked = masked_area(percent)
        max_img = all_maxes * masked
        masked_max = max_img.nonzero()
        return masked_max
    maxes = masked_peaks(mask_percent)
    if peak_range is not False:
        peak_min, peak_max = peak_range
        if len(maxes[0]) < peak_min:
            while len(maxes[0]) < peak_min and mask_percent >= 1:
                mask_percent -= 1
                maxes = masked_peaks(mask_percent)
        if len(maxes[0]) > peak_max:
            while len(maxes[0]) > peak_max and mask_percent <= 99:
                mask_percent += 1
                maxes = masked_peaks(mask_percent)
    # Plot if desired
    if plot is not False:
        if plot is True:
            fig, ax = plt.subplots(figsize=[6,3])
        else:
            ax = plot
        ax.imshow(masked_area(mask_percent))
        ax.scatter(maxes[1], maxes[0], c='w', s=30)
        ax.imshow(masked_area(mask_percent)) # resets limits
        ax.set_title("Peaks (white) in non-masked region (red)")
        plt.draw()
        plt.tight_layout()
        plt.show()
    zipped_maxes = zip(maxes[0], maxes[1])
    return zipped_maxes



## Sort peaks into pairs
def extract_pairs(center, points, plot=False, pimg=None):
    """Return sets of points representing pairs
    Takes:
        center: the center of the diffraction pattern
        points: the points, clustered by theta or not
        plot: True to plot, or axis to use
        pimg: image points taken from, for plotting
    """
    dist_f = lambda p: np.hypot(p[1] - center[0], p[0] - center[1])
    ang_f = lambda pl, pr: np.arctan2(pr[1]-pl[1], pr[0]-pl[0])
    def closest_point(pt, pts, tol=0.10):
        """Find closest pt in pts to pt with tolerance tol"""
        # Find distances
        # Note that both dists and angles (below) are normalized
        dist = dist_f(pt)
        dists = [(dist_f(p)-dist)/dist for p in pts]
        # Find angles
        rev_center = (center[1], center[0])
        if pt[1] < center[0]: # vec in quads 1,4
            pt_to_cent = ang_f(pt, rev_center)
            angles = [(pt_to_cent - ang_f(rev_center, p))/np.pi for p in pts]
        else: # vec in quads 2,3
            pt_to_cent = ang_f(rev_center, pt)
            angles = [(pt_to_cent - ang_f(p, rev_center))/np.pi for p in pts]
        # Find closest match in distance and angle
        ang_and_dist = np.abs(np.multiply(angles, dists))
        match = np.argmin(ang_and_dist)
        # Check that match is within tolerance
        if dists[match] <= tol:
            pt2 = pts.pop(match)
            return (pt, pt2), pts
        else:
            msg = "Point "+str(pt)+" has no match within tolerance"
            warnings.warn(msg)
            return None, pts
    # Apply to all points
    points = copy.deepcopy(points)
    points = [points[i] for i in np.argsort(map(dist_f, points))]
    pairs = []
    while len(points)>1:
        point = points.pop(0)
        pair, points = closest_point(point, points)
        if pair is not None:
            pairs.append(pair)
    # Plot if option to plot passed
    if plot is not False:
        colors = list(np.tile(
            ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '.25', '.5', '.75'], 5))
        if plot is True:
            fig, ax = plt.subplots(figsize=[6,3])
        else:
            ax = plot
        for pair in pairs:
            c = colors.pop(0)
            ax.scatter(pair[0][1], pair[0][0], c=c, s=40)
            ax.scatter(pair[1][1], pair[1][0], c=c, s=40)
        if pimg is not None:
            ax.imshow(pimg)
        ax.set_title('Peak pairs matched by color')
        plt.draw()
        plt.tight_layout()
        plt.show()
    return pairs

def extract_d10(pairs, horizontal = True, plot = False, pimg = None):
    """Return the two points representing the most likely d10 pair
    Takes:
        pairs: pairs of points clustered by theta angle and center distance
        horizontal: True if d10 line is closer to horizontal
        plot: plot if true, or axis to use
        pimg: img to superimpose plot on if passed
    Returns:
        d10: points in pair line closest to center
    """
    # Find relevant pair info
    d_f = lambda p: np.hypot(p[0][0]-p[0][1], p[1][0]-p[1][1])
    def a_f(pair):
        pl = pair[pair[0][0]>pair[0][1]]
        pr = pair[pair[0][0]<pair[0][1]]
        return np.arctan2(pr[0]-pl[0], pr[1]-pl[1])
    dists = map(d_f, pairs)
    angles = map(a_f, pairs)
    # Choose the horizontal line, or not as determined by passed options
    hori = np.argmin(np.abs(angles))
    if horizontal is False:
        hori = int(not hori)
    # Sort points by distance
    d_f = lambda p: np.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1])
    dists = [d_f(p) for p in pairs[hori]]
    sortind = np.argsort(dists)
    d10 = pairs[hori][sortind[0]]
    # Plot if option to plot passed
    if plot is not False:
        if plot is True:
            fig, ax = plt.subplots(figsize=[6,3])
        else:
            ax = plot
        ax.scatter(d10[0][1], d10[0][0], c='m', s=40)
        ax.scatter(d10[1][1], d10[1][0], c='m', s=40)
        if pimg is not None:
            ax.imshow(pimg)
        plt.draw()
        plt.tight_layout()
        plt.show()
    return d10

def extract_highest(peaks, img, n_highest=2):
    """Extract the n highest peaks.
    Takes:
        peaks: list of row,col peak locations
        img: img which peaks are drawn from
        n_highest: number of peaks to extract (2)
    Gives:
        highest: the n highest peaks
    """
    heights = [peak_height(p, img) for p in peaks]
    ordered = np.argsort(heights)
    highest = [peaks[ind] for ind in ordered[-n_highest:]]
    return highest


## Find peak properties
def img_roi(peak, img, region=2):
    """Extract a ROI around a peak, extending region pixels in each dir"""
    roi = img[peak[0]-region-1:peak[0]+region, 
              peak[1]-region-1:peak[1]+region]
    return roi

def peak_height(peak, img, region=2):
    """Simple peak height extraction, max of immediate region"""
    roi = img_roi(peak, img, region)
    height = roi.max()
    return height

def fit_peak(peak, img, region=6, starting = None):
    """Fit a peak and surrounding area to a Pearson VII distribution
    Takes:
        peak: row,col location of peak in img
        img: peak img
        region: area to fit over, default (6) gives 12x12 roi
        starting: optionally preload starting conditions
    Gives:
        H: the height of the distribution (should be >0)
        K: controls the spread (should be >0)
        M: controls the rate of decay of the tails (should be >0)
    """
    # Residual for optimization
    def residual(hkm, roi, size, center):
        """Return the residual from fitting a peak to the passed ROI"""
        H, K, M = hkm # Pearson peak coeffs
        res = np.sum(np.abs(roi - support.pearson(size, center, H, K, M)))
        return res
    # Snag roi, set up initial values
    roi = img_roi(peak, img, region)
    size = roi.shape
    center = np.divide(size, 2)
    if starting is not None:
        # Unpack and use
        H_start, K_start, M_start = starting
    else:
        # Educated guesses
        H_start = np.max(roi)
        K_start = 0.2
        M_start = 0.4
    # Optimize peak parts
    opt_res = optimize.minimize(residual, (H_start, K_start, M_start),
                                args = (roi, size, center), 
                                jac = False,
                                bounds = ((0, np.inf), (0, 10), 
                                          (0, 10)))
    success = opt_res['success']
    H, K, M = opt_res['x'] 
    return H, K, M

## Test if run directly
def main():
    import fiber_lines
    SAMPLEFILE = './test_images/sampleimg1.tif'
    img = support.image_as_numpy(SAMPLEFILE)
    block = blocked_region.find_blocked_region(img, plot=True)
    max_points = peaks_from_image(img, block, plot=True)
    success, thetas, clus_peaks = fiber_lines.optimize_thetas(block[0],
                                  max_points, plot=True, pimg=img)
    pairs = extract_pairs(block[0], max_points, True, img)
    d10 = extract_d10(pairs, plot=True, pimg=img)
    
if __name__ == '__main__':
	main()
