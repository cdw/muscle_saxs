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
    
    Parameters
    ----------
    img : np.array
        image to find peaks in
    block : tuple
        ((center_col, center_row), center_rad) of the blocked region
    smooth : int, optional
        the (odd) number of pixels to smooth over, default is 1
    mask_percent : int, optional
        the brightness cutoff for peak locations in percent
    peak_range : tuple, optional
        (min, max) number of peaks to find, if passed then jiggle the
        mask_percent until the requisite number of peaks are found and
        return those, if unable to, return found peaks. This supersedes
        the mask_percent if passed, default is false
    plot : boolean or matplotlib.axis, optional
        whether or not to plot the points, or axis to use, default is
        false
    
    Returns
    -------
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
def extract_pairs(block, points, plot=False, pimg=None):
    """Return sets of points representing pairs
    
    Parameters
    ----------
    block : tuple
        ((center_col, center_row), center_rad) of the blocked region
        We're really using this as an analog for the center of the
        diffraction pattern. This is a possible target for improvement.
    points : list 
        the points, clustered by theta or not
    plot : boolean or matplotlib.axis, optional
        whether or not to plot the points, or axis to use, default is
        false
    pimg : np.array
        image points taken from, for plotting
    """
    center, radius = block
    xc, yc = center
    dist_f = lambda (y, x): np.hypot(x - xc, y - yc)
    ang_f = lambda (ly,lx), (ry,rx): np.arctan2(ry-ly, rx-lx)
    def closest_point(pt, pts, dtol = 0.1, atol=0.1):
        """Find closest point within a tolerance.
        
        Parameters
        ----------
        pt : tuple
            the row,col point of interest
        pts : list
            the potential matches
        dtol : float
            distance tolerance as percentage (0 to 1)
        atol : float
            the angular tol in radians
        
        Returns
        -------
        pair :  tuple
            pair of points if match is found
        points : list
            pts minus the match if it was found
        """
        # Test circle intersections
        def intersect_test(pt1, pt2):
            """Does the line segment pass within radius of the block center?
            This is more complicated than one might expect because it is
            checking if the line *segment* passes within the radius of the
            block center, not the infinite line defined by pt1 and pt2. 
            See http://tinyurl.com/6ve7bjz for more info.
            """
            y1, x1 = pt1
            y2, x2 = pt2
            i, j = (x2-x1), (y2-y1)
            s = i**2 + j**2
            u = ((xc - x1)*i + (yc - y1)*j)/s
            u = np.clip(u, 0, 1)
            dist = np.sqrt((x1 + u*i - xc)**2 + (y1 + u*j - yc)**2)
            return radius >= dist
        intersect_bool = np.array([intersect_test(pt, p) for p in pts])
        # Find distances
        # Note that both dists and angles (below) are normalized
        dist = dist_f(pt)
        dists = [(dist_f(p)-dist)/dist for p in pts]
        dist_bool = np.less(dists, dtol) # no greater than dtol% difference 
        # Find angles
        if pt[1] < xc: # vec in quads 1,4
            pt_to_cent = ang_f((yc,xc), pt)
            angles = [(pt_to_cent - ang_f((yc,xc), p))/np.pi for p in pts]
        else: # vec in quads 2,3
            pt_to_cent = ang_f(pt, (yc,xc))
            angles = [(pt_to_cent - ang_f(p, (yc,xc)))/np.pi for p in pts]
        angles = np.abs(angles)
        angles = np.min((angles, np.abs(1-angles)), 0)
        angles_bool = np.less(angles, atol) # no greater than atol radians
        # Find closest match in distance and angle
        allowable = intersect_bool * dist_bool * angles_bool
        ang_and_dist = np.abs(np.multiply(angles, dists))
        try: 
            match = [i for i in np.argsort(ang_and_dist) if allowable[i]][0]
        except IndexError:
            msg = "Point "+str(pt)+" has no match"
            #warnings.warn(msg) # seems unnecessary, restore if missed
            return None, pts
        # Remove matched point and return 
        pt2 = pts.pop(match)
        return (pt, pt2), pts
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
    
    Parameters
    ----------
    pairs : list
        pairs of points clustered by theta angle and center distance
    horizontal : boolean
        True if d10 line is closer to horizontal
    plot : boolean or matplotlib.axes
        plot if true, or axis to use
    pimg : np.array
        img to superimpose plot on if passed
   
    Returns
    -------
    d10: tuple
        points in pair line closest to center
    """
    # Find relevant pair info
    ll = lambda p: (p[p[0][1]>p[1][1]], p[p[0][1]<p[1][1]])
    pairs = map(ll, pairs) # All pairs points go left to right
    a_f = lambda p: np.arctan2(p[1][0]-p[0][0], p[1][1]-p[0][1]) # ang func
    angles = np.array(map(a_f, pairs))
    # Filter by horizontal or not, as desired
    if horizontal == True:
        pairs = [p for p,a in zip(pairs, angles) if a<np.radians(45)]
    else:
        pairs = [p for p,a in zip(pairs, angles) if a>np.radians(45)]
    # Find distances, choose smallest
    d_f = lambda p: np.hypot(p[0][0]-p[1][0], p[0][1]-p[1][1]) # dist func
    dists = np.array(map(d_f, pairs))
    if len(dists) == 0:
        # no d10 found
        return np.NaN
    d10 = pairs[np.argmin(dists)]
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
    
    Parameters
    ----------
    peaks : list
        list of row,col peak locations
    img : np.array
        img which peaks are drawn from
    n_highest : int
        number of peaks to extract (2)
    
    Returns
    -------
    highest : list
        the n highest peaks
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
    
    Parameters
    ----------
    peak : tuple
        row,col location of peak in img
    img : np.array
        peak img
    region : int
        area to fit over, default (6) gives 12x12 roi
    starting : tuple
        optionally preloaded starting conditions
    
    Returns
    -------
    H : float
        the height of the distribution (should be >0)
    K : float
        controls the spread (should be >0)
    M : float
        controls the rate of decay of the tails (should be >0)
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
