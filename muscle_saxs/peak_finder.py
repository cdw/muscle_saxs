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
import cv2
from scipy import ndimage, optimize
import matplotlib.pyplot as plt
# Local package imports
import blocked_region
from support import *


## Build up the peak locations
def peaks_from_image(img, block, smooth=1, mask_percent=80, plot = False):
    """Return peak locations from an img.
    Takes:
        img: image to find peaks in
        block: ((center_col, center_row), center_rad) of the blocked region
        smooth: the (odd) number of pixels to smooth over
        mask_percent: the brightness cutoff for peak locations (%)
        plot: whether or not to plot the points, or axis to use
    Gives:
        peaks: peak locations matching 
    """
    # Smooth the image and find local maxima
    smoothed = ndimage.filters.gaussian_filter(img, smooth)
    maxes = ndimage.filters.maximum_filter(smoothed, size=(3,3))==smoothed
    # Mask image regions that we don't want peaks in
    above_background = img>np.percentile(img, mask_percent)
    blocked_region = np.ones_like(img, dtype=np.int8)
    (b_x, b_y), b_r = block
    extra_around_block = 10 # pix number to add to blocked radius
    cv2.circle(blocked_region, (int(round(b_x)), int(round(b_y))), 
               int(round(b_r+extra_around_block)), 0, -1)
    edge_region = np.ones_like(img, dtype=np.int8)
    s_y, s_x = edge_region.shape
    edge_dist = 5 # pixels around edge to mask
    cv2.rectangle(edge_region, (0,0), (s_x, edge_dist), 0, -1)
    cv2.rectangle(edge_region, (0,0), (edge_dist, s_y), 0, -1)
    cv2.rectangle(edge_region, (s_x,0), (s_x - edge_dist, s_y), 0, -1)
    cv2.rectangle(edge_region, (0,s_y), (s_x, s_y-edge_dist), 0, -1)
    masked = above_background * blocked_region * edge_region
    del(above_background, blocked_region, extra_around_block, 
        edge_region, s_y, s_x, edge_dist)
    # Find the peaks that aren't masked
    max_img = maxes * masked
    maxes = max_img.nonzero()
    # Plot if desired
    if plot is not False:
        if plot is True:
            fig, ax = plt.subplots(figsize=[6,3])
        else:
            ax = plot
        ax.imshow(masked)
        ax.scatter(maxes[1], maxes[0], c='w', s=30)
        ax.imshow(masked) # resets limits
        ax.set_title("Peaks (white) in non-masked region (red)")
        plt.draw()
        plt.tight_layout()
        plt.show()
    zipped_maxes = zip(maxes[0], maxes[1])
    return zipped_maxes


## Find peak lines and cluster
def _dist_to_line_gen(x1, y1, theta):
    """Return a function which finds the distance to a line specified by x1,
    y1 and theta.
    """
    #x1, y1 = b_x, b_y # Center of blocked region
    x2, y2 = x1+1, y1+1*np.tan(theta)
    dist = lambda x0, y0: (np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / 
                           np.sqrt((y2-y1)**2 + (x2-x1)**2))
    return dist

def _cluster_by_line_dist(theta1, theta2, center, points, dists=None):
    """Given the angles of two lines passing through a center, cluster a set
    of points by their distances to those two lines.
    """
    # Generate or use passed distance to theta-defined-lines functions
    if dists is None:
        dist1 = _dist_to_line_gen(center[0], center[1], theta1)
        dist2 = _dist_to_line_gen(center[0], center[1], theta2)
    else:
        dist1 = dists[0]
        dist2 = dists[1]
    # Cluster points by distance to theta-defined-lines
    out = [[], []]
    for point in points:
        if dist1(point[1], point[0]) < dist2(point[1], point[0]):
            out[0].append(point)
        else:
            out[1].append(point)
    return out

def _lines_residual(thetas, center, points):
    """What is the residual of points clustered by distance to lines through 
    the center specified by angles of those two lines."""
    # Unpack thetas
    theta1, theta2 = thetas
    # Generate distance to theta-defined-lines functions
    dist1 = _dist_to_line_gen(center[0], center[1], theta1)
    dist2 = _dist_to_line_gen(center[0], center[1], theta2)
    dists = (dist1, dist2)
    # Cluster the points
    points1, points2 = _cluster_by_line_dist(theta1, theta2, 
                                            center, points, dists)
    # Find the residuals
    pointsdists = []
    [pointsdists.append(dist1(p[1], p[0])) for p in points1]
    [pointsdists.append(dist2(p[1], p[0])) for p in points2]
    residual = np.sum(pointsdists)
    return residual

def optimize_thetas(center, points, starting_thetas=None, 
                    plot=False, pimg=None):
    """Find the line angles which result in the smallest residuals between 
    the peak points and the lines passing through center with angles theta1
    and theta2
    
    Takes:
        center: (x,y) center of blocked region
        points: [(y1,x1),(y2,x2)...] peak points
        starting_thetas: (ang1, ang2) or None (default of 0, pi/4)
        plot: to plot or not, that is this question (or what axis to use?)
        pimg: optional img to use in plotting
    Gives:
        success: True if the optimization converged
        thetas: (ang1, ang2)
        points_out: [[(y11, x11),...], [(y21, x21),...]]
    """
    # Unpack initial thetas if passed
    if starting_thetas is None:
        theta1, theta2 = 0, np.pi/4
    else:
        theta1, theta2 = starting_thetas
    # Configure and run optimizer
    opt_res = optimize.minimize(_lines_residual, 
                                (theta1, theta2), 
                                args=(center, points), 
                                jac=False, 
                                bounds=((-np.pi, np.pi), (-np.pi, np.pi)))
    success = opt_res['success']
    thetas = opt_res['x']
    # Cluster points by final thetas
    points_out = _cluster_by_line_dist(thetas[0], thetas[1], center, points)
    # Plot if option to plot passed
    if plot is not False:
        po = [np.array(p) for p in points_out]
        x, y = center
        t1, t2 = thetas
        t1x1, t1y1 = x+50, y+50*np.tan(t1) 
        t1x2, t1y2 = x-50, y-50*np.tan(t1) 
        t2x1, t2y1 = x+50, y+50*np.tan(t2) 
        t2x2, t2y2 = x-50, y-50*np.tan(t2) 
        if plot is True:
            fig, ax = plt.subplots(figsize=[6,3])
        else:
            ax = plot
        ax.scatter(po[0][:,1], po[0][:,0], c='g', s=40)
        ax.scatter(po[1][:,1], po[1][:,0], c='r', s=40)
        ax.plot([t1x1, t1x2], [t1y1, t1y2], color = 'g', linewidth=3)
        ax.plot([t2x1, t2x2], [t2y1, t2y2], color = 'r', linewidth=3)
        if pimg is not None:
            ax.imshow(pimg)
        ax.set_title('Fiber line identification')
        plt.draw()
        plt.tight_layout()
        plt.show()
    return success, thetas, points_out


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


## Find peak properties
def _roi(peak, img, region=2):
    """Extract a ROI around a peak, extending region pixels in each dir"""
    roi = img[peak[0]-region:peak[0]+region, peak[1]-region:peak[1]+region]
    return roi

def peak_height(peak, img, region=2):
    """Simple peak height extraction, max of immediate region"""
    roi = _roi(peak, img, region)
    height = roi.max()
    return height

def fit_peak(peak, img, region=6, starting = None):
    """Fit a peak and surrounding area to a Pearson VII distribution
    Takes:
        peak: row,col location of peak in img
        img: peak img
        region: area to fit over, default (6) gives 12x12 roi
        starting:
    Gives:

    """
    # Residual for optimization
    def residual(hkm, roi, size, center):
        """Return the residual from fitting a peak to the passed ROI"""
        H, K, M = hkm # Pearson peak coeffs
        res = np.sum(np.abs(roi - pearson(size, center, H, K, M)))
        return res
    # Snag roi, set up initial values
    roi = _roi(peak, img, region)
    size = roi.shape
    center = np.divide(size, 2) - 1
    if starting is not None:
        # Unpack and use
        H_start, K_start, M_start = starting
    else:
        # Educated guesses
        H_start = np.max(roi)
        K_start = 0.8
        M_start = 0.4
    # Optimize peak parts
    opt_res = optimize.minimize(residual, (H_start, K_start, M_start),
                                args = (roi, size, center), 
                                jac = False,
                                bounds = ((0, np.inf), (0, np.inf), 
                                          (0, np.inf)))
    success = opt_res['success']
    H, K, M = opt_res['x'] 
    return H, K, M

## Test if run directly
def main():
    SAMPLEFILE = './test_images/sampleimg1.tif'
    img = image_as_numpy(SAMPLEFILE)
    block = blocked_region.find_blocked_region(img, plot=True)
    max_points = peaks_from_image(img, block, plot=True)
    success, thetas, clus_peaks = optimize_thetas(block[0],
                                  max_points, plot=True, pimg=img)
    pairs = extract_pairs(block[0], max_points, True, img)
    d10 = extract_d10(pairs, plot=True, pimg=img)
    
if __name__ == '__main__':
	main()
