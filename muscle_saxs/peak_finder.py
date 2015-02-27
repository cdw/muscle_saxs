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
from xray_background import find_blocked_region
from support import *


## Build up the peak locations
def peaks_from_image(img, smooth=1, mask_percent=80, plot = False):
    """Return peak locations from an img.
    Takes:
        img: image to find peaks in
        smooth: the (odd) number of pixels to smooth over
        mask_percent: the brightness cutoff for peak locations (%)
        plot: whether or not to plot the points
    Gives:
        peaks: peak locations matching 
    """
    # Smooth the image and find local maxima
    smoothed = ndimage.filters.gaussian_filter(img, 1)
    maxes = ndimage.filters.maximum_filter(smoothed, size=(3,3))==smoothed
    # Mask image regions that we don't want peaks in
    above_background = img>np.percentile(img, mask_percent)
    blocked_region = np.ones_like(img, dtype=np.int8)
    (b_x, b_y), b_r = find_blocked_region(img)
    extra_around_block = 5 # pix number to add to blocked radius
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
    if plot is True:
        fig, ax = plt.subplots(figsize=[6,3])
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
        plot: to plot or not, that is this question
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
    if plot is True:
        po = [np.array(p) for p in points_out]
        x, y = center
        t1, t2 = thetas
        t1x1, t1y1 = x+50, y+50*np.tan(t1) 
        t1x2, t1y2 = x-50, y-50*np.tan(t1) 
        t2x1, t2y1 = x+50, y+50*np.tan(t2) 
        t2x2, t2y2 = x-50, y-50*np.tan(t2) 
        fig, ax = plt.subplots(figsize=[6,3])
        ax.scatter(po[0][:,1], po[0][:,0], c='g', s=40)
        ax.scatter(po[1][:,1], po[1][:,0], c='r', s=40)
        ax.plot([t1x1, t1x2], [t1y1, t1y2], color = 'g', linewidth=3)
        ax.plot([t2x1, t2x2], [t2y1, t2y2], color = 'r', linewidth=3)
        if pimg is not None:
            ax.imshow(pimg)
        plt.draw()
        plt.tight_layout()
        plt.show()
    return success, thetas, points_out


## Sort peaks into pairs
def extract_d10(thetas, center, points, horizontal = True):
    """Return the two points representing the most likely d10 pair
    Takes:
        thetas: the two angles of the diffraction lines
        center: the center of the diffraction pattern
        points: the points, clustered by theta
        horizontal: True if d10 line is closer to horizontal
    """
    # Choose the horizontal line, or not as determined by passed options
    hori = int(np.abs(thetas[0]) > np.abs(thetas[1]))
    if horizontal is False:
        hori = int(not hori)
    # Sort points by distance
    d_f = lambda p: np.hypot(p[1] - center[0], p[0] - center[1])
    dists = [d_f(p) for p in points[hori]]
    sortind = np.argsort(dists)
    if dists[sortind[1]] < dists[sortind[0]]*1.10:
        return (points[hori][sortind[0]], points[hori][sortind[1]])
    else:
        warnings.warn("supposed d10 points are more than 10% different")
        return (points[hori][sortind[0]], points[hori][sortind[1]])
    
def extract_pairs(center, point_clus, plot=False, pimg=None):
    """Return sets of points representing pairs
    Takes:
        center: the center of the diffraction pattern
        point_clus: the points, clustered by theta
    """
    def closest_point(pt, pts, tol=0.10):
        """Find closest pt in pts to pt with tolerance tol"""
        # Find distances
        d_f = lambda p: np.hypot(p[1] - center[0], p[0] - center[1])
        dists = [d_f(p) for p in pts]
        dist = d_f(pt)
        # Find closest match in distance
        match = np.argmin(np.abs((dists-dist)/dist))
        # Check that match is within tolerance
        if dist*(1-tol) <= dists[match] <= dist*(1+tol):
            pt2 = pts.pop(match)
            return (pt, pt2), pts
        else:
            warnings.warn("Point "+str(pt)+" has no match within tolerance")
            return None, pts
    # Apply across clusters
    point_clus = copy.deepcopy(point_clus)
    pair_clus = []
    for points in point_clus:
        pairs = []
        while len(points)>1:
            point = points.pop(0)
            pair, points = closest_point(point, points)
            if pair is not None:
                pairs.append(pair)
        pair_clus.append(pairs)
    # Plot if option to plot passed
    if plot is True:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '.25', '.5', '.75']
        fig, ax = plt.subplots(figsize=[6,3])
        for line in pair_clus:
            for pair in line:
                c = colors.pop(0)
                ax.scatter(pair[0][1], pair[0][0], c=c, s=40)
                ax.scatter(pair[1][1], pair[1][0], c=c, s=40)
        if pimg is not None:
            ax.imshow(pimg)
        plt.draw()
        plt.tight_layout()
        plt.show()
    return pair_clus


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


## Test if run directly
def main():
    SAMPLEFILE = 'sampleimg1.tif'
    img = image_as_numpy(SAMPLEFILE)
    block_center, block_radius = find_blocked_region(img, plot=True)
    max_points = peaks_from_image(img, plot=True)
    success, thetas, clus_peaks = optimize_thetas(block_center,
                                  max_points, plot=True, pimg=img)
    pairs = extract_pairs(block_center, clus_peaks, True, img)

if __name__ == '__main__':
	main()
