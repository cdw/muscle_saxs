#!/usr/bin/env python
# encoding: utf-8
"""
xray_background.py - remove background from small angle x-ray scattering imgs

Created by Dave Williams on 2014-10-09
"""

# System imports
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Local package imports
from support import *


## Contour processing
def find_blocked_region(img, plot=False):
    """Locate the centrally blocked region of the image.
    Given an image, find the center region where the blocker is, defined as
    the inner-most (and roundest) area with pixel values less than two
    standard deviations from the mean pixel value.
    Takes: 
        img - numpy array of grey values
        plot - whether to plot the process (false by default)
    Gives:
        center - (x,y) value locations of the center of the blocked region
        radius - the radius of the blocked region
    """
    # Segment the image
    bar = img.mean() + 2 * img.std()  # bar to leap
    bin_img = (img<bar).astype(np.uint8)*255 # binary that image
    # Process the image into contours
    cont, hier = cv2.findContours(bin_img, cv2.cv.CV_RETR_LIST,
                                  cv2.cv.CV_CHAIN_APPROX_NONE)
    hier = hier[0]
    # Blocked region is lowest contour, create a circle for it
    blocked_region = _lowest_contour_in_hierarchy(cont, hier)
    ((x, y), radius) = cv2.minEnclosingCircle(blocked_region)
    # Plot if option to do so is passed
    if plot is True:
        fig, ax = plt.subplots(figsize=[6,3])
        circle = plt.Circle((x,y), radius, facecolor='m', alpha=0.5)
        ax.imshow(img)
        ax.add_patch(circle)
        plt.draw()
        fig.tight_layout()
        plt.show()
    return ((x, y), radius)

def _lowest_contour_in_hierarchy(contours, hierarchy):
    """Return the lowest, most junior, node(s) in a contour set.
    Contour sets consist of a series of entries such as [a, b, c, d] where:
        a is index of the next contour on the same level
        b is index of the prior contour on the same level
        c is the first child of this contour
        d is the parent of this contour
    and a value of -1 means none-such exists.
    """
    level_list = _node_levels([[]], hierarchy, 0, 0) # walk tree to find levels
    lowest = level_list[-1]
    if len(lowest) > 1:
        warnings.warn("More than one lowest contour, taking roundest one")
        return _roundest_contour([contours[i] for i in lowest])
    else:
        return contours[lowest[0]]

def _node_levels(tree, hier, level, node):
    """Creates list of which nodes are on which level, by recursively walking 
    the hierarchy tree.
    """
    h = hier[node] # the current node in the hierarchy
    if len(tree) < (level+1):
        tree.append([]) 
    tree[level].append(node) # add our current node to the structure
    if h[0] >= 0: # then keep walking this level
        _node_levels(tree, hier, level, h[0])
    if h[2] >= 0: # then walk the child level
        _node_levels(tree, hier, level+1, h[2])
    return tree

def _roundest_contour(contours):
    """Given a list of contours, return the roundest."""
    ratios = [np.sqrt(cv2.contourArea(c))/cv2.arcLength(c, 1) for c in contours]
    highest_ratio_contour = contours[ratios.index(max(ratios))] 
    return highest_ratio_contour


## Optimize the circle center based on blocked region

def background_collapse(center, img, thetas, plot=False):
    """Collapse the image background, ignoring the peak regions.
    Good ideas to be had here: http://goo.gl/2xEApw
    Takes:
        center: x,y center of blocked image
        img: from which background is extracted
        thetas: angles of lines we wish to miss
        plot: if we should plot the exclusion regions and profile
    Gives:
        back: profile of background
    """
    # Construct a radial distance img
    row, col = np.indices(img.shape)
    r = np.sqrt((row-center[0])**2 + (col-center[1])**2)
    # Coerce into ints for bincount
    r = r.astype(np.int)
    img = img.astype(np.int)
    # Mask the lines
    mask = Image.new('1', img.shape[::-1], color=1)
    bb = [center[0] - mask.size[0], center[1] - mask.size[1], 
          center[0] + mask.size[0], center[1] + mask.size[1]]
    #m_thetas = np.concatenate(([t+np.pi for t in thetas], thetas))
    m_thetas = thetas
    m_thetas = np.round(np.degrees(m_thetas)).astype(np.int)
    theta_pm = 20
    m_angles = [(t-theta_pm, t+theta_pm) for t in m_thetas]
    draw = ImageDraw.Draw(mask)
    for angle in m_angles:
        draw.pieslice(bb, angle[0], angle[1], fill=0)
    mask = np.array(mask.getdata(), np.uint).reshape(
        mask.size[1], mask.size[0])
    plt.imshow(img*mask, interpolation='nearest')
    img = img*mask
    # Do the counting
    flat_count = np.bincount(r.ravel(), img.ravel())
    rad_occurances = np.bincount(r.ravel()) 
    radial_profile = flat_count/rad_occurances
    # Kill the blocked region
    highest_ind = radial_profile.argmax()
    background = radial_profile[highest_ind:]
    return radial_profile


def optimize_center_location(img, blocked_circle):
    ((x,y), radius) = blocked_circle
    #TODO incomplete
    

## Test if run directly
def main():
    import peak_finder
    SAMPLEFILE = 'sampleimg1.tif'
    data = image_as_numpy(SAMPLEFILE) # load
    bcenter, brad = find_blocked_region(data, True) # find blocker
    unorg_peaks = peak_finder.peaks_from_image(data, plot=True) # find peaks
    success, thetas, peaks = peak_finder.optimize_thetas(bcenter, unorg_peaks)

if __name__ == '__main__':
	main()

