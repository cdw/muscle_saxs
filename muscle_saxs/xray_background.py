#!/usr/bin/env python
# encoding: utf-8
"""
xray_background.py - remove background from small angle x-ray scattering imgs

Created by Dave Williams on 2014-10-09
"""

import os, sys
import warnings
import numpy as np
from PIL import Image
import cv2

## Configuration options
SAMPLEFILE = 'sampleimg1.tif'


## Small support functions
def _image_as_numpy(filename):
    """Load an image and convert it to a numpy array of floats."""
    return np.array(Image.open(filename), dtype=np.float) 

def _define_circle_points(center, radius):
    """Create a list of pixel locations to look at on a circle.
    Note that this isn't aliased etc.
    """
    res = np.pi/radius # set resolution to avoid double counting a pixel
    x = center[0] + np.round(radius * np.cos(np.arange(-np.pi, np.pi, res)))
    y = center[1] + np.round(radius * np.sin(np.arange(-np.pi, np.pi, res)))
    return x, y

def _yank_circle_pixels(img, center, radius):
    """Return a list of pixel values in a circle from the passed image."""
    x, y = _define_circle_points(center, radius) 
    ## Filter out out-of-bounds points
    yx = zip(y, x)  # yx b/c row,column
    y_max, x_max = img.shape
    inbounds = lambda yx: 0 <= yx[0] <= y_max and 0 <= yx[1] <= x_max
    yx_inbounds = filter(inbounds, yx)
    if len(yx) != len(yx_inbounds):
        warnings.warn("Circle is clipped by image limits.")
    ## Find pix
    pix = [img[yx] for yx in yx_inbounds]
    return pix


## Contour processing
def _find_blocked_region(img, plot=False):
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
    bar = img.mean() + 2 * img.std()  # bar to leap
    bin_img = (img<bar).astype(np.uint8)*255 # binary that image
    cont, hier = cv2.findContours(bin_img, cv2.cv.CV_RETR_LIST,
                                  cv2.cv.CV_CHAIN_APPROX_NONE)
    hier = hier[0]
    blocked_region = _lowest_contour_in_hierarchy(cont, hier)
    ((x, y), radius) = cv2.minEnclosingCircle(blocked_region)
    if plot is True:
        fig, ax = plt.subplots(figsize=[6,3])
        circle = plt.Circle((x,y), radius, facecolor='m', alpha=0.5)
        ax.imshow(img)
        fig.tight_layout()
        ax.add_patch(circle)
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

def optimize_center_location(img, blocked_circle):
    ((x,y), radius) = blocked_circle
    #TODO incomplete
