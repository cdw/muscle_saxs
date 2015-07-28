# encoding: utf-8
"""
fiber_lines.py - find the lines of peaks present in the image

Created by CDW on 2015.10.13
"""

#System imports
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


## Support functions
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


## Find peak lines and cluster
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


