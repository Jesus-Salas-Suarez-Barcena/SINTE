# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:21:49 2023

@author: jesus
"""

import numpy as np
from operations import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

# source: http://geomalgorithms.com/a06-_intersect-2.html

def ray_intersect_triangle(p0, p1, triangle):
    """
    Unused

    Parameters
    ----------
    p0 : TYPE
        DESCRIPTION.
    p1 : TYPE
        DESCRIPTION.
    triangle : TYPE
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    # Tests if a ray starting at point p0, in the direction
    # p1 - p0, will intersect with the triangle.
    #
    # arguments:
    # p0, p1: numpy.ndarray, both with shape (3,) for x, y, z.
    # triangle: numpy.ndarray, shaped (3,3), with each row
    #     representing a vertex and three columns for x, y, z.
    #
    # returns:
    #    0.0 if ray does not intersect triangle,
    #    1.0 if it will intersect the triangle,
    #    2.0 if starting point lies in the triangle.

    v0, v1, v2 = triangle
    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)

    b = np.inner(normal, p1 - p0)
    a = np.inner(normal, v0 - p0)

    # Here is the main difference with the code in the link.
    # Instead of returning if the ray is in the plane of the
    # triangle, we set rI, the parameter at which the ray
    # intersects the plane of the triangle, to zero so that
    # we can later check if the starting point of the ray
    # lies on the triangle. This is important for checking
    # if a point is inside a polygon or not.

    if (b == 0.0):
        # ray is parallel to the plane
        if a != 0.0:
            # ray is outside but parallel to the plane
            return 0
        else:
            # ray is parallel and lies in the plane
            rI = 0.0
    else:
        rI = a / b

    if rI < 0.0:
        return 0

    w = p0 + rI * (p1 - p0) - v0

    denom = np.inner(u, v) * np.inner(u, v) - \
        np.inner(u, u) * np.inner(v, v)

    si = (np.inner(u, v) * np.inner(w, v) - \
        np.inner(v, v) * np.inner(w, u)) / denom

    if (si < 0.0) | (si > 1.0):
        return 0

    ti = (np.inner(u, v) * np.inner(w, u) - \
        np.inner(u, u) * np.inner(w, v)) / denom

    if (ti < 0.0) | (si + ti > 1.0):
        return 0

    if (rI == 0.0):
        # point 0 lies ON the triangle. If checking for
        # point inside polygon, return 2 so that the loop
        # over triangles can stop, because it is on the
        # polygon, thus inside.
        return 2

    return 1


def ray_intersect_triangle2(p0, p1, triangle, debugg=False):
    """
    Checks if a ray intersects with a triangle.

    Parameters
    ----------
    p0 : 3D vector: np.array
        Initial point of ray.
    p1 : 3D vector: np.array
        Initial point of ray.
    triangle : 9D np.array
        Contains the vertices coordinates of the triangle.
    debugg : Boolean, optional
        Plot the points involved, never do in a loop. The default is False.

    Returns
    -------
    int
        0 if false, 1 if true and 2 if any point is contained.

    """

    v01 = p1 - p0
    k = v01/np.linalg.norm(v01)
    t0, t1, t2 = triangle


    v1 = t1 - t0
    v2 = t2 - t0
    v3 = t2 - t1

    N = np.cross(v1, v2)
    N = N/np.linalg.norm(N)
    if np.dot(N, k) == 0:
        return 0

    pt = inter_pr(N, t0, k, p0)

    vt0 = p0 - pt
    vt1 = p1 - pt

    d01 = np.linalg.norm(v01)
    dt0 = np.linalg.norm(vt0)
    dt1 = np.linalg.norm(vt1)

    if debugg == True:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot([p0[0]], [p0[1]], [p0[2]],
                markerfacecolor='g', markeredgecolor='g',
                marker='o', markersize=5, alpha=0.6)
        ax.text(p0[0], p0[1], p0[2], 'P0')
        ax.plot([p1[0]], [p1[1]], [p1[2]],
                markerfacecolor='g', markeredgecolor='g',
                marker='o', markersize=5, alpha=0.6)
        ax.text(p1[0], p1[1], p1[2], 'P1')
        ax.plot([t0[0]], [t0[1]], [t0[2]],
                markerfacecolor='b', markeredgecolor='b',
                marker='o', markersize=5, alpha=0.6)
        ax.plot([t1[0]], [t1[1]], [t1[2]],
                markerfacecolor='b', markeredgecolor='b',
                marker='o', markersize=5, alpha=0.6)
        ax.plot([t2[0]], [t2[1]], [t2[2]],
                markerfacecolor='b', markeredgecolor='b',
                marker='o', markersize=5, alpha=0.6)
        ax.plot([pt[0]], [pt[1]], [pt[2]],
                markerfacecolor='k', markeredgecolor='k',
                marker='o', markersize=5, alpha=0.6)
        ax.plot3D([t0[0], t1[0]], [t0[1], t1[1]], [t0[2], t1[2]], color='k')
        ax.plot3D([t0[0], t2[0]], [t0[1], t2[1]], [t0[2], t2[2]], color='k')
        ax.plot3D([t2[0], t1[0]], [t2[1], t1[1]], [t2[2], t1[2]], color='k')
        ax.quiver(pt[0], pt[1], pt[2],
                  N[0]/100, N[1]/100, N[2]/100, color='k')
        ax.quiver(p0[0], p0[1], p0[2],
                  k[0]/100, k[1]/100, k[2]/100, color='k')
        # ax.scatter(triangle_int[0, 0, :], triangle_int[1, 0, :],
        #            triangle_int[2, 0, :], marker='o')
    if dt1 < 1e-4:
        return 2
    elif d01 < (dt0 + dt1):
        return 0
    elif d01 == (dt0 + dt1):
        vtc1 = pt - t0
        vtc2 = pt - t1
        vtc3 = pt - t2

        A_tri = np.linalg.norm(np.cross(v1, v2))/2
        A_tri1 = np.linalg.norm(np.cross(v1, vtc1))/2
        A_tri2 = np.linalg.norm(np.cross(v2, vtc3))/2
        A_tri3 = np.linalg.norm(np.cross(v3, vtc2))/2
        # print(A_tri - (A_tri1 + A_tri2 + A_tri3))
        dA = abs(A_tri - (A_tri1 + A_tri2 + A_tri3))
        if dA > 1e-7:
            return 0
        elif dA <= 1e-7:
            return 1

def ray_triangle_intersection(ray_start, ray_vec, triangle):
    """Moeller–Trumbore intersection algorithm.

    Parameters
    ----------
    ray_start : np.ndarray
        Length three numpy array representing start of point.

    ray_vec : np.ndarray
        Direction of the ray.

    triangle : np.ndarray
        ``3 x 3`` numpy array containing the three vertices of a
        triangle.

    Returns
    -------
    bool
        ``True`` when there is an intersection.

    tuple
        Length three tuple containing the distance ``t``, and the
        intersection in unit triangle ``u``, ``v`` coordinates.  When
        there is no intersection, these values will be:
        ``[np.nan, np.nan, np.nan]``

    """
    # define a null intersection
    null_inter = np.array([np.nan, np.nan, np.nan])

    # break down triangle into the individual points
    v1, v2, v3 = triangle
    eps = 0.000000001

    # compute edges
    edge1 = v2 - v1
    edge2 = v3 - v1
    pvec = np.cross(ray_vec, edge2)
    det = edge1.dot(pvec)

    if abs(det) < eps:  # no intersection
        return 0, null_inter
    inv_det = 1.0 / det
    tvec = ray_start - v1
    u = tvec.dot(pvec) * inv_det

    if u < 0.0 or u > 1.0:  # if not intersection
        return 0, null_inter

    qvec = np.cross(tvec, edge1)
    v = ray_vec.dot(qvec) * inv_det
    if v < 0.0 or u + v > 1.0:  # if not intersection
        return 0, null_inter

    t = edge2.dot(qvec) * inv_det
    if t < eps:
        return 0, null_inter

    print('Ding!')
    return 1, np.array([t, u, v])



