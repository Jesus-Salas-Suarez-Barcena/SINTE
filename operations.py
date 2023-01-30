# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:36:27 2023

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import pandas as pd
from numpy import random as rd
import constants
import scipy as sc
import scipy.signal as sig
import csv as csv
import time as tm
import xlwt as xl
import collections.abc


def rot_matrix(phi, psi, theta):
    """
    Constructs a 3x3 rotation matrix using the Euler angles.

    Parameters
    ----------
    phi : Scalar
        First angle.
    psi : Scalar.
        Second Angle.
    theta : Scalar.
        Third Angle.

    Returns
    -------
    rot : 3x3 matrix np.array
        Rotation matrix.

    """

    r11 = np.cos(psi)*np.cos(theta)
    r12 = np.sin(phi)*np.sin(psi)*np.cos(theta)-np.cos(phi)*np.sin(theta)
    r13 = np.cos(phi)*np.sin(psi)*np.cos(theta)+np.sin(phi)*np.sin(theta)
    r21 = np.cos(psi)*np.sin(theta)
    r22 = np.sin(phi)*np.sin(psi)*np.sin(theta)+np.cos(phi)*np.cos(theta)
    r23 = np.cos(phi)*np.sin(psi)*np.sin(theta)-np.sin(phi)*np.cos(theta)
    r31 = -np.sin(psi)
    r32 = np.sin(phi)*np.cos(psi)
    r33 = np.cos(phi)*np.cos(psi)


    rot = np.array([[r11, r12, r13],
                    [r21, r22, r23],
                    [r31, r32, r33]])

    return rot

def FRT(r_array, r_c, rot):
    """
    Frame of reference transform. changes from the objects FR
    to the general FR.

    Parameters
    ----------
    r_array : 3D vector array np.array
        Vector array to transform.
    r_c : 3D vector np.array
        Center of the FR in the general FR.
    rot : 3x3 matrix np.array.
        Rotation matrix from the FR to the general FR.

    Returns
    -------
    r_transformed : 3D vector array np.array
        Vector array transformed.

    """
    D, N = r_array.shape
    ones = np.ones(N)
    r_transformed = np.dot(np.array([r_c]).T, np.array([ones]))\
        + np.dot(rot, r_array)

    return r_transformed

def closeto(rr, r, ds):
    """
    Checks the points inside a sphere of radius ds arraound r.

    Parameters
    ----------
    rr : 3D vector array np.array
        Array of points to check.
    r : 3D vector np.array
        Test point.
    ds : Scalar
        Radius of the sphere.

    Returns
    -------
    data : Dictionary
        Data on the points close to the point r.

    """



    D, N = rr.shape
    ones = np.ones(N)
    rr_r = rr - np.dot(np.array([r]).T, np.array([ones]))
    dd = np.linalg.norm(rr_r, axis=0)

    arr_c = dd<=ds

    xx = rr[0, :]
    yy = rr[1, :]
    zz = rr[2, :]

    xx_close = xx[arr_c]
    yy_close = yy[arr_c]
    zz_close = zz[arr_c]

    rr_close = np.array([xx_close, yy_close, zz_close])

    # if isinstance(tag, (collections.abc.Sequence, np.ndarray)):
    #     tag_c = tag[arr_c]
    # else:
    #     tag_c = tag

    data = {
        'dd': dd,
        'rr_r': rr_r,
        'array': arr_c,
        'rr': rr_close,
        }

    return data

def infront(rr, r, k):
    """
    Checks which points are in front of r.

    Parameters
    ----------
    rr : 3D vector array np.array
        Array of points to check.
    r : 3D vector np.array
        Test point.
    k : 3D vector np.array
        Direction to check.

    Returns
    -------
    data : Dictionary
        Data on the points in front of point r.

    """

    xx = rr[0, :]
    yy = rr[1, :]
    zz = rr[2, :]
    N = len(xx)
    ones = np.ones(N)
    rr_r = rr - np.dot(np.array([r]).T, np.array([ones]))
    rr_k = rr_r[0, :]*k[0] + rr_r[1, :]*k[1] + rr_r[2, :]*k[2]

    arr_f = rr_k > 0
    xx_f = xx[arr_f]
    yy_f = yy[arr_f]
    zz_f = zz[arr_f]
    rr_f = np.array([xx_f, yy_f, zz_f])

    # if isinstance(tag, (collections.abc.Sequence, np.ndarray)):
    #     tag_f = tag[arr_f]
    # else:
    #     tag_f = tag

    data = {
        'rr': rr_f,
        'array': arr_f,
        }


    return data

def behind(rr, r, k):
    """
    Checks which points are behind r.

    Parameters
    ----------
    rr : 3D vector array np.array
        Array of points to check.
    r : 3D vector np.array
        Test point.
    k : 3D vector np.array
        Direction to check.

    Returns
    -------
    data : Dictionary
        Data on the points in behind of point r.

    """


    xx = rr[0, :]
    yy = rr[1, :]
    zz = rr[2, :]
    N = len(xx)
    ones = np.ones(N)
    rr_r = rr - np.dot(np.array([r]).T, np.array([ones]))
    rr_k = rr_r[0, :]*k[0] + rr_r[1, :]*k[1] + rr_r[2, :]*k[2]

    arr_f = rr_k < 0
    xx_f = xx[arr_f]
    yy_f = yy[arr_f]
    zz_f = zz[arr_f]
    rr_f = np.array([xx_f, yy_f, zz_f])

    # if isinstance(tag, (collections.abc.Sequence, np.ndarray)):
    #     if len(tag) > 0:
    #         tag_f = tag[arr_f]
    #     else:
    #         rag_f = tag
    # else:
    #     tag_f = tag

    data = {
        'rr': rr_f,
        'array': arr_f,
        }


    return data


def goesthrough(rr, r, k, ds, dx):
    """
    Checks the points inside a cylinder from r to r + k*ds.

    Parameters
    ----------
    rr : 3D vector array np.array
        Array of points to check.
    r : 3D vector np.array
        Test point.
    k : 3D vector np.array
        Direction to check.
    ds : Scalar
        length of the cylinder.
    dx : Scalar
        Radius of the cylinder.

    Returns
    -------
    data : Dictionary
        Data on the points inside the cylinder.

    """

    r1 = r + k*ds
    data_behind = behind(rr, r1, k)
    data_front = infront(rr, r, k)
    data_close = closeto(rr, r1, ds)
    xx = rr[0, :]
    yy = rr[1, :]
    zz = rr[2, :]

    N = len(xx)
    ones = np.ones(N)
    rr_r = -rr + np.dot(np.array([r]).T, np.array([ones]))
    dr = np.linalg.norm(np.cross(k, rr_r.T), axis=1)/np.linalg.norm(k)




    arr_g = dr <= dx
    arr_f = data_front['array']
    arr_b = data_behind['array']
    arr_c = data_close['array']

    arr_t = arr_g*arr_f*arr_c*arr_b

    xx_g = xx[arr_t]
    yy_g = yy[arr_t]
    zz_g = zz[arr_t]
    rr_goes = np.array([xx_g, yy_g, zz_g])

    # if isinstance(tag, (collections.abc.Sequence, np.ndarray)):
    #     tag_g = data_close['tag'][arr_g]
    # else:
    #     tag_g = tag

    data = {
        'rr': rr_goes,
        'array': arr_t,
        }

    return data


def plane_average(rr):
    """
    Unused. Makes an average of the plane with the points array.
    Parameters
    ----------
    rr : 3D vector array np.array
        Points to average.

    Returns
    -------
    data : Dictionary
        Data on the plane.

    """
    D, N = rr.shape
    if N < 3:
        # data = {
        #     'points': 2,
        #     }
        data ={}
    elif N < 4:
        v1 = rr[:, 0]-rr[:, 1]
        v2 = rr[:, 0]-rr[:, 2]
        N_array = np.cross(v1, v2)
        r_c = (rr[:, 0] + rr[:, 1] + rr[:, 2])/3
        N_c = N_array
        C = -np.dot(r_c, N_c)
        data = {
            'r': r_c,
            'N': N_c,
            'C': C,
            }
    else:
        N_array = np.zeros((3, N-2))
        for i in range(N-2):
            v1 = rr[:, i]-rr[:, i+1]
            v2 = rr[:, i]-rr[:, i+2]
            N_array[:, i] = np.cross(v1, v2)
        r_c = np.sum(rr, axis=1)/N
        N_c = np.sum(rr, axis=1)/(N-2)
        C = -np.dot(r_c, N_c)

        data = {
            'r': r_c,
            'N': N_c,
            'C': C,
            }

    return data

# def eq_solver(M, C):
#     n, m = M.shape
#     M_d = np.linalg.det(M)
#     M_c = M
#     M_c[:, n-1] = C
#     M_cd = np.linalg.det(M_c)
#     if M_d == 0 and M_cd != 0:
#         raise Exception('Incompatible System of equations')
#     elif  M_d == 0 and M_cd == 0:
#         raise Exception('Compatible but indetermined')

#     else:
#         r = np.dot(np.linalg.inv(M), C)

#     return r

def inter_pr(N_p, r_p, k_c, r_c):
    """
    Intersects between a line and a plane

    Parameters
    ----------
    N_p : 3D vector np.array
        Normal vector of the plane.
    r_p : 3D vector np.array
        Point of reference for the plane.
    k_c : 3D vector np.array
        Direction of the line.
    r_c : 3D vector np.array
        Point of reference for the line.

    Returns
    -------
    r : 3D vector np.array
        Point of intersection.

    """

    A = np.dot(N_p, r_p)
    B = k_c[1]*r_c[0] - k_c[0]*r_c[1]
    C = k_c[2]*r_c[1] - k_c[1]*r_c[2]
    V = np.array([A, B, C])
    M = np.array([N_p, [k_c[1], -k_c[0], 0], [0, k_c[2], -k_c[1]]])

    # print(M)
    # print(V)

    r = np.linalg.solve(M, V)

    return r

def hiig(k_c, r_c, N_p, r_p):
    """
    Unused

    Parameters
    ----------
    k_c : 3D vector np.array
        DESCRIPTION.
    r_c : 3D vector np.array
        DESCRIPTION.
    N_p : 3D vector np.array
        DESCRIPTION.
    r_p : 3D vector np.array
        DESCRIPTION.

    Returns
    -------
    trigger : TYPE
        DESCRIPTION.

    """
    v = r_c - r_p
    Nk = np.dot(k_c, N_p)
    trigger = Nk < 0
    return trigger


def reflect(p0, p1, triangle, debugg=False):
    """
    Reflects the trajectory from point 0 to 1 over a triangle.

    Parameters
    ----------
    p0 : 3D vector np.array
        Initial point of the trajectory.
    p1 : 3D vector np.array
        Final point of the trajectory.
    triangle : 9D vector np.array
        Points of the triangle.
    debugg : Boolean, optional
        If True creates figures with the reflection points
        and the triangle. The default is False.

    Returns
    -------
    pr : 3D vector np.array
        Point of reflection.
    kr : 3D vector np.array
        Reflected direction.

    """

    vp = p1 - p0
    k = vp/np.linalg.norm(vp)

    t0, t1, t2 = triangle

    v1 = t1 - t0
    v2 = t2 - t0

    N = np.cross(v1, v2)
    N = N/np.linalg.norm(N)

    if np.dot(k, N)>0:
        N = -N

    pt = inter_pr(N, t0, k, p0)

    vpp = p1 - pt
    vpr = vpp - 2*N*np.dot(N, vpp)
    pr = pt + vpr
    kr = vpr/np.linalg.norm(vpr)

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
        ax.plot([pr[0]], [pr[1]], [pr[2]],
                markerfacecolor='y', markeredgecolor='y',
                marker='o', markersize=5, alpha=0.6)
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
        ax.quiver(pr[0], pr[1], pr[2],
                  kr[0]/100, kr[1]/100, kr[2]/100, color='k')
        ax.quiver(p0[0], p0[1], p0[2],
                  k[0]/100, k[1]/100, k[2]/100, color='k')
        # ax.scatter(triangle_int[0, 0, :], triangle_int[1, 0, :],
        #            triangle_int[2, 0, :], marker='o')

    return pr, kr


def triinsidecyl(p0, p1, triangles, dx):
    """
    Checks which triangles are inside the cylinder of radius dx.

    Parameters
    ----------
    p0 : 3D vector np.array
        Initial point of the trajectory.
    p1 : 3D vector np.array
        Final point of the trajectory.
    triangle : 9D vector np.array
        Points of the triangle.
    dx : Scalar
        Radius of the cylinder.

    Returns
    -------
    index : np.array
        Indices of the triangles inside the cylinder.
    barray : Boolean arra
        Indices of the triangles inside the cylinder.
    data_cond : Dictionary
        Information on the condition.

    """

    k = (p1 -p0)/np.linalg.norm(p1 - p0)
    D, N_tri, T = triangles.shape
    ones = np.ones((1, N_tri))
    p0_array = np.zeros((D, N_tri, T))
    p0_array[0, :, :] = p0[0]
    p0_array[1, :, :] = p0[1]
    p0_array[2, :, :] = p0[2]
    p1_array = np.zeros((D, N_tri, T))
    p1_array[0, :, :] = p1[0]
    p1_array[1, :, :] = p1[1]
    p1_array[2, :, :] = p1[2]

    v_t0 = triangles - p0_array
    v_t1 = triangles - p1_array

    dot_t0k = (v_t0[0, :, :]*k[0] + v_t0[1, :, :]*k[1]\
               + v_t0[2, :, :]*k[2])
    dot_t1k = (v_t1[0, :, :]*k[0] + v_t1[1, :, :]*k[1]\
               + v_t1[2, :, :]*k[2])

    cond_ft = dot_t0k > 0
    cond_f = cond_ft[:, 0]*cond_ft[:, 1]*cond_ft[:, 2]
    cond_bt = dot_t1k < 0
    cond_b = cond_bt[:, 0]*cond_bt[:, 1]*cond_bt[:, 2]

    dtri = np.sqrt(np.linalg.norm(v_t0, axis=0)**2 - dot_t0k**2)

    cond_ct = dtri <= dx
    cond_c = cond_ct[:, 0]*cond_ct[:, 1]*cond_ct[:, 2]
    all_index = np.array(range(N_tri))
    barray = cond_c*cond_f*cond_b
    index = all_index[cond_c*cond_f*cond_b]
    data_cond = {
        'f': cond_f,
        'b': cond_b,
        'c': cond_c,
            }
    return index, barray, data_cond

def betweensphere(p0, p1, rr_tri, ds):
    """
    Checks the points inside of a sphere centered between two points.

    Parameters
    ----------
    p0 : 3D vector np.array
        Initial point of the trajectory.
    p1 : 3D vector np.array
        Final point of the trajectory.
    rr_tri : 9D vector np.array
        Points of the triangle.
    ds : Scalar
        Diameter of the sphere.
    Returns
    -------
    index : np.array
        Indices of the triangles inside the sphere.
    cond_tri : Boolean array
        Indices of the triangles inside the sphere.

    """

    k = (p1 -p0)/np.linalg.norm(p1 - p0)
    D, N_tri, T = rr_tri.shape
    ones = np.ones((1, N_tri))
    pm = (p1 + p0)/2
    # p0_array = np.zeros((D, N_tri, T))
    # p0_array[0, :, :] = p0[0]
    # p0_array[1, :, :] = p0[1]
    # p0_array[2, :, :] = p0[2]
    # p1_array = np.zeros((D, N_tri, T))
    # p1_array[0, :, :] = p1[0]
    # p1_array[1, :, :] = p1[1]
    # p1_array[2, :, :] = p1[2]
    pm_array = np.zeros((D, N_tri, T))
    pm_array[0, :, :] = pm[0]
    pm_array[1, :, :] = pm[1]
    pm_array[2, :, :] = pm[2]

    v_rt_tri = rr_tri - pm_array

    d_rt_tri = np.linalg.norm(v_rt_tri, axis=0)

    cond = d_rt_tri <= ds/2
    cond_tri = cond[:, 0]*cond[:, 1]*cond[:, 2]
    all_index = np.array(range(N_tri))
    index = all_index[cond_tri]

    return index, cond_tri


def dbctorad(P, B):
    P_B = P + 10*np.log10(B)
    rms = np.sqrt(2*(10**(P_B/10)))
    radHz = rms/(B/2)

    return radHz

def alfvenfreq(n_e, B, q, R_0):
    """
    Alfven frequency function

    Parameters
    ----------
    n_e : Scalar
        Electron density (m^-3).
    B : Scalar
        Magnetic field (T).
    q : Scalar
        Safety factor.
    R_0 : Scalar
        Major radius (m).

    Returns
    -------
    f_a : Scalar
        Alfven frequency (Hz).

    """

    v_a = B/np.sqrt(constants.mu_0*(constants.m_i + constants.m_e)*n_e)
    f_a = v_a/(4*np.pi*q*R_0)


    return f_a

def cutdens(f):
    """
    Cut-off density function

    Parameters
    ----------
    f : Scalar
        Frequency (Hz).

    Returns
    -------
    n_c : Scalar
        Cut-off density (m^-3).

    """
    omega = 2*np.pi*f
    n_c = (constants.m_e*constants.epsilom_0)/(constants.q_e**2)*omega**2
    # print('n_c =', n_c/1e19)
    return n_c

def fringe(nel, f):
    """
    Fringe calculator for a certain line integrated electron density.

    Parameters
    ----------
    nel : Scalar
        Line integrated electron density (m^-2).
    f : Scalar
        Frequency (Hz).

    Returns
    -------
    F : Scalar
        Fringe.

    """

    lmbd = constants.c/f
    k = 2*np.pi/lmbd
    phase = k/(2*cutdens(f))*nel
    F = phase/(2*np.pi)
    return F