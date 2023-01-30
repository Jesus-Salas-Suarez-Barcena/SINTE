# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 09:50:34 2023

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import pandas as pd
from numpy import random as rd
import constants
# from time_density_evolution import *
import scipy as sc
import scipy.signal as sig
import csv as csv
# from IQDemodulation import *
# from IQDemodulation_samples import *
# from sample_FFT_filter import *
# from FFT_filter import *
# from Interferometer_Signals import *
# from Chord_path import *
# from Laser_chord_objects import *
from Plateau import *
import time as tm
# import seaborn as sb
# from Interferometer import *
# from Chord_path3D import *
# import tabulet as tb
import xlwt as xl
from plot_torus import *
from mesh import *
from geometries import *
from ray_intersect_triangle import *
from scipy.spatial import Delaunay
from operations import *

def checknreflect(p0, p1, rr_tri, dx, tag=None, debug=False):
    """
    Checks if the ray goes through a triangle.

    Parameters
    ----------
    p0 : 3D vector np.array
        Initial point of the trajectory.
    p1 : 3D vector np.array
        Final point of the trajectory.
    rr_tri : 9D vector array np.array
        Points of the triangles.
    dx : Scalar
        Proximity where to check the points.
    tag : String array, optional
        Tag of each triangle: wall or mirror. The default is None.
    debug : Boolean, optional
        If true generates the plots of the reflections.
        The default is False.

    Returns
    -------
    r_ref : 3D vector np.array
        Reflected point.
    k_ref : 3D vector np.array
        Reflected direction.

    """

    index, barray, data_cond = triinsidecyl(p0, p1, rr_tri, dx)
    rr_tri_alt = rr_tri[:, index, :]
    # t3 = tm.time()
    D, N_tri_alt, T = rr_tri_alt.shape
    inter_array = np.zeros(N_tri_alt)
    tuple_array = np.zeros((3, N_tri_alt))

    for i in range(N_tri_alt):
        inter_array[i]= \
            ray_intersect_triangle2(p0, p1, rr_tri_alt[:, i, :].T)


    # print(inter_array)
    triangle_int_alt = rr_tri_alt[:, inter_array == 1, :]
    # print(triangle_int_alt)
    D, N_c, T = triangle_int_alt.shape
    # print(N_c)


    if N_c == 0:
        triangle_int_alt2 = rr_tri_alt[:, inter_array == 2, :]
        D, N_c2, T = triangle_int_alt2.shape
        if N_c2 == 0:
            v01 = p1 - p0
            k_ref = v01/np.linalg.norm(v01)
            r_ref = p1
            if isinstance(tag, (list, tuple, np.ndarray)):
                touch = 'air'
                return r_ref, k_ref, touch
        else:
            vp = p1 - p0
            k = vp/np.linalg.norm(vp)

            t0, t1, t2 = triangle_int_alt2[:, 0, :]

            v1 = t1 - t0
            v2 = t2 - t0

            N = np.cross(v1, v2)
            N = N/np.linalg.norm(N)

            if np.dot(k, N)>0:
                N = -N

            r_ref = p1
            k_ref = k - 2*N*np.dot(N, k)
            k_ref = k_ref/np.linalg.norm(k_ref)

    else:
        # print('Reflected?')
        r_ref, k_ref = reflect(p0, p1, triangle_int_alt[:, 0, :].T,
                               debugg=debug)
        # print('Sure')

    if isinstance(tag, (list, tuple, np.ndarray)):
        tag_tri = tag[index]
        tag_alt = tag_tri[inter_array == 1]
        if tag_alt[0] == 1:
            touch = 'mirror'
        elif tag_alt[0] == 2:
            touch = 'wall'
        else:
            touch = 'air'
        return r_ref, k_ref, touch
    else:
        return r_ref, k_ref


    return r_ref, k_ref



def checknreflect2(p0, p1, rr_tri, ds, tag=None, debug=False):
    """
    Checks if the ray goes through a triangle.

    Parameters
    ----------
    p0 : 3D vector np.array
        Initial point of the trajectory.
    p1 : 3D vector np.array
        Final point of the trajectory.
    rr_tri : 9D vector array np.array
        Points of the triangles.
    ds : Scalar
        Proximity where to check the points.
    tag : String array, optional
        Tag of each triangle: wall or mirror. The default is None.
    debug : Boolean, optional
        If true generates the plots of the reflections.
        The default is False.

    Returns
    -------
    r_ref : 3D vector np.array
        Reflected point.
    k_ref : 3D vector np.array
        Reflected direction.

    """

    index, cond_array = betweensphere(p0, p1, rr_tri, ds)
    v01 = p1 - p0
    k_ref = v01/np.linalg.norm(v01)

    rr_tri_alt = rr_tri[:, index, :]

    D, N_tri_alt, T = rr_tri_alt.shape
    inter_array = np.zeros(N_tri_alt)
    tuple_array = np.zeros((3, N_tri_alt))

    for i in range(N_tri_alt):
        inter_array[i]= \
            ray_intersect_triangle2(p0, p1, rr_tri_alt[:, i, :].T)
    # print(inter_array)
    triangle_int_alt = rr_tri_alt[:, inter_array == 1, :]
    # print(triangle_int_alt)
    D, N_c, T = triangle_int_alt.shape
    # print(N_c)


    if N_c == 0:
        triangle_int_alt2 = rr_tri_alt[:, inter_array == 2, :]
        D, N_c2, T = triangle_int_alt2.shape
        if N_c2 == 0:
            r_ref = p1
            if isinstance(tag, (list, tuple, np.ndarray)):
                touch = 'air'
                return r_ref, k_ref, touch
        else:
            vp = p1 - p0
            k = vp/np.linalg.norm(vp)

            t0, t1, t2 = triangle_int_alt2[:, 0, :]

            v1 = t1 - t0
            v2 = t2 - t0

            N = np.cross(v1, v2)
            N = N/np.linalg.norm(N)

            if np.dot(k, N)>0:
                N = -N

            r_ref = p1
            k_ref = k - 2*N*np.dot(N, k)
            k_ref = k_ref/np.linalg.norm(k_ref)

    else:
        # print('Reflected?')
        r_ref, k_ref = reflect(p0, p1, triangle_int_alt[:, 0, :].T,
                               debugg=debug)
        # print('Sure')

    if isinstance(tag, (list, tuple, np.ndarray)):
        tag_tri = tag[index]
        if N_c == 0:
            tag_alt = tag_tri[inter_array == 2]
        else:
            tag_alt = tag_tri[inter_array == 1]
        if tag_alt[0] == 1:
            touch = 'mirror'
        elif tag_alt[0] == 2:
            touch = 'wall'
        else:
            touch = 'air'
        return r_ref, k_ref, touch
    else:
        return r_ref, k_ref


    return r_ref, k_ref