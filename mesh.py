# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:15:57 2023

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
from operations import *
from scipy.spatial import Delaunay
# import chart_studio.plotly as py
# import plotly.graph_objs as go
# import extra_functions as ex

# %% Definitions

def plane_mesh(r1, r2, r3, r4, ds, tag='0'):
    """


    Parameters
    ----------
    r1 : vector typ np.array
        Vertex of the rectangle.
    r2 : vector typ np.array
        Vertex of the rectangle.
    r3 : vector typ np.array
        Vertex of the rectangle.
    r4 : vector typ np.array
        Vertex of the rectangle.
    ds : Scalar
        Size of the mesh.
    tag : string array, optional
        String array containing the type of object it is: mirror or wall.
        The default is '0'.

    Returns
    -------
    plane_data : Dictionary
        Contains the information on the mesh.

    """

    v_12 = r2 - r1
    v_13 = r3 - r1
    N = np.cross(v_12, v_13)
    N = N/np.linalg.norm(N)
    r_c = (r1 + r2 + r3 + r4)/4
    C = np.dot(N, r1)
    vn_12 = v_12/np.linalg.norm(v_12)
    vn_13 = v_13/np.linalg.norm(v_13)
    dr = ds*(vn_12 + vn_13)
    dx = dr[0]
    dy = dr[1]
    dz = dr[2]

    M = np.array([[v_12[0], v_13[0]],
                  [v_12[1], v_13[1]]])
    detM = np.linalg.det(M)
    dxdydz = np.array([dx, dy])
    if detM == 0:
        M = np.array([[v_12[0], v_13[0]],
                      [v_12[2], v_13[2]]])
        detM = np.linalg.det(M)
        dxdydz = np.array([dx, dz])
        if detM == 0:
            M = np.array([[v_12[1], v_13[1]],
                          [v_12[2], v_13[2]]])
            detM = np.linalg.det(M)
            dxdydz = np.array([dy, dz])
    M_inv = np.linalg.inv(M)
    dsdt = np.dot(M_inv, dxdydz)
    ds = dsdt[0]
    dt = dsdt[1]

    Nt = abs(int(1/dt))
    Ns = abs(int(1/ds))
    t = np.linspace(0, 1, num=Nt)
    s = np.linspace(0, 1, num=Ns)

    tt, ss = np.meshgrid(t, s)

    tt_flat = tt.flatten()
    ss_flat = ss.flatten()

    Points2D = np.stack([tt_flat, vv_flat]).T

    rr = np.array([[r1]]).T + np.array([[v_12]]).T*ss + np.array([[v_13]]).T*tt
    # rr_flat = np.array([[r1]]).T + np.array([[v_12]]).T*ss_flat\
    #         + np.array([[v_13]]).T*tt_flat

    tri = Delaunay(Points2D)

    # rr_tri = np.array([[r1]]).T + np.array([[v_12]]).T\
    #     *ss_flat[tri.simplices[:]]\
    #         + np.array([[v_13]]).T*tt_flat[tri.simplices[:]]
    xx_tri = r1[0] + v_12[0]*ss_flat[tri.simplices[:]] \
        + v_13[0]*tt_flat[tri.simplices[:]]
    yy_tri = r1[1] + v_12[1]*ss_flat[tri.simplices[:]] \
        + v_13[1]*tt_flat[tri.simplices[:]]
    zz_tri = r1[2] + v_12[2]*ss_flat[tri.simplices[:]] \
        + v_13[2]*tt_flat[tri.simplices[:]]

    rr_tri = [xx_tri, yy_tri, zz_tri]



    D, N_l = rr.shappe
    ones = np.ones(N_l)
    N_array = np.dot(np.array([N]).T, np.array([ones]))
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(rr[0, :, :].ravel(), rr[1, :, :].ravel(), rr[2, :, :].ravel())
    # plt.title(tag)



    plane_data = {
        'N': N,
        'N_array': N_array,
        'r_c': r_c,
        'v1': vn_12,
        'v2': vn_13,
        'dt': dt,
        'ds': ds,
        'Nt': Nt,
        'Ns': Ns,
        't': t,
        's': s,
        'rr': rr,
        }


    return plane_data


def cylinder_wall_mesh(r_c, rho, h, sec, rot, ds, tag='0'):
    """


    Parameters
    ----------
    r_c : 3D Vector np.array
        Center of the cylinder.
    rho : Scalar
        Radius of the cylinder.
    h : Scalar
        Height of the cylinder.
    sec : Scalar
        Section of the cylinder.
    rot : Matrix 3x3 np.array
        Rotation matrix from the cylinder FR to the general FR.
    ds : Scalar.
        Size of the mesh.
    tag : String array, optional
        String array containing the type of object it is: mirror or wall.
        The default is '0'.

    Returns
    -------
    wall_data : Dictionary
        Contains the information on the mesh.

    """

    N_h = int(h/ds)
    N_t = int(sec*rho/ds)
    dt = sec/N_t
    h_array = np.linspace(-h/2, h/2, num=N_h)
    if sec < 2*np.pi:
        t_array = np.linspace(0, sec, num=N_t)
    else:
        t_array = np.linspace(0, sec-dt, num=N_t)

    hh, tt = np.meshgrid(h_array, t_array)

    hh_flat = hh.flatten()
    tt_flat = tt.flatten()
    Points2D = np.vstack([hh_flat, tt_flat]).T
    tri = Delaunay(Points2D)

    hh_rav = hh_flat
    tt_rav = tt_flat
    zz_cyl = hh_rav
    xx_cyl = rho*np.cos(tt_rav)
    yy_cyl = rho*np.sin(tt_rav)
    rr_cyl = np.array([xx_cyl, yy_cyl, zz_cyl])
    rr_cylc = rr_cyl
    rr_cylc[0:1, :] = 0
    N_cyl_array = (rr_cyl - rr_cylc)/np.linalg.norm(rr_cyl - rr_cylc, axis=0)
    rr_block = FRT(rr_cyl, r_c, rot)
    N_array = FRT(rr_cyl, np.array([0, 0, 0]), rot)

    xx_tri = rr_block[0, tri.simplices[:]]
    yy_tri = rr_block[1, tri.simplices[:]]
    zz_tri = rr_block[2, tri.simplices[:]]

    rr_tri = np.array([xx_tri, yy_tri, zz_tri])


    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(rr_block[0, :], rr_block[1, :], rr_block[2, :])
    # plt.title(tag)


    wall_data ={
        'rr': rr_block,
        'N_array': N_array,
        'rr_tri': rr_tri,
        }
    return wall_data
def disc_mesh(r_c, rho, sec, rot, ds, tag='0'):
    """


    Parameters
    ----------
    r_c : 3D Vector np.array
        Center of the disc.
    rho : Scalar
        Radius of the disc.
    sec : Scalar
        Section of the disc.
    rot : Matrix 3x3 np.array
        Rotation matrix from the disc FR to the general FR.
    ds : Scalar.
        Size of the mesh.
    tag : String array, optional
        String array containing the type of object it is: mirror or wall.
        The default is '0'.

    Returns
    -------
    top_data : Dictionary
        Contains the information on the mesh.

    """

    N_r = int(rho/ds)
    rho_array = np.linspace(0, rho, num=N_r)
    # N_t = map(int, rho_array/ds)
    rr_top = np.array([])

    xx_cyl = np.array([])
    yy_cyl = np.array([])
    zz_cyl = np.array([])
    rho_array_new = np.array([])
    t_array_new = np.array([])

    for rho_ in rho_array:
            N_t = int(sec*rho_/ds) + 1
            dt = sec/N_t
            ones = np.ones(N_t)
            rho_array_step = rho_*ones
            if sec < 2*np.pi:
                t_array = np.linspace(0, sec, num=N_t)
            else:
                t_array = np.linspace(0, sec-dt, num=N_t)
            if rho_ == 0:
                x_cyl = np.array([0])
                y_cyl = np.array([0])
                z_cyl = np.array([0])
            else:
                x_cyl = rho_*np.cos(t_array)
                y_cyl = rho_*np.sin(t_array)
                z_cyl = 0*t_array

            xx_cyl = np.concatenate((xx_cyl, x_cyl))
            yy_cyl = np.concatenate((yy_cyl, y_cyl))
            zz_cyl = np.concatenate((zz_cyl, z_cyl))
            rho_array_new = np.concatenate((rho_array_new, rho_array_step))
            t_array_new = np.concatenate((t_array_new, t_array))
    Points2D = np.vstack([t_array_new, rho_array_new]).T
    tri = Delaunay(Points2D)
    rr_cyl = np.array([xx_cyl, yy_cyl, zz_cyl])
    rr_block = FRT(rr_cyl, r_c, rot)

    xx_tri = rr_block[0, tri.simplices[:]]
    yy_tri = rr_block[1, tri.simplices[:]]
    zz_tri = rr_block[2, tri.simplices[:]]
    rr_tri = np.array([xx_tri, yy_tri, zz_tri])

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(rr_block[0, :], rr_block[1, :], rr_block[2, :])
    # plt.title(tag)

    top_data = {
        'rr': rr_block,
        'rr_tri': rr_tri,
        }

    # data1=ex.plotly_trisurf(rr_block[0, :], rr_block[1, :], rr_block[2, :],
    #                      tri.simplices, colormap=cm.RdBu, plot_edges=True)
    # axis = dict(
    # showbackground=True,
    # backgroundcolor="rgb(230, 230,230)",
    # gridcolor="rgb(255, 255, 255)",
    # zerolinecolor="rgb(255, 255, 255)",
    #     )

    # layout = go.Layout(
    #          title='Moebius band triangulation',
    #          width=800,
    #          height=800,
    #          scene=dict(
    #          xaxis=dict(axis),
    #          yaxis=dict(axis),
    #          zaxis=dict(axis),
    #         aspectratio=dict(
    #             x=1,
    #             y=1,
    #             z=0.5
    #         ),
    #         )
    #         )

    # fig1 = go.Figure(data=data1, layout=layout)

    # py.iplot(fig1, filename='Disc_trisurf')
    return top_data