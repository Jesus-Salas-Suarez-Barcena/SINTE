# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:10:03 2023

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
from mesh import *
from operations import *


def geo_model(geo_array):
    """


    Parameters
    ----------
    geo_array : Array of dictionaries
        Array containg the info on all the objects.

    Returns
    -------
    geo_array : Array of dictionaries
        Array containg the info on all the objects but meshed.

    """

    for i, geo in enumerate(geo_array):
        geo = load_geo(geo)
        geo_array[i] = geo

    return geo_array



def load_geo(geo):
    """


    Parameters
    ----------
    geo : Dictionary.
        Contains the type of object and the shape parameters:
            obj = {
                'label': 'Name',
                'type': 'mirror/wall',
                'shape': np.array(['shape', position(x, y, z)),
                                   orientation(phi, psi, theta]),
                                   size(w1 ,w2, w3)]),
                'meshed': False(Never set to True),
                }

    Returns
    -------
    geo : Dictionary.
        Adds to the input dictionary the mesh and points
        that defines the set shape.

    """


    k1_ref = np.array([1, 0, 0])
    k2_ref = np.array([0, 1, 0])
    k3_ref = np.array([0, 0, 1])

    xx_block = np.array([])
    yy_block = np.array([])
    zz_block = np.array([])
    rr_tri = np.array([[[]]])

    shape = geo['shape'][0]
    tp = geo['type']
    dx = 0.001

    if shape == 'rectangularprism':
        r_rect = geo['shape'][1]
        o_rect = geo['shape'][2]
        w_rect = geo['shape'][3]

        phi = o_rect[0]
        psi = o_rect[1]
        theta = o_rect[2]

        rot = rot_matrix(phi, psi, theta)

        k1_rect = np.dot(rot, k1_ref)
        k2_rect = np.dot(rot, k2_ref)
        k3_rect = np.dot(rot, k3_ref)


        I = w_rect*np.eye(3)/2
        k_mat = np.array([k1_rect, k2_rect, k3_rect])
        M = np.dot(I, k_mat)

        # Vertex points
        r_1 = r_rect + k1_rect*w_rect[0]/2 + k2_rect*w_rect[1]/2\
            + k3_rect*w_rect[2]/2
        r_2 = r_rect - k1_rect*w_rect[0]/2 + k2_rect*w_rect[1]/2\
            + k3_rect*w_rect[2]/2
        r_3 = r_rect + k1_rect*w_rect[0]/2 - k2_rect*w_rect[1]/2\
            + k3_rect*w_rect[2]/2
        r_4 = r_rect - k1_rect*w_rect[0]/2 - k2_rect*w_rect[1]/2\
            + k3_rect*w_rect[2]/2
        r_5 = r_rect + k1_rect*w_rect[0]/2 + k2_rect*w_rect[1]/2\
            - k3_rect*w_rect[2]/2
        r_6 = r_rect - k1_rect*w_rect[0]/2 + k2_rect*w_rect[1]/2\
            - k3_rect*w_rect[2]/2
        r_7 = r_rect + k1_rect*w_rect[0]/2 - k2_rect*w_rect[1]/2\
            - k3_rect*w_rect[2]/2
        r_8 = r_rect - k1_rect*w_rect[0]/2 - k2_rect*w_rect[1]/2\
            - k3_rect*w_rect[2]/2

        r_vertex = np.array([r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8])

        r_edges = np.array([[r_1, r_2], [r_1, r_3], [r_1, r_5],
                            [r_2, r_4], [r_2, r_6], [r_8, r_4], [r_8, r_6],
                            [r_7, r_8], [r_7, r_3], [r_7, r_5],
                            [r_3, r_4], [r_5, r_6]])
        # # Test point relative to center
        # r_testr = r_test - r_rect
        # # Test point relative to center in prism RF
        # rot_inv = np.linalg.inv(rot)
        # r_testrr = np.dot(rot_inv, r_testr)


        # Mesh of the planes
        plane1_data = plane_mesh(r_1, r_2, r_5, r_6, dx, tag='1')
        N_1 = -plane1_data['N']
        r_c1 = plane1_data['r_c']
        plane2_data = plane_mesh(r_3, r_4, r_7, r_8, dx, tag='2')
        N_2 = plane2_data['N']
        r_c2 = plane2_data['r_c']
        plane3_data = plane_mesh(r_1, r_2, r_3, r_4, dx, tag='3')
        N_3 = plane3_data['N']
        r_c3 = plane3_data['r_c']
        plane4_data = plane_mesh(r_1, r_3, r_5, r_7, dx, tag='4')
        N_4 = plane4_data['N']
        r_c4 = plane4_data['r_c']
        plane5_data = plane_mesh(r_5, r_6, r_7, r_8, dx, tag='5')
        N_5 = -plane5_data['N']
        r_c5 = plane5_data['r_c']
        plane6_data = plane_mesh(r_2, r_4, r_6, r_8, dx, tag='6')
        N_6 = -plane6_data['N']
        r_c6 = plane6_data['r_c']

        N_array = np.array([N_1, N_2, N_3, N_4, N_5, N_6])
        r_c_array = np.array([r_c1, r_c2, r_c3, r_c4, r_c5, r_c6])


        xx_block = np.concatenate((xx_block,
                                    plane1_data['rr'][0, :, :].ravel()))
        xx_block = np.concatenate((xx_block,
                                    plane2_data['rr'][0, :, :].ravel()))
        xx_block = np.concatenate((xx_block,
                                    plane3_data['rr'][0, :, :].ravel()))
        xx_block = np.concatenate((xx_block,
                                    plane4_data['rr'][0, :, :].ravel()))
        xx_block = np.concatenate((xx_block,
                                    plane5_data['rr'][0, :, :].ravel()))
        xx_block = np.concatenate((xx_block,
                                    plane6_data['rr'][0, :, :].ravel()))
        yy_block = np.concatenate((yy_block,
                                    plane1_data['rr'][1, :, :].ravel()))
        yy_block = np.concatenate((yy_block,
                                    plane2_data['rr'][1, :, :].ravel()))
        yy_block = np.concatenate((yy_block,
                                    plane3_data['rr'][1, :, :].ravel()))
        yy_block = np.concatenate((yy_block,
                                    plane4_data['rr'][1, :, :].ravel()))
        yy_block = np.concatenate((yy_block,
                                    plane5_data['rr'][1, :, :].ravel()))
        yy_block = np.concatenate((yy_block,
                                    plane6_data['rr'][1, :, :].ravel()))
        zz_block = np.concatenate((zz_block,
                                    plane1_data['rr'][2, :, :].ravel()))
        zz_block = np.concatenate((zz_block,
                                    plane2_data['rr'][2, :, :].ravel()))
        zz_block = np.concatenate((zz_block,
                                    plane3_data['rr'][2, :, :].ravel()))
        zz_block = np.concatenate((zz_block,
                                    plane4_data['rr'][2, :, :].ravel()))
        zz_block = np.concatenate((zz_block,
                                    plane5_data['rr'][2, :, :].ravel()))
        zz_block = np.concatenate((zz_block,
                                    plane6_data['rr'][2, :, :].ravel()))

        rr_block = np.array([xx_block, yy_block, zz_block])

        rr_tri = np.vstack((plane1_data['rr_tri'],
                            plane2_data['rr_tri']))
        rr_tri = np.vstack((rr_tri, plane3_data['rr_tri']))
        rr_tri = np.vstack((rr_tri, plane4_data['rr_tri']))
        rr_tri = np.vstack((rr_tri, plane5_data['rr_tri']))
        rr_tri = np.vstack((rr_tri, plane6_data['rr_tri']))


        N_p = len(xx_block)
        tag = np.zeros(N_p, 'U8')
        tag[:] = tp
        D, N_tri, T = rr_tri.shape
        tag_tri = np.zeros(N_tri, 'U8')
        tag_tri[:] = tp
        geo['tag_tri'] = tag_tri



        geo['r_vertex'] = r_vertex
        geo['r_edges'] = r_edges
        geo['N'] = N_array
        geo['r_c'] = r_c_array
        geo['k'] = k_mat
        geo['rot'] = rot
        geo['rr'] = rr_block
        geo['tag'] = tag
        geo['meshed'] = True
        geo['rr_tri'] = rr_tri


    if shape == 'cylinder':
        r_cyl = geo['shape'][1]
        o_cyl = geo['shape'][2]
        w_cyl = geo['shape'][3]
        phi = o_cyl[0]
        psi = o_cyl[1]
        theta = o_cyl[2]
        rho = w_cyl[0]
        h = w_cyl[2]
        sec = w_cyl[1]

        rot = rot_matrix(phi, psi, theta)

        k1_cyl = np.dot(rot, k1_ref)
        k2_cyl = np.dot(rot, k2_ref)
        k3_cyl = np.dot(rot, k3_ref)

        r_1 = r_cyl + k3_cyl*h/2
        r_2 = r_cyl - k3_cyl*h/2

        t = np.linspace(0, sec, num=100)
        r_circ_pre = np.array([rho*np.cos(t),
                               rho*np.sin(t),
                               0*t])
        no_rot = np.eye(3)
        r_circ1 = FRT(r_circ_pre, np.array([0, 0, h/2]), no_rot)
        r_circ2 = FRT(r_circ_pre, np.array([0, 0, -h/2]), no_rot)
        r_circ1 = FRT(r_circ1, r_cyl, rot)
        r_circ2 = FRT(r_circ2, r_cyl, rot)
        r_circ = np.array([r_circ1, r_circ2])

        cyl_wall_data = cylinder_wall_mesh(r_cyl, rho, h, sec, rot, dx)
        top_data = disc_mesh(r_1, rho, sec, rot, dx)
        bottom_data = disc_mesh(r_2, rho, sec, rot, dx)
        xx_block = np.concatenate((xx_block, cyl_wall_data['rr'][0, :]))
        xx_block = np.concatenate((xx_block, top_data['rr'][0, :]))
        xx_block = np.concatenate((xx_block, bottom_data['rr'][0, :]))
        yy_block = np.concatenate((yy_block, cyl_wall_data['rr'][1, :]))
        yy_block = np.concatenate((yy_block, top_data['rr'][1, :]))
        yy_block = np.concatenate((yy_block, bottom_data['rr'][1, :]))
        zz_block = np.concatenate((zz_block, cyl_wall_data['rr'][2, :]))
        zz_block = np.concatenate((zz_block, top_data['rr'][2, :]))
        zz_block = np.concatenate((zz_block, bottom_data['rr'][2, :]))

        rr_tri = np.concatenate((top_data['rr_tri'], bottom_data['rr_tri']),
                                axis=1)
        # print(rr_tri.shape)
        # print(cyl_wall_data['rr_tri'].shape)
        rr_tri = np.concatenate((rr_tri, cyl_wall_data['rr_tri']), axis=1)
        # print(rr_tri.shape)

        if sec < 2*np.pi:

            x_cyl3 = rho*np.cos(0)
            y_cyl3 = rho*np.sin(0)
            z_cyl3 = h/2
            r_cyl3 = np.array([x_cyl3, y_cyl3, z_cyl3])
            x_cyl4 = rho*np.cos(0)
            y_cyl4 = rho*np.sin(0)
            z_cyl4 = -h/2
            r_cyl4 = np.array([x_cyl4, y_cyl4, z_cyl4])
            x_cyl5 = rho*np.cos(sec)
            y_cyl5 = rho*np.sin(sec)
            z_cyl5 = h/2
            r_cyl5 = np.array([x_cyl5, y_cyl5, z_cyl5])
            x_cyl6 = rho*np.cos(sec)
            y_cyl6 = rho*np.sin(sec)
            z_cyl6 = -h/2
            r_cyl6 = np.array([x_cyl6, y_cyl6, z_cyl6])
            r_3 = r_cyl + np.dot(rot, r_cyl3)
            r_4 = r_cyl + np.dot(rot, r_cyl4)
            r_5 = r_cyl + np.dot(rot, r_cyl5)
            r_6 = r_cyl + np.dot(rot, r_cyl6)
            r_points = np.array([r_1, r_2, r_3, r_4, r_5])

            r_edges = np.array([[r_1, r_2], [r_3, r_4], [r_5, r_6],
                                [r_1, r_3], [r_2, r_4],
                                [r_1, r_5], [r_2, r_6]])

            plane1_data = plane_mesh(r_1, r_2, r_3, r_4, dx)
            N_1 = -plane1_data['N']
            r_c1 = plane1_data['r_c']
            plane2_data = plane_mesh(r_1, r_2, r_5, r_6, dx)
            N_2 = -plane2_data['N']
            r_c2 = plane2_data['r_c']
            N_array = np.array([N_1, N_2])
            r_c_array = np.array([r_c1, r_c2])
            xx_block = np.concatenate((xx_block, plane1_data['rr'][0, :]))
            xx_block = np.concatenate((xx_block, plane2_data['rr'][0, :]))
            yy_block = np.concatenate((yy_block, plane1_data['rr'][1, :]))
            yy_block = np.concatenate((yy_block, plane2_data['rr'][1, :]))
            zz_block = np.concatenate((zz_block, plane1_data['rr'][2, :]))
            zz_block = np.concatenate((zz_block, plane2_data['rr'][2, :]))
            rr_tri = np.concatenate((rr_tri, plane1_data['rr_tri']), axis=1)
            rr_tri = np.concatenate((rr_tri, plane2_data['rr_tri']), axis=1)
        else:
            r_points = np.array([r_1, r_2])
            r_edges = np.array([])
            N_array = None
            r_c_array = None


        N_p = len(xx_block)
        tag = np.zeros(N_p, 'U8')
        tag[:] = tp
        rr_block = np.array([xx_block, yy_block, zz_block])

        geo['r_points'] = r_points
        geo['r_edges'] = r_edges
        geo['r_circ'] = r_circ
        geo['N'] = N_array
        geo['r_c'] = r_c_array
        geo['rot'] = rot
        geo['rr'] = rr_block
        geo['tag'] = tag
        geo['meshed'] = True
        geo['rr_tri'] = rr_tri

        D, N_tri, T = rr_tri.shape
        tag_tri = np.zeros(N_tri, 'U8')
        tag_tri[:] = tp
        geo['tag_tri'] = tag_tri


    return geo



def plot_geo(geo_array, ax):
    """


    Parameters
    ----------
    geo_array : Array of dictionaries
        Array containg the info on all the objects to plot.
    ax : Subplot object.
        subplot where to plot the shape.

    Returns
    -------
    plot_array : Array of dictionaries.
        Information on plots.
    geo_array :  Dictionary.
        Same as for load geo.

    """

    plot_array = []
    for i, geo in enumerate(geo_array):

        plot_info, geo = plot_obj(geo, ax)
        plot_array.append(plot_info)
        geo_array[i] = geo

    return plot_array, geo_array

def plot_obj(geo, ax):
    """


    Parameters
    ----------
    geo : Dictionary.
        Same as for load geo.
    ax : Subplot object.
        subplot where to plot the shape.

    Returns
    -------
    plot_info : Dictionary.
        Information on plots.
    geo : Dictionary.
        Same as for load geo.

    """
    plot_info = {}

    temp_array = []
    temp_e_array = []
    temp_cyl_array = []
    if geo['shape'][0] == 'rectangularprism':
        if geo['meshed']:
            for i in range(8):
                r = geo['r_vertex'][i, :]
                temp1 = ax.plot([r[0]], [r[1]], [r[2]],
                                markerfacecolor='r', markeredgecolor='r',
                                marker='o', markersize=5, alpha=0.6)
                temp2 = ax.text(r[0], r[1], r[2], 'P' + str(i+1))
                temp = np.array([temp1, temp2])
                temp_array.append(temp)

            for j in range(12):
                r_e = geo['r_edges'][j, :, :]
                temp_e = ax.plot3D(r_e[:, 0], r_e[:, 1], r_e[:, 2], color='k')
                temp_e_array.append(temp_e)
            r_rect = geo['shape'][1]
            temp_rect1 = ax.plot([r_rect[0]], [r_rect[1]], [r_rect[2]],
                            markerfacecolor='r', markeredgecolor='r',
                            marker='o', markersize=5, alpha=0.6)
            temp_rect2 = ax.text(r_rect[0], r_rect[1], r_rect[2], geo['label'])
            temp_rect = [temp_rect1, temp_rect2]
        else:
            geo = load_geo(geo)
            for i in range(8):
                r = geo['r_vertex'][i, :]
                temp1 = ax.plot([r[0]], [r[1]], [r[2]],
                                markerfacecolor='r', markeredgecolor='r',
                                marker='o', markersize=5, alpha=0.6)
                temp2 = ax.text(r[0], r[1], r[2], 'P' + str(i+1))
                temp = np.array([temp1, temp2])
                temp_array.append(temp)

            for j in range(12):
                r_e = geo['r_edges'][j, :, :]
                temp_e = ax.plot3D(r_e[:, 0], r_e[:, 1], r_e[:, 2], color='k')
                temp_e_array.append(temp_e)
            r_rect = geo['shape'][1]
            temp_rect1 = ax.plot([r_rect[0]], [r_rect[1]], [r_rect[2]],
                            markerfacecolor='r', markeredgecolor='r',
                            marker='o', markersize=5, alpha=0.6)
            temp_rect2 = ax.text(r_rect[0], r_rect[1], r_rect[2], geo['label'])
            temp_rect = [temp_rect1, temp_rect2]
        plot_info = {
            'Vertex plot': temp_array,
            'Edges plot': temp_e_array,
            'Center plot': temp_rect
            }

    if geo['shape'][0] == 'cylinder':
            if geo['meshed']:
                N_p, D_p = geo['r_points'].shape
                for i in range(N_p):
                    r = geo['r_points'][i, :]
                    temp1 = ax.plot([r[0]], [r[1]], [r[2]],
                                    markerfacecolor='r', markeredgecolor='r',
                                    marker='o', markersize=5, alpha=0.6)
                    temp2 = ax.text(r[0], r[1], r[2], 'P' + str(i+1))
                    temp = np.array([temp1, temp2])
                    temp_array.append(temp)
                if len(geo['r_edges'])>0:
                    N_e, D_e = geo['r_edges'].shape
                    for i in range():
                        r_e = geo['r_edges'][j, :, :]
                        temp_e = ax.plot3D(r_e[:, 0], r_e[:, 1], r_e[:, 2],
                                           color='k')
                        temp_e_array.append(temp_e)
                r_e = geo['r_circ']
                temp_e = ax.plot3D(r_e[0, 0, :], r_e[0, 1, :],
                                   r_e[0, 2, :], color='k')
                temp_e_array.append(temp_e)
                temp_e = ax.plot3D(r_e[1, 0, :], r_e[1, 1, :],
                                   r_e[1, 2, :], color='k')
                temp_e_array.append(temp_e)

                for j in range(4):
                    temp_cyl = ax.plot3D(np.array([r_e[0, 0, 25*j],
                                                   r_e[1, 0, 25*j]]),
                                         np.array([r_e[0, 1, 25*j],
                                                  r_e[1, 1, 25*j]]),
                                         np.array([r_e[0, 2, 25*j],
                                                   r_e[1, 2, 25*j]]),
                                         color='k')
                    temp_cyl_array.append(temp_cyl)
                v_disc = r_e[0, :, 0] - geo['r_points'][0, :]
                v_wall = r_e[0, :, 0] - r_e[1, :, 0]
                alfa = np.arcsin(np.dot(v_disc, v_wall)\
                                 /(np.linalg.norm(v_disc)*np.linalg.norm(v_wall)))
            else:
                geo = load_geo(geo)
                N_p, D_p = geo['r_points'].shape
                for i in range(N_p):
                    r = geo['r_points'][i, :]
                    temp1 = ax.plot([r[0]], [r[1]], [r[2]],
                                    markerfacecolor='r', markeredgecolor='r',
                                    marker='o', markersize=5, alpha=0.6)
                    temp2 = ax.text(r[0], r[1], r[2], 'P' + str(i+1))
                    temp = np.array([temp1, temp2])
                    temp_array.append(temp)
                if len(geo['r_edges'])>0:
                    N_e, D_e = geo['r_edges'].shape
                    for i in range():
                        r_e = geo['r_edges'][j, :, :]
                        temp_e = ax.plot3D(r_e[:, 0], r_e[:, 1], r_e[:, 2],
                                           color='k')
                        temp_e_array.append(temp_e)
                r_e = geo['r_circ']
                temp_e = ax.plot3D(r_e[0, 0, :], r_e[0, 1, :],
                                   r_e[0, 2, :], color='k')
                temp_e_array.append(temp_e)
                temp_e = ax.plot3D(r_e[1, 0, :], r_e[1, 1, :],
                                   r_e[1, 2, :], color='k')
                temp_e_array.append(temp_e)
                for j in range(4):
                    temp_cyl = ax.plot3D(np.array([r_e[0, 0, 25*j],
                                                   r_e[1, 0, 25*j]]),
                                         np.array([r_e[0, 1, 25*j],
                                                  r_e[1, 1, 25*j]]),
                                         np.array([r_e[0, 2, 25*j],
                                                   r_e[1, 2, 25*j]]),
                                         color='k')
                    temp_cyl_array.append(temp_cyl)
                # v_disc = r_e[0, :, 0] - geo['r_points'][0, :]
                # v_wall = r_e[0, :, 0] - r_e[1, :, 0]
                # alfa = np.arcsin(np.dot(v_disc, v_wall)\
                #                  /(np.linalg.norm(v_disc)*np.linalg.norm(v_wall)))
            r_cyl = geo['shape'][1]
            temp_cyl1 = ax.plot([r_cyl[0]], [r_cyl[1]], [r_cyl[2]],
                            markerfacecolor='r', markeredgecolor='r',
                            marker='o', markersize=5, alpha=0.6)
            temp_cyl2 = ax.text(r_cyl[0], r_cyl[1], r_cyl[2], geo['label'])
            temp_cyl = [temp_cyl1, temp_cyl2]

            plot_info = {
                'Vertex plot': temp_array,
                'Edges plot': temp_e_array,
                'Center plot': temp_cyl,
                'faces plot': temp_cyl_array
                }

    return plot_info, geo
