# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:52:01 2022

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
# import seaborn as sb
from scipy import interpolate as intp
from operations import *
from checknreflect import *


# plt.close('all')

def Chord_path3D(Plasma, Chord, Source):
    """


    Parameters
    ----------
    Plasma : Object
        Contains the info of the distributions in the plasma.
    Chord : Object
        Contains the info of the chord geometry.
    Source : Object
        Contains the info of the lightsource.

    Returns
    -------
    Density profile in chord, realistic geometry of the chord, deviation from
    the ideal case and power loss
    .

    """

# %% Parameter imputs
    # Plasma

    # Plasma radius (m)
    a_p = 0.25
    # Plasma position (m)
    R_p = 0.4
    # Peak density (m-3)
    # n_p = 1e19
    # Elongation
    k = 2
    # Toroidal magnetic field (T)
    B_T0 = Plasma.Bt
    # Magnetic center (m)
    R_0 = Plasma.R0

    # Objects mesh
    rr = Plasma.rr
    rr_tri = Plasma.rr_tri
    tag = Plasma.tag
    tag[tag == 'mirror'] = 1
    tag[tag == 'wall'] = 2
    tag = np.array(list(map(int, tag)))
    tag_tri = Plasma.tag_tri
    tag_tri[tag_tri == 'mirror'] = 1
    tag_tri[tag_tri == 'wall'] = 2
    tag_tri = np.array(list(map(int, tag_tri)))

    # Source

    # Wavelength (mu)
    lmbd = Source.lmbd
    # Frequency Stability (Hz)
    # deltaf_st = 1e5
    # # Power Stability (%)
    # deltaPP = 1
    # # Intermediate frquency (Hz)
    # deltaf = 4e7
    # # Polarization
    # E_p = np.array([0,1,0])
    r_chord_ini = Chord.r_ini
    # r_chord_fin = Chord.r_fin


    # Initial direction of chord
    # k_ini = np.divide(np.subtract(r_chord_fin, r_chord_ini),np.linalg.norm(np.subtract(r_chord_fin, r_chord_ini)))
    k_ini = Chord.k_dir


    # Maximum length
    s_max = Chord.s_max
    # Ray optics simulation
    ds = 0.01
    dx = 0.05


    # Angular frequency
    omega = 2*constants.pi*constants.c/(lmbd*1e-6)
    # Wave number
    K = 2*constants.pi/(lmbd*1e-6)
    # Cut-off density
    n_c = (constants.m_e*constants.epsilom_0*omega**2)/(constants.q_e**2)




# %% Load R Z distributions

    data_ne = Plasma.n_e
    data_Te = Plasma.T_e
    data_Br = Plasma.Br
    data_Bz = Plasma.Bz
    data_q = Plasma.q

    data_R = Plasma.R
    data_Z = Plasma.Z


    # %% Grid

    # Grid limits
    R_max = np.amax(data_R)
    R_min = np.amin(data_R)
    Z_max = np.amax(data_Z)
    Z_min = np.amin(data_Z)

    # x_ = np.linspace(-R_max, R_max, 500)
    # y_ = np.linspace(-R_max, R_max, 500)
    # z_ = np.linspace(Z_min, Z_max, 500)
    # XX, YY, ZZ = np.meshgrid(x_, y_, z_)
    RR_p, ZZ_p = np.meshgrid(data_R, data_Z)


    n_e_int2D = intp.interp2d(data_R, data_Z, data_ne)
    T_e_int2D = intp.interp2d(data_R, data_Z, data_Te)
    Br_int2D = intp.interp2d(data_R, data_Z, data_Br)
    Bz_int2D = intp.interp2d(data_R, data_Z, data_Bz)
    q_int2D = intp.interp2d(data_R, data_Z, data_q)

    # RR = np.sqrt(XX**2 + YY**2)
    # TT = np.arctan(np.divide(YY, XX)) + np.pi*(XX < 0)
# %% 3D arrays
    # n_e_3D = n_e_int2D(RR, ZZ)
    # T_e_3D = T_e_int2D(RR, ZZ)
    # Bx_3D = Br_int2D(RR, ZZ)*np.cos(TT)
    # By_3D = Br_int2D(RR, ZZ)*np.sin(TT)
    # Bz_3D = Bz_int2D(RR, ZZ)

# %% Refraction index
    # n_ref_3D = np.sqrt(np.add((1), np.multiply(-1e19/n_c, n_e_3D)))
    # grad_nx_3D, grad_ny_3D, grad_nz_3D = np.gradient(n_ref_3D, x_, y_, z_)
    n_ref = np.sqrt(np.add((1), np.multiply(-1e19/n_c, data_ne)))
    grad_nz, grad_nr = np.gradient(n_ref, data_Z, data_R)
    n_ref_int2D = intp.interp2d(data_R, data_Z, n_ref)
    grad_nr_int2D = intp.interp2d(data_R, data_Z, grad_nr)
    grad_nz_int2D = intp.interp2d(data_R, data_Z, grad_nz)



# %% Density oscillations
    if Plasma.shape_dist == 'Gauss':
        n_e_fluct = np.exp(np.multiply(-1, np.add(np.power(np.multiply(np.subtract(RR_p, Plasma.r_os), 1/float(Plasma.shape_param[0])), 2),np.power(np.multiply(np.subtract(ZZ_p, Plasma.z_os), 1/float(Plasma.shape_param[1])), 2))))

    n_e_fluct_int2D = intp.interp2d(data_R, data_Z, n_e_fluct)
# %% Interpolate 3D
    # n_e_int3D = intp.RegularGridInterpolator((x_, y_, z_), n_e_3D)
    # T_e_int3D = intp.RegularGridInterpolator((x_, y_, z_), T_e_3D)
    # Bx_int3D = intp.RegularGridInterpolator((x_, y_, z_), Bx_3D)
    # By_int3D = intp.RegularGridInterpolator((x_, y_, z_), By_3D)
    # Bz_int3D = intp.RegularGridInterpolator((x_, y_, z_), Bz_3D)
    # n_ref_int3D = intp.RegularGridInterpolator((x_, y_, z_), n_ref_3D)
    # grad_nx_int3D = intp.RegularGridInterpolator((x_, y_, z_), grad_nx_3D)
    # grad_ny_int3D = intp.RegularGridInterpolator((x_, y_, z_), grad_ny_3D)
    # grad_nz_int3D = intp.RegularGridInterpolator((x_, y_, z_), grad_nz_3D)
    # n_e_fluct_int3D = intp.RegularGridInterpolator((x_, y_, z_), n_e_fluct)


# %% Ray optics
    # Grid precision
    dr = abs(data_R[1]-data_R[0])
    dz = abs(data_Z[1]-data_Z[0])


    # Makes the loop end when the final point of the r array is further than the plane perpendicular to the straight chord at r_fin
    # d = np.linalg.norm(np.subtract(r_chord_fin, r_chord_ini))
    d_step = 0
    # i_iter = 0
    r_step = r_chord_ini
    r_sstep = r_chord_ini
    # phi = np.arctan(y/x)+((x < 0)*np.pi)
    k_sstep = k_ini

    s = 0
    P_att = 1
    k_step = k_ini

    # Path arrays (increase size with each loop)
    r_array = np.array([r_step])
    r_s_array = np.array([r_sstep])
    r_rel_array = np.array([[0, 0, 0]])
    k_array = np.array([k_ini])
    n_fluc_array = np.array([0])
    n_e_array = np.array([1])
    n_array = np.array([0])
    grad_n_array = np.array([[0, 0, 0]])
    s_array = np.array([0])
    T_e_array = np.array([0])
    B_array = np.array([[0,0,0]])
    P_array = np.array([1])
    alpha_array = np.array([0])
    alpha_alt_array = np.array([0])
    K_array =np.multiply(K, np.array([[k_ini[0],k_ini[1],0]]))
    nu_i_array = np.array([0])
    theta_array = np.array([0])
    d_t_array = np.array([0])
    touch_r_array = np.array(['air'])
    touch_s_array = np.array(['air'])
    q_array = np.array([0])
    B_norm_array = np.array([0])
    B_t_array = np.array([0])
#     while d > d_step:

# %% loop
    # for i in range(140):
    touch_r = 'Air'
    touch_s = 'Air'

    while s_max > s and touch_s != 'wall':
        r = np.sqrt(r_step[0]**2 + r_step[1]**2)
        z = r_step[2]
        if r >= R_max or r <= R_min or z >= Z_max or z <= Z_min:
            n_step = 1
            grad_nx_step = 0
            grad_ny_step = 0
            grad_nz_step = 0
            n_e_step = 0
            T_e_step = 0
            alpha_step = 0
            B_step = np.array([0, 0, 0])
            n_fluc_step = 0
            grad_step = np.array([0, 0, 0])
            q_step = 0
            B_norm = 0
        else:

            # Parameters for the iteration
            # n_step = n_ref_int3D((x, y, z))
            # grad_nx_step = grad_nx_int3D((x, y, z))
            # grad_ny_step = grad_ny_int3D((x, y, z))
            # grad_nz_step = grad_nz_int3D((x, y, z))
            # n_e_step = n_e_int3D((x, y, z))
            # n_fluc_step = n_e_fluct_int3D((x, y, z))
            # T_e_step = T_e_int3D((x, y, z))
            # Bx_step = Bx_int3D((x, y, z))
            # By_step = By_int3D((x, y, z))
            # Bz_step = Bz_int3D((x, y, z))
            # phi_step = np.arctan(y/x)+((x < 0)*np.pi)
            # Bt_step = B_T0*R_0/np.sqrt(x**2+y**2)
            # Bt_x_step = Bt_step*np.sin(phi_step)
            # Bt_y_step = Bt_step*np.cos(phi_step)
            phi_step = np.arctan(r_step[1]/r_step[0])+((r_step[0] < 0)*np.pi)
            n_step = float(n_ref_int2D(r, z))
            grad_nx_step = float(grad_nr_int2D(r, z)*np.cos(phi_step))
            grad_ny_step = float(grad_nr_int2D(r, z)*np.sin(phi_step))
            grad_nz_step = float(grad_nz_int2D(r, z))
            n_e_step = float(n_e_int2D(r, z))
            n_fluc_step = float(n_e_fluct_int2D(r, z))
            T_e_step = float(T_e_int2D(r, z))
            Bx_step = float(Br_int2D(r, z)*np.cos(phi_step))
            By_step = float(Br_int2D(r, z)*np.sin(phi_step))
            Bz_step = float(Bz_int2D(r, z))
            Bt_step = float(B_T0*R_0/r)
            Bt_x_step = Bt_step*np.sin(phi_step)
            Bt_y_step = Bt_step*np.cos(phi_step)
            q_step = float(q_int2D(r, z))

            grad_step = np.array([grad_nx_step, grad_ny_step, grad_nz_step])

            # Magnetic field
            B_step = np.array([Bx_step+Bt_x_step, By_step+Bt_y_step, Bz_step])
        B_norm = np.linalg.norm(B_step)
        I_B = np.multiply(B_step, 1/B_norm)
        # Cyclotron frequency
        omega_h = B_norm*constants.q_e*constants.mu_0/constants.m_e
        # Electron ion collision frequency
        if T_e_step > 0:
            nu_i = constants.nu_ic*n_e_step*1e19*(300*constants.Kb_eV/(T_e_step*1e3))**(3/2)
        elif T_e_step == 0:
            nu_i =0

        # Proppagation direction
        K_full =np.multiply(K, k_step)

        # Plasma frequency
        omega_p = np.sqrt(n_e_step*1e19/constants.m_e/constants.epsilom_0*constants.q_e**2)
        # dielectric matrix
        M_epsilom = np.array([[omega+1j*nu_i, 1j*omega_h*I_B[2], -1j*omega_h*I_B[1]],
                              [-1j*omega_h*I_B[2], omega+1j*nu_i, 1j*omega_h*I_B[0]],
                              [1j*omega_h*I_B[1], -1j*omega_h*I_B[0], omega+1j*nu_i]])
        epsilom_matrix = np.subtract(np.identity(3),np.multiply(omega_p**2/omega,np.linalg.inv(M_epsilom)))

        K_refracted = epsilom_matrix.dot(K_full.T)
        alpha_step = np.linalg.norm(np.imag(K_refracted))
        alpha_step_alt = K*nu_i*omega_p**2/(omega**3)

        gradk = np.dot(k_step, grad_step)
        grad_n_step = np.array([grad_nx_step, grad_ny_step,
                                grad_nz_step])
        dk_step = ds*(grad_n_step-k_step*gradk)/n_step

        k_step = k_step + dk_step
        r_pre = r_step
        r_step = r_step + k_step*ds
        r_pos = r_step

        r_spre = r_sstep
        r_sstep = r_sstep + k_sstep*ds
        r_spos = r_sstep

        # Check if the ray touches something
        r_step, k_step, touch_r = checknreflect2(r_pre, r_pos, rr_tri, dx,
                                              tag=tag_tri)
        r_sstep, k_sstep, touch_s = checknreflect2(r_spre, r_spos, rr_tri, dx,
                                              tag=tag_tri)


        d_t = np.linalg.norm(r_step-r_sstep)
        s = s + ds
        P_att = P_att*np.exp(-alpha_step*ds)


        # Distance to initial point on chord
        d_step = np.linalg.norm(r_step-r_chord_ini)
        # Direction


# %% Saving relevant variables
        # Deflection
        theta_step = np.arcsin(d_t/s)


        # saving values to respective arrays
        r_array = np.concatenate((r_array, np.array([r_step])))
        r_rel_array = np.concatenate((r_rel_array, np.array([r_step-r_chord_ini])))
        r_s_array = np.concatenate((r_s_array, np.array([r_sstep])))
        k_array = np.concatenate((k_array, np.array([k_step])))
        K_array = np.concatenate((K_array, np.array([K_refracted])))
        s_array = np.concatenate((s_array, np.array([s])))
        n_array = np.concatenate((n_array, np.array([n_step])))
        grad_n_array = np.concatenate((grad_n_array,np.array([grad_step])))
        n_e_array = np.concatenate((n_e_array,np.array([n_e_step])))
        n_fluc_array = np.concatenate((n_fluc_array, np.array([n_fluc_step])))
        T_e_array = np.concatenate((T_e_array, np.array([T_e_step])))
        B_array = np.concatenate((B_array, np.array([B_step])))
        P_array = np.concatenate((P_array, np.array([P_att])))
        alpha_array = np.concatenate((alpha_array, np.array([alpha_step])))
        alpha_alt_array = np.concatenate((alpha_alt_array,np.array([alpha_step_alt])))
        # nu_i_array = np.concatenate((nu_i_array,np.array([nu_i])))
        theta_array = np.concatenate((theta_array, np.array([theta_step])))
        d_t_array = np.concatenate((d_t_array, np.array([d_t])))
        touch_r_array = np.concatenate((touch_r_array, np.array([touch_r])))
        touch_s_array = np.concatenate((touch_s_array, np.array([touch_s])))
        q_array = np.concatenate((q_array, np.array([q_step])))
        B_norm_array = np.concatenate((B_norm_array, np.array([B_norm])))
        B_t_array = np.concatenate((B_t_array, np.array([Bt_step])))


    # i_iter = i_iter + 1



    # Power loss
    P_db = np.multiply(-10, np.log10(np.divide(P_array, P_array[0])))

    # Alfvén frequency
    v_a = np.divide(B_norm_array,
                         np.sqrt(constants.mu_0\
                                 *(constants.m_i + constants.m_e)*\
                                     n_e_array*1e19))*(n_e_array > 0)
    v_a[np.isnan(v_a)] = 0
    f_a = v_a/(4*np.pi*q_array*R_0)*(n_e_array > 0)
    f_a[np.isnan(f_a)] = 0

    n_e =  1e19*data_ne
    n_e_array_scaled = 1e19*n_e_array

    P_loss = np.amax(P_db)
    theta_max = np.amax(theta_array)*360/(2*constants.pi)

    n_fluc = np.multiply(n_fluc_array, Plasma.n_os)

    data_chord = {
        'n_e': n_e_array_scaled,
        'n': n_array,
        's': s_array,
        'r': r_array,
        'r_rel': r_rel_array,
        'r_s': r_s_array,
        'P_loss': P_loss,
        'theta': theta_max,
        'd': d_t_array,
        'n_fluc': n_fluc,
        'k': k_array,
        'grad_n': grad_n_array,
        'q': q_array,
        'B_norm': B_norm_array,
        'v_a': v_a,
        'f_a': f_a,
        'touch_r': touch_r_array,
        'touch_s': touch_s_array,
        }




    return data_chord




