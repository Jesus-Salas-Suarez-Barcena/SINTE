# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:07:29 2022

@author: jesus
"""

# %% Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import pandas as pd
from numpy import random as rd
import constants
from time_density_evolution import *
import scipy as sc
import scipy.signal as sig
import csv as csv
from IQDemodulation import *
from IQDemodulation_samples import *
from sample_FFT_filter import *
from FFT_filter import *
from Interferometer_Signals import *
# from Chord_path import *
from Laser_chord_objects import *
from Plateau import *
import time as tm
# import seaborn as sb
from Interferometer import *
from Chord_path3D import *
# import tabulet as tb
import xlwt as xl
from plot_torus import *
from operations import *
# %% Close
plt.close('all')

# %% Plasma

# Density oscillations
density_oscillations = {
    'f_os': 1e5,
    'n_os': 1e17,
    'r_os': 0.45,
    'z_os': 0.0,
    'Shape': np.array(['Gauss', 0.01, 0.02]),
    'Type': 'Cosine',
}

# Vibrations
vibrations = {
    'dl_vib': 1e-5,
    'f_vib': 1e1,
}

# Objects(Walls and mirrors)
r_mirrorc = np.array([0, 0.15, 0])
r_mirroru = np.array([0, 0.15, 0.45])
r_mirrorl = np.array([0, 0.15, -0.45])
obj1 = {
    'label': 'Central Mirror',
    'type': 'mirror',
    'shape': np.array(['cylinder', r_mirrorc,
                       np.array([np.pi/2, 0, 0]),
                       np.array([0.05, 2*np.pi, 0.01])]),
    'meshed': False,
    }
obj2 = {
    'label': 'Upper Mirror',
    'type': 'mirror',
    'shape': np.array(['cylinder', r_mirroru,
                       np.array([np.pi/2, 0, 0]),
                       np.array([0.05, 2*np.pi, 0.01])]),
    'meshed': False,
    }
obj3 = {
    'label': 'Lower Mirror',
    'type': 'mirror',
    'shape': np.array(['cylinder', r_mirrorl,
                       np.array([np.pi/2, 0, 0]),
                       np.array([0.05, 2*np.pi, 0.01])]),
    'meshed': False,
    }
# obj2 = {
#     'label': 'Central wall',
#     'type': 'wall',
#     'shape': np.array(['cylinder', np.array([0, 0, 0]),
#                        np.array([0, 0, 0]),
#                        np.array([0.14, 2*np.pi, 1.6])]),
#     'meshed': False,
#     }

objects = [obj1, obj2, obj3]
# Generate plasma from files and setting the toroidal field

# Old profiles
# Phase 1
# Plasma1 = Plasma('Rcoordinate.txt', 'Zcoordinate.txt', 'ne_S1000020B_RZ.txt',
#                  'Te_S1000020B_RZ.txt', 'Br_S1000020B_RZ.txt',
#                  'Bz_S1000020B_RZ.txt', 0.1, 0.4, density_oscillations)
# # Phase 2
# Plasma2 = Plasma('Rcoordinate.txt', 'Zcoordinate.txt', 'ne_S2000021B_RZ.txt',
#                  'Te_S2000021B_RZ.txt', 'Br_S2000021B_RZ.txt',
#                  'Bz_S2000021B_RZ.txt', 0.3, 0.4, density_oscillations)
# # Phase 3
# Plasma3 = Plasma('Rcoordinate.txt', 'Zcoordinate.txt', 'ne_S3000006B_RZ.txt',
#                  'Te_S3000006B_RZ.txt', 'Br_S3000006B_RZ.txt',
#                  'Bz_S3000006B_RZ.txt', 1, 0.4, density_oscillations)

# New profiles
# Phase 1 positive triangularity
Plasma1pd = Plasma('Rcoordinate_S1028-+delta.txt',
                   'Zcoordinate_S1028-+delta.txt', 'ne_S1028-+delta_RZ.txt',
                   'Te_S1028-+delta_RZ.txt', 'Br_S1028-+delta_RZ.txt',
                   'Bz_S1028-+delta_RZ.txt', 'q_S1028-+deltaRZ.txt',
                   0.1, 0.4, density_oscillations, objects)
# Phase 1 negative triangularity
Plasma1nd = Plasma('Rcoordinate_S1028--delta.txt',
                   'Zcoordinate_S1028--delta.txt', 'ne_S1028--delta_RZ.txt',
                   'Te_S1028--delta_RZ.txt', 'Br_S1028--delta_RZ.txt',
                   'Bz_S1028--delta_RZ.txt', 'q_S1028--deltaRZ.txt',
                   0.1, 0.4, density_oscillations, objects)
# Phase 2 positive triangularity
Plasma2pd = Plasma('Rcoordinate_S2023-+delta.txt',
                   'Zcoordinate_S2023-+delta.txt', 'ne_S2023-+delta_RZ.txt',
                   'Te_S2023-+delta_RZ.txt', 'Br_S2023-+delta_RZ.txt',
                   'Bz_S2023-+delta_RZ.txt', 'q_S2023-+deltaRZ.txt',
                   0.4, 0.4, density_oscillations, objects)
# Phase 2 negative triangularity
Plasma2nd = Plasma('Rcoordinate_S2023--delta.txt',
                   'Zcoordinate_S2023--delta.txt', 'ne_S2023--delta_RZ.txt',
                   'Te_S2023--delta_RZ.txt', 'Br_S2023--delta_RZ.txt',
                   'Bz_S2023--delta_RZ.txt', 'q_S2023--deltaRZ.txt',
                   0.4, 0.4, density_oscillations, objects)

# %% Chords
r_HM = np.array([0.1, 1, 0.0])
r_HU = np.array([0.1, 1, 0.45])
r_HL = np.array([0.1, 1, -0.45])
r_VM = np.array([0.0, 0.45, 0.8])
r_VI = np.array([0.0, 0.30, 0.8])
r_VO = np.array([0.0, 0.60, 0.8])
rf_HM = np.array([0.0, 0.152, 0.0])
rf_HU = np.array([0.0, 0.152, 0.45])
rf_HL = np.array([0.0, 0.152, -0.45])
rf_VM = np.array([0.0, 0.45, -0.8])
rf_VI = np.array([0.0, 0.30, -0.8])
rf_VO = np.array([0.0, 0.60, -0.8])
k_HM =(rf_HM - r_HM)/np.linalg.norm(rf_HM - r_HM)
k_HU =(rf_HU - r_HU)/np.linalg.norm(rf_HU - r_HU)
k_HL =(rf_HL - r_HL)/np.linalg.norm(rf_HL - r_HL)
k_VM =(rf_VM - r_VM)/np.linalg.norm(rf_VM - r_VM)
k_VI =(rf_VI - r_VI)/np.linalg.norm(rf_VI - r_VI)
k_VO =(rf_VO - r_VO)/np.linalg.norm(rf_VO - r_VO)


HM = Chord('HM', r_HM, k_HM, 2, 50, 50, 50)
HU = Chord('HU', r_HU, k_HU, 2, 50, 50, 50)
HL = Chord('HL', r_HL, k_HL, 2, 50, 50, 50)
VM = Chord('VM', r_VM, k_VM, 2, 50, 50, 50)
VI = Chord('VI', r_VI, k_VI, 2, 50, 50, 50)
VO = Chord('VO', r_VO, k_VO, 2, 50, 50, 50)
# HM_back = Chord('HM_back', np.array([0.0, 0.15, 0.0]), np.array([0.1, 1.0, 0.0]),
#            50, 50, 50)

# Chords = [HM, HU, HL, VM, VI, VO]
Chords = [HM]


# %% Source

# CO2_1pd = Source(10.6, 4e7, 6e4, 0.1)
# HeNe_1pd = Source(0.633, 4e7, 1e6, 0.1)
# CO2CH2F2_1pd = Source(214.6, 1e6, 1e5, 1)
# MW110_1pd = Source(2750, 1e6, 1e5, 1)
# MW280_1pd = Source(1050, 1e6, 1e5, 1)
# MW400_1pd = Source(750, 1e6, 1e5, 1)
# CO2_1nd = Source(10.6, 4e7, 6e4, 0.1)
# HeNe_1nd = Source(0.633, 4e7, 1e6, 0.1)
# CO2CH2F2_1nd = Source(214.6, 1e6, 1e5, 1)
# MW110_1nd = Source(2750, 1e6, 1e5, 1)
# MW280_1nd = Source(1050, 1e6, 1e5, 1)
# MW400_1nd = Source(750, 1e6, 1e5, 1)
# CO2_2pd = Source(10.6, 4e7, 6e4, 0.1)
# HeNe_2pd = Source(0.633, 4e7, 1e6, 0.1)
# CO2CH2F2_2pd = Source(214.6, 1e6, 1e5, 1)
# MW110_2pd = Source(2750, 1e6, 1e5, 1)
# MW280_2pd = Source(1050, 1e6, 1e5, 1)
# MW400_2pd = Source(750, 1e6, 1e5, 1)
# CO2_2nd = Source(10.6, 4e7, 6e4, 0.1)
# HeNe_2nd = Source(0.633, 4e7, 1e6, 0.1)
# CO2CH2F2_2nd = Source(214.6, 1e6, 1e5, 1)
# MW110_2nd = Source(2750, 1e6, 1e5, 1)
# MW280_2nd = Source(1050, 1e6, 1e5, 1)
# MW400_2nd = Source(750, 1e6, 1e5, 1)

# Noiseless
CO2_1pd = Source(10.6, 4e7, 0, 0)
HeNe_1pd = Source(0.633, 4e7, 0, 0)
CO2CH2F2_1pd = Source(214.6, 1e6, 0, 0)
MW110_1pd = Source(2750, 1e6, 0, 0)
MW280_1pd = Source(1050, 1e6, 0, 0)
MW400_1pd = Source(750, 1e6, 0, 0)
CO2_1nd = Source(10.6, 4e7, 0, 0)
HeNe_1nd = Source(0.633, 4e7, 0, 0)
CO2CH2F2_1nd = Source(214.6, 1e6, 0, 0)
MW110_1nd = Source(2750, 1e6, 0, 0)
MW280_1nd = Source(1050, 1e6, 0, 0)
MW400_1nd = Source(750, 1e6, 0, 0)
CO2_2pd = Source(10.6, 4e7, 0, 0)
HeNe_2pd = Source(0.633, 4e7, 0, 0)
CO2CH2F2_2pd = Source(214.6, 1e6, 0, 0)
MW110_2pd = Source(2750, 1e6, 0, 0)
MW280_2pd = Source(1050, 1e6, 0, 0)
MW400_2pd = Source(750, 1e6, 0, 0)
CO2_2nd = Source(10.6, 4e7, 0, 0)
HeNe_2nd = Source(0.633, 4e7, 0, 0)
CO2CH2F2_2nd = Source(214.6, 1e6, 0, 0)
MW110_2nd = Source(2750, 1e6, 0, 0)
MW280_2nd = Source(1050, 1e6, 0, 0)
MW400_2nd = Source(750, 1e6, 0, 0)





# %% Compute paths

CO2_1pd.compute(Plasma1pd, Chords)
HeNe_1pd.compute(Plasma1pd, Chords)
CO2CH2F2_1pd.compute(Plasma1pd, Chords)
MW110_1pd.compute(Plasma1pd, Chords)
MW280_1pd.compute(Plasma1pd, Chords)
MW400_1pd.compute(Plasma1pd, Chords)
CO2_1nd.compute(Plasma1nd, Chords)
HeNe_1nd.compute(Plasma1nd, Chords)
CO2CH2F2_1nd.compute(Plasma1nd, Chords)
MW110_1nd.compute(Plasma1nd, Chords)
MW280_1nd.compute(Plasma1nd, Chords)
MW400_1nd.compute(Plasma1nd, Chords)
CO2_2pd.compute(Plasma2pd, Chords)
HeNe_2pd.compute(Plasma2pd, Chords)
CO2CH2F2_2pd.compute(Plasma2pd, Chords)
MW110_2pd.compute(Plasma2pd, Chords)
MW280_2pd.compute(Plasma2pd, Chords)
MW400_2pd.compute(Plasma2pd, Chords)
CO2_2nd.compute(Plasma2nd, Chords)
HeNe_2nd.compute(Plasma2nd, Chords)
CO2CH2F2_2nd.compute(Plasma2nd, Chords)
MW110_2nd.compute(Plasma2nd, Chords)
MW280_2nd.compute(Plasma2nd, Chords)
MW400_2nd.compute(Plasma2nd, Chords)

# %% Chord density profile

fig_c = plt.figure()
ax_c = fig_c.add_subplot(1, 1, 1)
ax_c.plot(MW110_1pd.s[:, 0], MW110_1pd.n_e[:, 0])
ax_c.set_xlabel('s (m)')
ax_c.set_ylabel(r'$n_e\;(m^{-3})$')

# %% Chord Alfven
fig_faHM = plt.figure()
ax_faHM = fig_faHM.add_subplot(1, 1, 1)
plt.title('HM in S1028+d')
ax_faHM.plot(MW110_1pd.s[:, 0], MW110_1pd.f_a[:, 0])
ax_faHM.set_xlabel('s (m)')
ax_faHM.set_ylabel(r'$f_a\;(Hz)$')

# %% RZ alfven
fig_faRZ = plt.figure()
ax_faRZ = fig_faRZ.add_subplot(1, 1, 1)
plt.title('HM in S1028+d')
RR, ZZ = np.meshgrid(Plasma1pd.R, Plasma1pd.Z)
pmapRZ = ax_faRZ.pcolormesh(RR, ZZ, Plasma1pd.f_a, shading = 'auto',)
cbar_faRZ = plt.colorbar(pmapRZ)
ax_faRZ.set_xlabel('R (m)')
ax_faRZ.set_ylabel('Z (m)')
plt.xlim([0.1, 0.9])
plt.ylim([-0.8, 0.8])
cbar_faRZ.set_label(r'$f_a\;(Hz)$')
ax_faRZ.set_aspect('equal')
# fig_faRZ.savefig('AlfvenRZ_S1028.png', dpi=900)


# %% Chord data
n_e_l = np.zeros((CO2_1pd.N_chords, 20))
theta = np.zeros((CO2_1pd.N_chords, 20))
P_loss = np.zeros((CO2_1pd.N_chords, 20))
lmbd = np.zeros((CO2_1pd.N_chords, 20))
N_chords = len(Chords)
deltal = 1.0  # m

book = xl.Workbook(encoding='utf-8')
sheet1 = book.add_sheet('Sheet 1')
sheet1.write(1, 0, 'Chords')
sheet1.write(2, 0, 'CO2')
sheet1.write(3, 0, 'CO2')
sheet1.write(4, 0, 'CO2')
sheet1.write(5, 0, 'CO2')
sheet1.write(6, 0, 'CO2CH2F2')
sheet1.write(7, 0, 'CO2CH2F2')
sheet1.write(8, 0, 'CO2CH2F2')
sheet1.write(9, 0, 'CO2CH2F2')
sheet1.write(10, 0, 'Microwave')
sheet1.write(11, 0, 'Microwave')
sheet1.write(12, 0, 'Microwave')
sheet1.write(13, 0, 'Microwave')
sheet1.write(14, 0, 'Microwave')
sheet1.write(15, 0, 'Microwave')
sheet1.write(16, 0, 'Microwave')
sheet1.write(17, 0, 'Microwave')
sheet1.write(18, 0, 'Microwave')
sheet1.write(19, 0, 'Microwave')
sheet1.write(20, 0, 'Microwave')
sheet1.write(21, 0, 'Microwave')
sheet1.write(0, 1, 'Wavelength (um)')

for i, chord in enumerate(Chords):
    data_CO2_1pd = CO2_1pd(chord.name)
    data_CO2_2pd = CO2_2pd(chord.name)
    data_CO2_1nd = CO2_1nd(chord.name)
    data_CO2_2nd = CO2_2nd(chord.name)
    data_CO2CH2F2_1pd = CO2CH2F2_1pd(chord.name)
    data_CO2CH2F2_2pd = CO2CH2F2_2pd(chord.name)
    data_CO2CH2F2_1nd = CO2CH2F2_1nd(chord.name)
    data_CO2CH2F2_2nd = CO2CH2F2_2nd(chord.name)
    data_MW110_1pd = MW110_1pd(chord.name)
    data_MW110_2pd = MW110_2pd(chord.name)
    data_MW110_1nd = MW110_1nd(chord.name)
    data_MW110_2nd = MW110_2nd(chord.name)
    data_MW280_1pd = MW280_1pd(chord.name)
    data_MW280_2pd = MW280_2pd(chord.name)
    data_MW280_1nd = MW280_1nd(chord.name)
    data_MW280_2nd = MW280_2nd(chord.name)
    data_MW400_1pd = MW400_1pd(chord.name)
    data_MW400_2pd = MW400_2pd(chord.name)
    data_MW400_1nd = MW400_1nd(chord.name)
    data_MW400_2nd = MW400_2nd(chord.name)
    n_e_l_chord = np.array([data_CO2_1pd['n_e_l'], data_CO2_1nd['n_e_l'],
                            data_CO2_2pd['n_e_l'], data_CO2_2nd['n_e_l'],
                            data_CO2CH2F2_1pd['n_e_l'],
                            data_CO2CH2F2_1nd['n_e_l'],
                            data_CO2CH2F2_2pd['n_e_l'],
                            data_CO2CH2F2_2nd['n_e_l'],
                            data_MW110_1pd['n_e_l'], data_MW110_1nd['n_e_l'],
                            data_MW110_2pd['n_e_l'], data_MW110_2nd['n_e_l'],
                            data_MW280_1pd['n_e_l'], data_MW280_1nd['n_e_l'],
                            data_MW280_2pd['n_e_l'], data_MW280_2nd['n_e_l'],
                            data_MW400_1pd['n_e_l'], data_MW400_1nd['n_e_l'],
                            data_MW400_2pd['n_e_l'], data_MW400_2nd['n_e_l']])
    n_e_l[i, :] = n_e_l_chord.T
    theta_chord = np.array([data_CO2_1pd['theta'], data_CO2_1nd['theta'],
                            data_CO2_2pd['theta'], data_CO2_2nd['theta'],
                            data_CO2CH2F2_1pd['theta'],
                            data_CO2CH2F2_1nd['theta'],
                            data_CO2CH2F2_2pd['theta'],
                            data_CO2CH2F2_2nd['theta'],
                            data_MW110_1pd['theta'], data_MW110_1nd['theta'],
                            data_MW110_2pd['theta'], data_MW110_2nd['theta'],
                            data_MW280_1pd['theta'], data_MW280_1nd['theta'],
                            data_MW280_2pd['theta'], data_MW280_2nd['theta'],
                            data_MW400_1pd['theta'], data_MW400_1nd['theta'],
                            data_MW400_2pd['theta'], data_MW400_2nd['theta']])
    theta[i, :] = theta_chord.T
    P_loss_chord = np.array([data_CO2_1pd['P_loss'], data_CO2_1nd['P_loss'],
                            data_CO2_2pd['P_loss'], data_CO2_2nd['P_loss'],
                            data_CO2CH2F2_1pd['P_loss'],
                            data_CO2CH2F2_1nd['P_loss'],
                            data_CO2CH2F2_2pd['P_loss'],
                            data_CO2CH2F2_2nd['P_loss'],
                            data_MW110_1pd['P_loss'], data_MW110_1nd['P_loss'],
                            data_MW110_2pd['P_loss'], data_MW110_2nd['P_loss'],
                            data_MW280_1pd['P_loss'], data_MW280_1nd['P_loss'],
                            data_MW280_2pd['P_loss'], data_MW280_2nd['P_loss'],
                            data_MW400_1pd['P_loss'], data_MW400_1nd['P_loss'],
                            data_MW400_2pd['P_loss'],
                            data_MW400_2nd['P_loss']])
    P_loss[i, :] = P_loss_chord.T
    lmbd_chord = np.array([data_CO2_1pd['lmbd'], data_CO2_1nd['lmbd'],
                           data_CO2_2pd['lmbd'], data_CO2_2nd['lmbd'],
                           data_CO2CH2F2_1pd['lmbd'],
                           data_CO2CH2F2_1nd['lmbd'],
                           data_CO2CH2F2_2pd['lmbd'],
                           data_CO2CH2F2_2nd['lmbd'],
                           data_MW110_1pd['lmbd'], data_MW110_1nd['lmbd'],
                           data_MW110_2pd['lmbd'], data_MW110_2nd['lmbd'],
                           data_MW280_1pd['lmbd'], data_MW280_1nd['lmbd'],
                           data_MW280_2pd['lmbd'], data_MW280_2nd['lmbd'],
                           data_MW400_1pd['lmbd'], data_MW400_1nd['lmbd'],
                           data_MW400_2pd['lmbd'], data_MW400_2nd['lmbd']])
    lmbd[i, :] = lmbd_chord.T
    k_chord = np.divide(2*constants.pi, lmbd_chord*1e-6)
    omega_chord = constants.c*k_chord
    n_c_chord = (constants.m_e*constants.epsilom_0)/(constants.q_e**2)*np.power(omega_chord, 2)
    Fringe_chord = np.multiply(np.divide(1/(2*constants.pi)*k_chord,
                                         2*n_c_chord), n_e_l_chord.T)
    dF_chord = Fringe_chord*0.001
    deltaf_st_chord = dF_chord*constants.c/deltal
    deltaPP_chord = 2*constants.pi*dF_chord
    sheet1.write(1, i+2, chord.name)
    sheet1.write(1, i+2+N_chords, chord.name)
    sheet1.write(1, i+2+2*N_chords, chord.name)
    sheet1.write(1, i+2+3*N_chords, chord.name)
    sheet1.write(1, i+2+4*N_chords, chord.name)
    sheet1.write(1, i+2+5*N_chords, chord.name)
    sheet1.write(1, i+2+6*N_chords, chord.name)
    sheet1.write(0, i+2, 'n_e_l')
    sheet1.write(0, i+2+N_chords, 'theta')
    sheet1.write(0, i+2+2*N_chords, 'P_loss')
    sheet1.write(0, i+2+3*N_chords, 'Fringes')
    sheet1.write(0, i+2+4*N_chords, 'dF at 0.1 %')
    sheet1.write(0, i+2+5*N_chords, 'deltaf_st')
    sheet1.write(0, i+2+6*N_chords, 'deltaPP')

    for j in range(20):
        if i == 0:
            sheet1.write(j+2, 1, float(lmbd_chord[j]))
        sheet1.write(j+2, i+2, float(n_e_l_chord[j]))
        sheet1.write(j+2, i+2+N_chords, float(theta_chord[j]))
        sheet1.write(j+2, i+2+2*N_chords, float(P_loss_chord[j]))
        sheet1.write(j+2, i+2+3*N_chords, float(Fringe_chord[0, j]))
        sheet1.write(j+2, i+2+4*N_chords, float(dF_chord[0, j]))
        sheet1.write(j+2, i+2+5*N_chords, float(deltaf_st_chord[0, j]))
        sheet1.write(j+2, i+2+6*N_chords, float(deltaPP_chord[0, j]))


book.save("Chords_data.xls")
# %% Plasma plot
figp_1 = plt.figure()
axp1pd = figp_1.add_subplot(1, 2, 1)
axp1pd.set_title('Phase 1: S1028+delta')
pmap_1pd = Plasma1pd.plot(axp1pd)
cbar_dens_p1pd = plt.colorbar(pmap_1pd)
axp1pd.plot(np.array([0.9, 0.1]), np.array([0.0, 0.0]), color='r')
axp1pd.plot(np.array([0.9, 0.1]), np.array([0.45, 0.45]), color='r')
axp1pd.plot(np.array([0.9, 0.1]), np.array([-0.45, -0.45]), color='r')
axp1pd.plot(np.array([0.45, 0.45]), np.array([0.8, -0.8]), color='r')
axp1pd.plot(np.array([0.30, 0.30]), np.array([0.8, -0.8]), color='r')
axp1pd.plot(np.array([0.60, 0.60]), np.array([0.8, -0.8]), color='r')
axp1pd.text(0.8, 0.05, 'HM', color='w')
axp1pd.text(0.8, 0.50, 'HU', color='w')
axp1pd.text(0.8, -0.40, 'HL', color='w')
axp1pd.text(0.35, 0.6, 'VM', color='w')
axp1pd.text(0.24, 0.6, 'VI', color='w')
axp1pd.text(0.50, 0.6, 'VO', color='w')
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.xlim([0.1, 0.9])
plt.ylim([-0.8, 0.8])
cbar_dens_p1pd.set_label(r'$n_e\;(m^{-3})$')
axp1pd.set_aspect('equal')
axp1nd = figp_1.add_subplot(1, 2, 2)
axp1nd.set_title('Phase 1: S1028-delta')
pmap_1nd = Plasma1nd.plot(axp1nd)
cbar_dens_p1nd = plt.colorbar(pmap_1nd)
axp1nd.plot(np.array([0.9, 0.1]), np.array([0.0, 0.0]), color='r')
axp1nd.plot(np.array([0.9, 0.1]), np.array([0.45, 0.45]), color='r')
axp1nd.plot(np.array([0.9, 0.1]), np.array([-0.45, -0.45]), color='r')
axp1nd.plot(np.array([0.45, 0.45]), np.array([0.8, -0.8]), color='r')
axp1nd.plot(np.array([0.30, 0.30]), np.array([0.8, -0.8]), color='r')
axp1nd.plot(np.array([0.60, 0.60]), np.array([0.8, -0.8]), color='r')
axp1nd.text(0.8, 0.05, 'HM', color='w')
axp1nd.text(0.8, 0.50, 'HU', color='w')
axp1nd.text(0.8, -0.40, 'HL', color='w')
axp1nd.text(0.35, 0.6, 'VM', color='w')
axp1nd.text(0.24, 0.6, 'VI', color='w')
axp1nd.text(0.50, 0.6, 'VO', color='w')
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.xlim([0.1, 0.9])
plt.ylim([-0.8, 0.8])
cbar_dens_p1nd.set_label(r'$n_e\;(m^{-3})$')
axp1nd.set_aspect('equal')

figp_2 = plt.figure()
axp2pd = figp_2.add_subplot(1, 2, 1)
axp2pd.set_title('Phase 2: S2023+delta')
pmap_2pd = Plasma2pd.plot(axp2pd)
cbar_dens_p2pd = plt.colorbar(pmap_2pd)
axp2pd.plot(np.array([0.9, 0.1]), np.array([0.0, 0.0]), color='r')
axp2pd.plot(np.array([0.9, 0.1]), np.array([0.45, 0.45]), color='r')
axp2pd.plot(np.array([0.9, 0.1]), np.array([-0.45, -0.45]), color='r')
axp2pd.plot(np.array([0.45, 0.45]), np.array([0.8, -0.8]), color='r')
axp2pd.plot(np.array([0.30, 0.30]), np.array([0.8, -0.8]), color='r')
axp2pd.plot(np.array([0.60, 0.60]), np.array([0.8, -0.8]), color='r')
axp2pd.text(0.8, 0.05, 'HM', color='w')
axp2pd.text(0.8, 0.50, 'HU', color='w')
axp2pd.text(0.8, -0.40, 'HL', color='w')
axp2pd.text(0.35, 0.6, 'VM', color='w')
axp2pd.text(0.24, 0.6, 'VI', color='w')
axp2pd.text(0.50, 0.6, 'VO', color='w')
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.xlim([0.1, 0.9])
plt.ylim([-0.8, 0.8])
cbar_dens_p2pd.set_label(r'$n_e\;(m^{-3})$')
axp2pd.set_aspect('equal')
axp2nd = figp_2.add_subplot(1, 2, 2)
axp2nd.set_title('Phase 2: S2023-delta')
pmap_2nd = Plasma2nd.plot(axp2nd)
cbar_dens_p2nd = plt.colorbar(pmap_2nd)
axp2nd.plot(np.array([0.9, 0.1]), np.array([0.0, 0.0]), color='r')
axp2nd.plot(np.array([0.9, 0.1]), np.array([0.45, 0.45]), color='r')
axp2nd.plot(np.array([0.9, 0.1]), np.array([-0.45, -0.45]), color='r')
axp2nd.plot(np.array([0.45, 0.45]), np.array([0.8, -0.8]), color='r')
axp2nd.plot(np.array([0.30, 0.30]), np.array([0.8, -0.8]), color='r')
axp2nd.plot(np.array([0.60, 0.60]), np.array([0.8, -0.8]), color='r')
axp2nd.text(0.8, 0.05, 'HM', color='w')
axp2nd.text(0.8, 0.50, 'HU', color='w')
axp2nd.text(0.8, -0.40, 'HL', color='w')
axp2nd.text(0.35, 0.6, 'VM', color='w')
axp2nd.text(0.24, 0.6, 'VI', color='w')
axp2nd.text(0.50, 0.6, 'VO', color='w')
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.xlim([0.1, 0.9])
plt.ylim([-0.8, 0.8])
cbar_dens_p2nd.set_label(r'$n_e\;(m^{-3})$')
axp2nd.set_aspect('equal')

# figp_2.savefig('PlasmaShot_S2023.png', dpi=900)

# %% Plots Chords
# x_t, y_t, z_t = plot_torus(1000, 0.45, 0.25, 0.5)
# ax.plot_surface(x_t, y_t, z_t, antialiased=True, color='orange')

fig_1pd = plt.figure()
# ax1pd = fig_1pd.add_subplot(1, 1, 1)
ax1pd = fig_1pd.gca(projection='3d')
plot_info, objects = plot_geo(objects, ax1pd)
chpMW110_1pd, chplMW110_1pd = MW110_1pd.plot_chords(ax1pd)
chpMW280_1pd, chplMW280_1pd = MW280_1pd.plot_chords(ax1pd, color='g')
chpCO2CH2F2_1pd, chplCO2CH2F2_1pd = CO2CH2F2_1pd.plot_chords(ax1pd, color='b')
chpCO2_1pd, chplCO2_1pd = CO2_1pd.plot_chords(ax1pd, color='r')
lgnd1pd = ax1pd.legend([chplMW110_1pd[0], chpMW110_1pd[0], chpMW280_1pd[0],
                        chpCO2CH2F2_1pd[0], chpCO2_1pd[0]],
                        ['Ideal chord', 'Microwave 110 GHz',
                        'Microwave 280 GHz', r'$CO_2 +CH_2F_2$',
                        r'$CO_2$'])
# ax1pd.axes.set_xlim3d(left=-0.05, right=0.05)
# ax1pd.axes.set_ylim3d(bottom=0.10, top=0.20)
# ax1pd.axes.set_zlim3d(bottom=-0.05, top=0.05)
ax1pd.set_xlabel('X (m)')
ax1pd.set_ylabel('Y (m)')
ax1pd.set_zlabel('Z (m)')
plt.show()
# ax.set_aspect('equal')
fig_1nd = plt.figure()
# ax1nd = fig_1nd.add_subplot(1, 1, 1)
ax1nd = fig_1nd.gca(projection='3d')
plot_info, objects = plot_geo(objects, ax1nd)
chpMW110_1nd, chplMW110_1nd = MW110_1nd.plot_chords(ax1nd)
chpMW280_1nd, chplMW280_1nd = MW280_1nd.plot_chords(ax1nd, color='g')
chpCO2CH2F2_1nd, chplCO2CH2F2_1nd = CO2CH2F2_1nd.plot_chords(ax1nd, color='b')
# chpCO2_1nd, chplCO2_1nd = CO2_1nd.plot_chords(ax1nd, color='r')
# lgnd1nd = ax1nd.legend([chplMW110_1nd[0], chpMW110_1nd[0], chpMW280_1nd[0],
#                         chpCO2CH2F2_1nd[0], chpCO2_1nd[0]],
#                        ['Ideal chord', 'Microwave 110 GHz',
#                         'Microwave 280 GHz', r'$CO_2 +CH_2F_2$',
#                         r'$CO_2$'])
# ax1nd.axes.set_xlim3d(left=-0.01, right=0.01)
# ax1nd.axes.set_ylim3d(bottom=0.15, top=0.17)
# ax1nd.axes.set_zlim3d(bottom=-0.01, top=0.01)
ax1nd.set_xlabel('X (m)')
ax1nd.set_ylabel('Y (m)')
ax1nd.set_zlabel('Z (m)')
plt.show()


fig_2pd = plt.figure()
# ax2pd = fig_2pd.add_subplot(1, 1, 1)
ax2pd = fig_2pd.gca(projection='3d')
plot_info, objects = plot_geo(objects, ax2pd)
chpMW110_2pd, chplMW110_2pd = MW110_2pd.plot_chords(ax2pd)
chpMW280_2pd, chplMW280_2pd = MW280_2pd.plot_chords(ax2pd, color='g')
chpCO2CH2F2_2pd, chplCO2CH2F2_2pd = CO2CH2F2_2pd.plot_chords(ax2pd, color='b')
# chpCO2_2pd, chplCO2_2pd = CO2_2pd.plot_chords(ax2pd, color='r')
# lgnd2pd = ax2pd.legend([chplMW110_2pd[0], chpMW110_2pd[0], chpMW280_2pd[0],
#                         chpCO2CH2F2_2pd[0], chpCO2_2pd[0]],
#                        ['Ideal chord', 'Microwave 110 GHz',
#                         'Microwave 280 GHz', r'$CO_2 +CH_2F_2$',
#                         r'$CO_2$'])
# ax2pd.axes.set_xlim3d(left=-0.01, right=0.01)
# ax2pd.axes.set_ylim3d(bottom=0.15, top=0.17)
# ax2pd.axes.set_zlim3d(bottom=-0.01, top=0.01)
ax2pd.set_xlabel('X (m)')
ax2pd.set_ylabel('Y (m)')
ax2pd.set_zlabel('Z (m)')
plt.show()
# ax.set_aspect('equal')
fig_2nd = plt.figure()
# ax2nd = fig_2nd.add_subplot(1, 1, 1)
ax2nd = fig_2nd.gca(projection='3d')
plot_info, objects = plot_geo(objects, ax2nd)
chpMW110_2nd, chplMW110_2nd = MW110_2nd.plot_chords(ax2nd)
chpMW280_2nd, chplMW280_2nd = MW280_2nd.plot_chords(ax2nd, color='g')
chpCO2CH2F2_2nd, chplCO2CH2F2_2nd = CO2CH2F2_2nd.plot_chords(ax2nd, color='b')
# chpCO2_2nd, chplCO2_2nd = CO2_2nd.plot_chords(ax2nd, color='r')
# lgnd2nd = ax2nd.legend([chplMW110_2nd[0], chpMW110_2nd[0], chpMW280_2nd[0],
#                         chpCO2CH2F2_2nd[0], chpCO2_2nd[0]],
#                        ['Ideal chord', 'Microwave 110 GHz',
#                         'Microwave 280 GHz', r'$CO_2 +CH_2F_2$',
#                         r'$CO_2$'])
# ax2nd.axes.set_xlim3d(left=-0.01, right=0.01)
# ax2nd.axes.set_ylim3d(bottom=0.15, top=0.17)
# ax2nd.axes.set_zlim3d(bottom=-0.01, top=0.01)
ax2nd.set_xlabel('X (m)')
ax2nd.set_ylabel('Y (m)')
ax2nd.set_zlabel('Z (m)')
plt.show()


# %% Print

# print('At phase 2 CO2 chord deviation is:', data_chord_CO22['theta'])
# print('At phase 2 MW of 2.75 mm chord deviation is:', data_chord_MW2['theta'])



# %% Signals

Signal_CO2 = Signal(0.2, 0.22, 5e8)
Signal_HeNe = Signal(0.2, 0.22, 5e8)
Signal_CO2CH2F2 = Signal(0.0, 0.5, 5e7)
Signal_MW110 = Signal(0.0, 0.5, 5e7)
Signal_MW280 = Signal(0.0, 0.5, 5e7)
Signal_MW400 = Signal(0.0, 0.5, 5e7)


# Plateau
# Pla, t_p = plateau(0, 0.1, 0.1, 0.5, 1e5)
Pla, t_p = plateau_exp(t_ini=0.0, t_fin=0.5, C=40, N_t=10000)


# Create signals
Chords_name = np.array(['HM','VM'])
# Chords_name = np.array(['All'])
# Chords = 'All'
Signal_CO2.make(CO2_1pd, Chords_name, Pla, t_p, vibrations)
Signal_HeNe.make(HeNe_1pd, Chords_name, Pla, t_p, vibrations)
Signal_CO2CH2F2.make(CO2CH2F2_1pd, Chords_name, Pla, t_p, vibrations)
Signal_MW110.make(MW110_1pd, Chords_name, Pla, t_p, vibrations)
Signal_MW280.make(MW280_1pd, Chords_name, Pla, t_p, vibrations)
Signal_MW400.make(MW400_1pd, Chords_name, Pla, t_p, vibrations)

# %% Comparator

data_signal_CO2 = Signal_CO2('HM')
data_signal_HeNe = Signal_HeNe('HM')
data_signal_CO2CH2F2 = Signal_CO2CH2F2('HM')
data_signal_MW110 = Signal_MW110('HM')
data_signal_MW280 = Signal_MW280('HM')
data_signal_MW400 = Signal_MW400('HM')


data_interferometer_CO2_HM = Phase_Comparator(data_signal_CO2, CO2_1pd, 1000)
data_interferometer_HeNe_HM = Phase_Comparator(data_signal_HeNe, HeNe_1pd,
                                               1000)
data_interferometer_CO2CH2F2_HM = Phase_Comparator(data_signal_CO2CH2F2,
                                                   CO2CH2F2_1pd, 200)
data_interferometer_MW110_HM = Phase_Comparator(data_signal_MW110, MW110_1pd,
                                                200)
data_interferometer_MW280_HM = Phase_Comparator(data_signal_MW280, MW280_1pd,
                                                200)
data_interferometer_MW400_HM = Phase_Comparator(data_signal_MW400, MW400_1pd,
                                                200)

t_mes_CO2 = data_interferometer_CO2_HM['t_mes']
t_theo_CO2 = data_interferometer_CO2_HM['t_theo']
t_mes_CO2CH2F2 = data_interferometer_CO2CH2F2_HM['t_mes']
t_theo_CO2CH2F2 = data_interferometer_CO2CH2F2_HM['t_theo']
t_mes_MW110 = data_interferometer_MW110_HM['t_mes']
t_theo_MW110 = data_interferometer_MW110_HM['t_theo']
t_mes_MW280 = data_interferometer_MW280_HM['t_mes']
t_theo_MW280 = data_interferometer_MW280_HM['t_theo']
t_mes_MW400 = data_interferometer_MW400_HM['t_mes']
t_theo_MW400 = data_interferometer_MW400_HM['t_theo']
Phase_CO2 = np.multiply(data_interferometer_CO2_HM['Phase_C_mes'], (-1))
Phase_HeNe = np.multiply(data_interferometer_HeNe_HM['Phase_C_mes'], (-1))
Phase_CO2CH2F2 = np.multiply(data_interferometer_CO2CH2F2_HM['Phase_C_mes'],
                             (-1))
Phase_MW110 = np.multiply(data_interferometer_MW110_HM['Phase_C_mes'], (-1))
Phase_MW280 = np.multiply(data_interferometer_MW280_HM['Phase_C_mes'], (-1))
Phase_MW400 = np.multiply(data_interferometer_MW400_HM['Phase_C_mes'], (-1))


n_e_lin_CO2 = data_interferometer_CO2_HM['n_e_theo']
n_e_lin_CO2CH2F2 = data_interferometer_CO2CH2F2_HM['n_e_theo']
n_e_lin_MW110 = data_interferometer_MW110_HM['n_e_theo']
n_e_lin_MW280 = data_interferometer_MW280_HM['n_e_theo']
n_e_lin_MW400 = data_interferometer_MW400_HM['n_e_theo']

Phase_nkl_CO2 = data_interferometer_CO2_HM['Phase_nkl']
Phase_nkl_CO2CH2F2 = data_interferometer_CO2CH2F2_HM['Phase_nkl']
Phase_nkl_MW110 = data_interferometer_MW110_HM['Phase_nkl']
Phase_nkl_MW280 = data_interferometer_MW280_HM['Phase_nkl']
Phase_nkl_MW400 = data_interferometer_MW400_HM['Phase_nkl']


n_e_mes_CO2CH2F2 = data_interferometer_CO2CH2F2_HM['n_e_mes']
n_e_mes_MW110 = data_interferometer_MW110_HM['n_e_mes']
n_e_mes_MW280 = data_interferometer_MW280_HM['n_e_mes']
n_e_mes_MW400 = data_interferometer_MW400_HM['n_e_mes']

FracFringes_CO2 = int(np.subtract(data_signal_CO2['phase_ref'], data_signal_CO2['phase'][0])/(constants.pi))
FracFringes_HeNe = int(np.subtract(data_signal_HeNe['phase_ref'], data_signal_HeNe['phase'][0])/(constants.pi))
FracFringes_CO2CH2F2 = int(np.subtract(data_signal_CO2CH2F2['phase_ref'], data_signal_CO2CH2F2['phase'][0])/(constants.pi))
FracFringes_MW110 = int(np.subtract(data_signal_MW110['phase_ref'], data_signal_MW110['phase'][0])/(np.pi))
FracFringes_MW280 = int(np.subtract(data_signal_MW280['phase_ref'], data_signal_MW280['phase'][0])/(constants.pi))
FracFringes_MW400 = int(np.subtract(data_signal_MW400['phase_ref'], data_signal_MW400['phase'][0])/(constants.pi))
# %% Comparator plots

# Phase plot

plt.figure()
plt.plot(t_mes_CO2, Phase_CO2, label='CO2')
plt.plot(t_mes_CO2, Phase_HeNe, label='HeNe')
plt.plot(t_theo_CO2, np.subtract(Phase_nkl_CO2[0], Phase_nkl_CO2),
         label='Theoretical')
plt.xlabel(r'$t (s)$')
plt.ylabel('Phase (rad)')
plt.legend()
plt.xlim([np.amin(t_mes_CO2), np.amax(t_mes_CO2)])
# plt.ylim([0, np.amax(Phase_CO2)])
plt.show()

plt.figure()
plt.plot(t_mes_CO2, np.subtract(Phase_CO2, np.multiply(CO2_1pd.k/HeNe_1pd.k,Phase_HeNe)), label='CO2')
# plt.plot(t_mes_CO2, Phase_HeNe, label='HeNe')
plt.plot(t_theo_CO2, np.subtract(Phase_nkl_CO2[0], Phase_nkl_CO2),
         label='Theoretical')
plt.xlabel(r'$t (s)$')
plt.ylabel('Phase (rad)')
plt.legend()
plt.xlim([np.amin(t_mes_CO2), np.amax(t_mes_CO2)])
# plt.ylim([0, np.amax(Phase_CO2)])
plt.show()

plt.figure()
plt.plot(t_mes_CO2CH2F2, np.add(constants.pi*FracFringes_CO2CH2F2, Phase_CO2CH2F2),
         label='CO2+CH2F2')
# plt.plot(t_mes_CO2CH2F2, np.subtract(data_interferometer_CO2CH2F2_HM['Phase_nkl_ds'][0],data_interferometer_CO2CH2F2_HM['Phase_nkl_ds']),
#          label='CO2+CH2F2 theo')
# plt.plot(t_mes_CO2CH2F2, Phase_MW, label = 'MW')
plt.plot(t_theo_CO2CH2F2, np.subtract(Phase_nkl_CO2CH2F2[0],Phase_nkl_CO2CH2F2), label = 'Theoretical')
plt.xlabel(r'$t (s)$')
plt.ylabel('Phase (rad)')
plt.legend()
plt.xlim([np.amin(t_mes_CO2CH2F2), np.amax(t_mes_CO2CH2F2)])
# plt.ylim([0, np.amax(Phase_CO2CH2F2)])
plt.show()

plt.figure()
# plt.plot(t_mes_CO2CH2F2, Phase_CO2CH2F2, label = 'CO2+CH2F2')
plt.plot(t_mes_MW110, np.add(constants.pi*FracFringes_MW110, Phase_MW110),
         label = '110 GHz')
plt.plot(t_mes_MW280, np.add(constants.pi*FracFringes_MW280, Phase_MW280),
         label = '280 GHz')
plt.plot(t_mes_MW400, np.add(constants.pi*FracFringes_MW400, Phase_MW400),
         label = '400 GHz')
plt.plot(t_theo_MW110, np.subtract(Phase_nkl_MW110[0],Phase_nkl_MW110),
         label = 'Theoretical 280 GHz')
plt.xlabel(r'$t (s)$')
plt.ylabel('Phase (rad)')
plt.legend()
plt.xlim([np.amin(t_mes_MW110), np.amax(t_mes_MW110)])
# plt.ylim([0, np.amax(Phase_MW)])
plt.show()


# Vibration cancel
n_e_NoVib = np.multiply(np.power((1/(2*CO2_1pd.n_c)-1/(2*HeNe_1pd.n_c)),-1)*1/CO2_1pd.k, np.subtract(Phase_CO2, np.multiply(CO2_1pd.k/HeNe_1pd.k,Phase_HeNe)))
n_e_CO2CH2F2_corrected = 2*np.multiply(CO2CH2F2_1pd.n_c/CO2CH2F2_1pd.k, np.add(constants.pi*FracFringes_CO2CH2F2, Phase_CO2CH2F2))
n_e_MW110_corrected = 2*np.multiply(MW110_1pd.n_c/MW110_1pd.k, np.add(constants.pi*(FracFringes_MW110), Phase_MW110))
n_e_MW400_corrected = 2*np.multiply(MW400_1pd.n_c/MW400_1pd.k, np.add(constants.pi*(FracFringes_MW400), Phase_MW400))
n_e_MW280_corrected = 2*np.multiply(MW280_1pd.n_c/MW280_1pd.k, np.add(constants.pi*(FracFringes_MW280), Phase_MW280))

# Line integrated density plots

plt.figure()
plt.plot(t_mes_CO2, n_e_NoVib, label=r'Two-Color $CO_2-HeNe$')
plt.plot(t_theo_CO2, n_e_lin_CO2, label='Theoretical')
plt.xlabel(r'$t (s)$')
plt.ylabel(r'$n_e (m^{-2})$')
plt.legend()
plt.xlim([np.amin(t_mes_CO2), np.amax(t_mes_CO2)])
plt.ylim([0.9*np.amax(n_e_lin_CO2[(t_theo_CO2 >= np.amin(t_mes_CO2))*(t_theo_CO2 <= np.amax(t_mes_CO2))]), 1.1*np.amax(n_e_lin_CO2[(t_theo_CO2 >= np.amin(t_mes_CO2))*(t_theo_CO2 <= np.amax(t_mes_CO2))])])
plt.show()

plt.figure()
plt.plot(t_mes_CO2CH2F2, n_e_CO2CH2F2_corrected, label=r'$CO_2+CH_2F_2$')
plt.plot(t_theo_CO2CH2F2, n_e_lin_CO2CH2F2, label='Theoretical')
plt.xlabel(r'$t (s)$')
plt.ylabel(r'$n_e (m^{-2})$')
plt.legend()
plt.xlim([np.amin(t_mes_CO2CH2F2), np.amax(t_mes_CO2CH2F2)])
plt.ylim([0.0*np.amax(n_e_lin_CO2CH2F2[(t_theo_CO2CH2F2 >= np.amin(t_mes_CO2CH2F2))*(t_theo_CO2CH2F2 <= np.amax(t_mes_CO2CH2F2))]), 1.2*np.amax(n_e_lin_CO2CH2F2[(t_theo_CO2CH2F2 >= np.amin(t_mes_CO2CH2F2))*(t_theo_CO2CH2F2 <= np.amax(t_mes_CO2CH2F2))])])
plt.show()

plt.figure()
plt.plot(t_mes_MW110, n_e_MW110_corrected, label='110 GHz')
plt.plot(t_mes_MW280, n_e_MW280_corrected, label='280 GHz')
plt.plot(t_mes_MW400, n_e_MW400_corrected, label='400 GHz')
plt.plot(t_theo_MW110, n_e_lin_MW110, label='Theoretical')
plt.xlabel(r'$t (s)$')
plt.ylabel(r'$n_e (m^{-2})$')
plt.legend()
plt.xlim([np.amin(t_mes_MW110), np.amax(t_mes_MW110)])
plt.ylim([0.0*np.amax(n_e_lin_MW110[(t_theo_MW110 >= np.amin(t_mes_MW110))*(t_theo_MW110 <= np.amax(t_mes_MW110))]), 1.2*np.amax(n_e_lin_MW110[(t_theo_MW110 >= np.amin(t_mes_MW110))*(t_theo_MW110 <= np.amax(t_mes_MW110))])])
plt.show()

# MW and CO2+CH2F2

# fig_rec = plt.figure()
# plt.plot(t_mes_MW, n_e_MW_corrected, label='MW')
# plt.plot(t_mes_CO2CH2F2, n_e_CO2CH2F2_corrected, label=r'$CO_2+CH_2F_2$')
# plt.plot(t_theo_MW, n_e_lin_MW, label='Theoretical')
# plt.xlabel(r'$t (s)$')
# plt.ylabel(r'$n_e (m^{-3})$')
# fig_rec.suptitle('Phase 2: S2000021B', fontsize="x-large")
# plt.legend()
# plt.xlim([np.amin(t_mes_MW), np.amax(t_mes_MW)])
# plt.ylim([0.0*np.amax(n_e_lin_MW[(t_theo_MW >= np.amin(t_mes_MW))*(t_theo_MW <= np.amax(t_mes_MW))]), 1.2*np.amax(n_e_lin_MW[(t_theo_MW >= np.amin(t_mes_MW))*(t_theo_MW <= np.amax(t_mes_MW))])])
# plt.show()

