# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:46:29 2022

@author: jesus


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from numpy import random as rd
import constants
from time_density_evolution import *
import hl_envelopes_idx as env
import scipy as sc
import scipy.signal as sig
import csv as csv
import time as tm


def Interferometer_Signals_Preamp(t_mes, lmbd, A_P, A_R, A_S, deltaf,
                                  deltaf_st, deltaPP, l_P, l_R, l_S,
                                  Phase_array, Phase_interpol, t_interpol,
                                  Phase_ref, vibrations):
    """


    Parameters
    ----------
    t_mes : array
        Time array (s) that samples the signal.
    lmbd : float
        Wavelength (um) of the light source.
    A_P : float
        Amplitud of the probing signal.
    A_R : TYPE
        Amplitud of the reference signal.
    A_S : TYPE
        Amplitud of the source signal.
    deltaf : TYPE
        Intermediate frequency (Hz).
    deltaf_st : TYPE
        Frequency stability (Hz).
    deltaPP : TYPE
        Power stability (%).
    l_P : TYPE
        Length of probing path.
    l_R : TYPE
        Length of reference path.
    l_S : TYPE
        Length of source path.
    Phase_array : TYPE
        Phase to insert on the signal.
    Phase_interpol : TYPE
        Phase with less samples to insert on the signal.
    t_interpol : TYPE
        Time array of the previous phase.
    Phase_ref : float.
        Phase to use as reference.
    vibrations : Dictionary
        Contains the info on the vibration to insert in the signals.

    Returns
    -------
    P_0 : Array
        Probing signal after preamp.
    R_0 : Array
        Reference signal after preamp.
    Phase_nkl : Array
        Total phase of the P_0 signal.

    """
# %%  Parameters
    # Time window
    t_mes_ini = np.amin(t_mes)
    t_mes_fin = np.amax(t_mes)

    window_ip = np.multiply(t_interpol >= t_mes_ini, t_interpol <= t_mes_fin)
    Phase_interpol_window = Phase_interpol[window_ip]
    t_interpol_window = t_interpol[window_ip]

    # Angular intermediate frequecny
    domega = 2*constants.pi*deltaf
    # Wave number
    k = 2*constants.pi/(lmbd*1e-6)
    # Total number of samples
    N_t = t_mes.size
    # Number of samples of phase to interpolate
    N_interpol = Phase_interpol_window.size
    print('Window size:', N_interpol)
    # Upsampling factor
    N_upsampling = round(N_t/N_interpol)
    print(N_upsampling)
    # Difference
    N_diff = round(N_t-N_upsampling)
    print(N_diff)

    t_mes_ini = t_mes[0]
    t_mes_fin = t_mes[int(N_t-1)]
    dt = (t_mes_fin-t_mes_ini)/N_t


    # Vibrations
    dl_vib = vibrations['dl_vib']
    f_vib = vibrations['f_vib']





# %% Phase from path length

    # Frequency stability
    deltaf_st_M = np.multiply(deltaf_st,rd.rand(int(N_t)))
    deltaf_st_S = np.multiply(deltaf_st,rd.rand(int(N_t)))

    domega_st_M = np.multiply(deltaf_st_M,2*constants.pi)
    domega_st_S = np.multiply(deltaf_st_S,2*constants.pi)


    Phase_P_path = np.add(np.multiply(deltaf_st_M,l_P*2*constants.pi/constants.c), 2*constants.pi/(lmbd*1e-6)*l_P+ 2*constants.pi*deltaf*l_P/constants.c)
    Phase_R_path = np.add(np.multiply(deltaf_st_M,l_R*2*constants.pi/constants.c), 2*constants.pi/(lmbd*1e-6)*l_R+ 2*constants.pi*deltaf*l_R/constants.c)
    Phase_S_path = np.add(np.multiply(deltaf_st_S,l_S*2*constants.pi/constants.c), 2*constants.pi/(lmbd*1e-6)*l_S)

    # Power stability
    P_ps = np.multiply(deltaPP/100,rd.rand(int(N_t)))
    R_ps = np.multiply(deltaPP/100,rd.rand(int(N_t)))
    S_ps = np.multiply(deltaPP/100,rd.rand(int(N_t)))




# %% Signals
    Phi_P = np.zeros(int(N_t))
    Phi_R = np.zeros(int(N_t))
    Phase_upsampled = np.zeros(int(N_t))
    t_signalloop = np.zeros(int(N_t))

    print('Generating signals:')
    print('\n')
    print('Wavelength =',lmbd,'um')
    print('Intermediate frequency =',deltaf*1e-6,'MHz')
    print('Frequecny stability =',deltaf_st*1e-3,'kHz')
    print('Power stability =', deltaPP,'%')
    print('Probing beam path length =',l_P,'m')
    print('Reference beam path length =',l_R,'m')
    print('Source beam path length =',l_S,'m')
    print('Initial time =',t_mes_ini*1000,'ms')
    print('Final time =',t_mes_fin*1000,'ms')
    print('\n')
    print('Progress:')
    print('0 %')
    progress_old = 0

    for i, n_e in enumerate(Phase_interpol_window):
        if i < N_interpol-1:
            t_step = t_mes[N_upsampling*i:N_upsampling*(i+1)]
            m = (Phase_interpol_window[i+1]-Phase_interpol_window[i])/(t_interpol_window[i+1]-t_interpol_window[i])
            n = Phase_interpol_window[i]-m*t_interpol_window[i]
            Phase_upsampled[N_upsampling*i:N_upsampling*(i+1)] = m*t_step+n
        if i == N_interpol-1:
            Phase_upsampled[N_upsampling*i:] = Phase_upsampled[N_upsampling*i-1]

    for j in range(N_t-1):
        t0 = tm.time()
        if j == 0:
            Phi_P[j] = 0
            Phi_R[j] = 0
        if j > 0:
            Phi_P[j] = Phi_P[j-1] + np.multiply(domega+domega_st_M[j]-domega_st_S[j],dt)
            Phi_R[j] = Phi_R[j-1] + np.multiply(domega+domega_st_M[j]-domega_st_S[j],dt)
        t1 = tm.time()
        t_signalloop[j]=t1-t0
        progress = j*100/N_t
        if  int(progress) ==  int(progress_old + 10):
                print(int(progress), '%')
                progress_old = int(progress)
    t_signalloop_av = np.mean(t_signalloop)
    Phase_vib = np.multiply(dl_vib*k,
                            np.sin(np.multiply(2*constants.pi*f_vib, t_mes)))
    print('100 %')
    print('\n')
    print('Time for every loop', t_signalloop_av*1000,'ms')
    print('\n')
    Phase_nkl = np.add(np.add(Phase_upsampled, Phase_array),Phase_vib)
    P_0 = np.multiply(A_P*A_S/2,np.cos(np.add(Phi_P,np.add(Phase_nkl,np.subtract(Phase_P_path,Phase_S_path)))))
    R_0 = np.multiply(A_R*A_S/2,np.cos(np.add(Phi_R,np.add(Phase_ref,np.subtract(Phase_R_path,Phase_S_path)))))
    print('Phase P ini:', Phase_nkl[0])
    print('Phase R ini:', Phase_ref)

    if deltaPP > 0:
        P_0 = np.multiply(np.multiply(P_0, np.add(P_ps, 1)), np.add(S_ps, 1))
        R_0 = np.multiply(np.multiply(R_0, np.add(R_ps, 1)), np.add(S_ps, 1))

# %% Plots
    N_ds = int(1e5)
    N_div = int(N_t/N_ds)
    Phase_nkl_downsampled = Phase_nkl[N_div*np.array(range(N_ds))]
    t_ds = t_mes[N_div*np.array(range(N_ds))]
    # Phase interpol
    plt.figure()
    plt.plot(t_interpol, np.subtract(Phase_ref, Phase_interpol), label='Phase')
    plt.plot(t_ds, np.subtract(Phase_ref, Phase_nkl_downsampled),
             label='Phase upsampled')
    plt.xlabel('t (s)')
    plt.ylabel('Phase (rad)')
    plt.xlim([np.amin(t_mes), np.amax(t_mes)])
    plt.legend()
    plt.show()

    return P_0, R_0, Phase_nkl
