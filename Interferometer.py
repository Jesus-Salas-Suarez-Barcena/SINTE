# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:25:20 2022

@author: jesus
"""

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
import time as tm
# import seaborn as sb



def Phase_Comparator(Signal_data,Source,Sampling_window):
# %% Parameter imputs
    # Samplig frequency
    fs = Signal_data['fs']


    # time
    t_mes = Signal_data['t_mes']
    t_mes_ini = np.amin(t_mes)
    t_mes_fin = np.amax(t_mes)
    # Chord calculation
    N_t = len(t_mes)

    # Plateau density evolution for phase calculation
    Phase_nkl = Signal_data['phase_nkl']
    n_e_lin = Signal_data['n_e_lin']
    t_nkl = Signal_data['t_dens']


    # Frequency_array =  sc.fft.fftfreq(int(Sampling_window),dt)

# %% I/Q Demodulation

    Phase_NC1 = np.zeros(int(N_t))
    FFT_new = np.zeros((int(Sampling_window),int(N_t/Sampling_window)), dtype=np.complex_)
    FFT_old = np.zeros((int(Sampling_window),int(N_t/Sampling_window)), dtype=np.complex_)
    t_IQ1 = np.zeros(int(N_t/Sampling_window))
    progress_old = 0
    for i in range(int(N_t/Sampling_window)):
        t0 = tm.time()
        Phase_NC1[int(Sampling_window*i):int(Sampling_window*(i+1))] = \
            IQDemodulation_Samples1(Signal_data['P'][int(Sampling_window*i):int(Sampling_window*(i+1))].ravel(),
                                   Signal_data['R'][int(Sampling_window*i):int(Sampling_window*(i+1))].ravel(),Source.deltaf,fs)
        t1 = tm.time()
        t_IQ1[i] = t1-t0

# %% Phase Correction
    t_downsampled = np.zeros(int(N_t/Sampling_window))
    Phase_NC1_downsampled = np.zeros(int(N_t/Sampling_window))
    Phase_Corrected_downsampled = np.zeros(int(N_t/Sampling_window))
    Phase_nkl_downsampled = np.zeros(int(N_t/Sampling_window))
    n_jumps = 0
    jump_iter = 0
    jump_array = np.zeros(int(N_t/Sampling_window))
    for j in range(int(N_t/Sampling_window)):
        if j < int(N_t/Sampling_window-1):
            t_downsampled[j] = t_mes[int(Sampling_window*j)]
            Phase_NC1_downsampled[j] = Phase_NC1[int(Sampling_window*j)]
            Phase_nkl_downsampled[j] = Signal_data['phase'][int(Sampling_window*j)]
        if j == int(N_t/Sampling_window-1):
            t_downsampled[j] = t_mes[int(Sampling_window*j-1)]
            Phase_NC1_downsampled[j] = Phase_NC1[int(Sampling_window*j-1)]
            Phase_nkl_downsampled[j] = Signal_data['phase'][int(Sampling_window*j-1)]
        if j > 0:
            jump =  Phase_NC1_downsampled[j]-Phase_NC1_downsampled[j-1]
            jump_c = np.amin(abs(np.array([jump, jump+np.pi, jump-np.pi])))
            if jump_c == abs(jump):
                jump_step = jump
            elif jump_c == abs(jump + np.pi):
                jump_step = jump + np.pi
            elif jump_c == abs(jump - np.pi):
                jump_step = jump - np.pi
            jump_array[j] = abs(jump)
            if  jump_step < np.pi/2:
                Phase_Corrected_downsampled[j] = Phase_Corrected_downsampled[j-1] + jump_step
            elif jump >= np.pi/2:
                Phase_Corrected_downsampled[j] = Phase_Corrected_downsampled[j-1] + jump_step
        elif j==0:
            Phase_Corrected_downsampled[j] = Phase_NC1_downsampled[j]

    t_IQ1_av = np.mean(t_IQ1)


    Phase_theo = np.subtract(abs(Phase_nkl_downsampled[0]), abs(Phase_nkl_downsampled))



# %% Integral line density

    Line_integrated_density_art1 = np.multiply((-2)*Source.n_c/Source.k,Phase_Corrected_downsampled)
    Line_integrated_density = np.multiply((2)*Source.n_c/Source.k,np.subtract(abs(Phase_nkl_downsampled[0]), abs(Phase_nkl_downsampled)))



    ## Mean Error from IQ demodulation

    Error_IQ1 = abs(np.subtract(Line_integrated_density_art1,Line_integrated_density))
    Error_IQ1_av = np.mean(Error_IQ1)

    print('I/Q Demodulation performance results:')
    print('Method 1: Time=', t_IQ1_av*1000, 'ms', 'dn =',"{:e}".format(Error_IQ1_av),'m-3')




    data_interferometer = {
        'n_e_mes': Line_integrated_density_art1,
        'n_e_theo': n_e_lin,
        't_mes': t_downsampled,
        't_theo': t_nkl,
        'Phase_NC_mes': Phase_NC1_downsampled,
        'Phase_C_mes': Phase_Corrected_downsampled,
        'Phase_nkl': Phase_nkl,
        'Phase_nkl_ds': Phase_nkl_downsampled,
        't_calc': t_IQ1_av,
        'Error': Error_IQ1_av,
        'jump_array': jump_array,
    }


    return data_interferometer

    ## Spectrum analysis over electron density

    # n_e_FFT = np.zeros((int(Sampling_window_n_e),int(N_t/Sampling_window_n_e)), dtype=np.complex_)
    # t_FFT_n_e = np.zeros(int(N_t/Sampling_window_n_e))
    # for i in range(int(N_t/Sampling_window_n_e)):
    #     t_FFT_n_e[i] = t_mes[int(Sampling_window_n_e*i)]
    #     n_e_FFT[:,i] = sc.fft.fft(Line_integrated_density_art1[int(Sampling_window_n_e*i):int(Sampling_window_n_e*(i+1))])


    # Frequency_array_n_e =  sc.fft.fftfreq(int(Sampling_window_n_e),dt)

    # tt_n_e, ff_n_e = np.meshgrid(t_FFT_n_e, Frequency_array_n_e)

    # fig_n_e = plt.figure()
    # ax_n_e = plt.axes(projection = '3d')
    # ax_n_e.plot_surface(ff_n_e,t_FFT_n_e,n_e_FFT, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # plt.show()

    # plt.figure()
    # plt.imshow(abs(n_e_FFT), cmap='hot', interpolation='nearest')
    # plt.show()

