# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:05:54 2022

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from numpy import random as rd
import constants
import time_density_evolution as tde
import hl_envelopes_idx as env
import scipy as sc
import scipy.signal as sig
import csv as csv

def sample_FFT_filter(Signal,Time,Sampling_window,f_cut,f_sampling):
    N_t = Time.size
    dt = 1/f_sampling
    
    Signal_FFT = np.zeros((int(N_t/Sampling_window),int(Sampling_window)))

   
    
    # Signal fractioned in sampling windows for FFT
    for i in range(int(N_t/Sampling_window)):
        if i > 0:
            if i < int(N_t/Sampling_window-1):
                Signal_FFT[i,:] = sc.fft(Signal[(int(Sampling_window*i)):(int(Sampling_window*(i+1)))])    
        if i == 0:
            Signal_FFT[i,:] = sc.fft(Signal[0:(int(Sampling_window*(i+1)))])

    
    Frequency_array =  sc.fft.fftfreq(int(Sampling_window),dt)
    plt.figure()
    plt.plot(Frequency_array,abs(Signal_FFT[int(N_t/Sampling_window/2),:]))
    # plt.imshow(Signal_FFT,interpolation='none',cmap=plt.cm.jet,origin='lower')
    for j,s in enumerate(Frequency_array):
        s_a = abs(s)
        if s_a > f_cut:
            if (j > 0) and (j < int(Sampling_window-0)):
                Signal_FFT[:,j] = 0
        
    # Inverse FFT
    Signal_filtered = np.zeros(int(N_t))
    for i in range(int(N_t/Sampling_window)):
        if i > 0:
            if i < int(N_t/Sampling_window-1):
                Signal_filtered[(int(Sampling_window*i)):(int(Sampling_window*(i+1)))] = sc.ifft(Signal_FFT[i,:] )    
        if i == 0:
            Signal_filtered[0:(int(Sampling_window*(i+1)))] = sc.ifft(Signal_FFT[i,:] )
    
    # plt.imshow(Signal_FFT,interpolation='none',cmap=plt.cm.jet,origin='lower')
    plt.plot(Frequency_array,abs(Signal_FFT[int(N_t/Sampling_window/2),:]))
    plt.title('Spectral plot')
    plt.xlabel('f (Hz)')
    plt.ylabel('Magnitude')
    plt.show()
    print(Signal_FFT[int(N_t/Sampling_window/2),0:3])
    
    return Signal_filtered