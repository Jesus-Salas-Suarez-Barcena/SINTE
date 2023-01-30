# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:06:01 2022

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
from IQDemodulation import *
from Interferometer_Signals import *




def FFT_filter(Signal,f_up,f_low,f_s):
        N_Samples = Signal.size
        Window = sig.get_window('blackman',int(N_Samples))
        Signal_windowed = np.multiply(Signal, Window)
        Signal_FFT = np.zeros(int(N_Samples), dtype=np.complex_)
        Signal_FFT_old = sc.fft.fft(Signal_windowed,n=N_Samples)
        Frequency_array =  sc.fft.fftfreq(int(N_Samples),1/f_s)
        for j,s in enumerate(Frequency_array):
            if (s < (f_up)) and (s > (f_low)):
                    Signal_FFT[j] = Signal_FFT_old[j]

        Signal_filtered = sc.fft.ifft(Signal_FFT)

        return Signal_filtered, Signal_FFT, Signal_FFT_old