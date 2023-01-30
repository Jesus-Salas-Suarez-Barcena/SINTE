# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 19:14:20 2022

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
from FFT_filter import *
from Interferometer_Signals import *



def IQDemodulation_Samples1(Signal_P,Signal_R,f,f_s):

    f_bandwidth = 1e4 # Hz
    f_up = f + f_bandwidth
    f_low = f - f_bandwidth

    Signal_P_filtered, Signal_P_FFT, Signal_P_FFT_old = FFT_filter(Signal_P,f_up,f_low,f_s)
    Signal_R_filtered, Signal_R_FFT, Signal_P_FFT_old = FFT_filter(Signal_R,f_up,f_low,f_s)


    PR_filtered = np.multiply(Signal_P_filtered,np.conj(Signal_R_filtered))


    Phase = np.arctan(np.divide(np.imag(PR_filtered),np.real(PR_filtered)))

    return Phase


def IQDemodulation_Samples2(Signal_P,Signal_R,f,f_s):

    N_Samples = Signal_P.size
    f_up = f/100
    f_low = -f/100

    Time = np.linspace(0,N_Samples/f_s,int(N_Samples))

    Signal_Pcos = np.multiply(Signal_P,np.cos(np.multiply(2*constants.pi*f,Time)))
    Signal_Rcos = np.multiply(Signal_R,np.cos(np.multiply(2*constants.pi*f,Time)))
    Signal_Psin = np.multiply(Signal_P,np.sin(np.multiply(2*constants.pi*f,Time)))
    Signal_Rsin = np.multiply(Signal_R,np.sin(np.multiply(2*constants.pi*f,Time)))

    Signal_IP_filtered, Signal_IP_FFT, Signal_IP_FFT_old = FFT_filter(Signal_Psin,f_up,f_low,f_s)
    Signal_IR_filtered, Signal_IR_FFT, Signal_IR_FFT_old = FFT_filter(Signal_Rsin,f_up,f_low,f_s)
    Signal_QP_filtered, Signal_QP_FFT, Signal_QP_FFT_old = FFT_filter(Signal_Pcos,f_up,f_low,f_s)
    Signal_QR_filtered, Signal_QR_FFT, Signal_QR_FFT_old = FFT_filter(Signal_Rcos,f_up,f_low,f_s)

    Phase_P = np.arctan(np.divide(np.real(Signal_IP_filtered),np.real(Signal_QP_filtered)))
    Phase_R = np.arctan(np.divide(np.real(Signal_IR_filtered),np.real(Signal_QR_filtered)))

    Phase = np.subtract(Phase_R,Phase_P)

    return Phase, Signal_IP_FFT, Signal_IP_FFT_old




def IQDemodulation_Samples3(Signal_P,Signal_R,f,f_s):

    N_Samples = Signal_P.size
    f_up = f/100
    f_low = -f/100

    Time = np.linspace(0,N_Samples/f_s,int(N_Samples))

    Signal_Pexp = np.multiply(Signal_P,np.exp(np.multiply((-1j)*2*constants.pi*f,Time)))
    Signal_Rexp = np.multiply(Signal_R,np.exp(np.multiply((-1j)*2*constants.pi*f,Time)))

    Signal_P_filtered, Signal_P_FFT, Signal_P_FFT_old = FFT_filter(Signal_Pexp,f_up,f_low,f_s)
    Signal_R_filtered, Signal_R_FFT, Signal_R_FFT_old = FFT_filter(Signal_Rexp,f_up,f_low,f_s)

    Phase_P = np.arctan(np.divide(np.imag(Signal_P_filtered),np.real(Signal_P_filtered)))
    Phase_R = np.arctan(np.divide(np.imag(Signal_R_filtered),np.real(Signal_R_filtered)))

    Phase = np.subtract(Phase_R,Phase_P)

    return Phase