# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:36:55 2022

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
from sample_FFT_filter import *

def IQdemodulation(Signal,Time,domega,Sampling_window,f_sampling):
    
    f_cut = domega/(2*constants.pi)/100
    
    Art_sine = np.multiply(1,np.sin(np.multiply(domega,Time)))
    Art_cosine = np.multiply(1,np.cos(np.multiply(domega,Time)))
        
        # Product with artificial signals
    I_NF = np.multiply(Signal,Art_sine)
    Q_NF = np.multiply(Signal,Art_cosine)
    
    ## FFT to filter signals 
    
    
    I_F = sample_FFT_filter(I_NF,Time,Sampling_window,f_cut,f_sampling)
    Q_F = sample_FFT_filter(Q_NF,Time,Sampling_window,f_cut,f_sampling)            
            
    # I/Q Demodulation


    Phase_NC = np.arctan(np.divide(I_F,Q_F))  
    
    plt.figure()
    plt.plot(Time[0:1000],I_NF[0:1000])
    plt.plot(Time[0:1000],Q_NF[0:1000])
    plt.plot(Time[0:1000],Signal[0:1000])
    plt.show()
    
    
    plt.figure()
    plt.plot(Time,I_F)
    plt.plot(Time,Q_F)
    plt.show()


    return Phase_NC        
    
