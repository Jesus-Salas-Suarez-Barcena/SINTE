# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:18:04 2022

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from numpy import random as rd
import constants
import hl_envelopes_idx as env
import scipy.signal as sc
import scipy.signal as sig
import csv as csv
import time as tm

def time_density_evolution(lmbd,t_mes_ini,t_mes_fin,N_t,Phase_SMART,t_p_ini,t_p_rise,t_p_dec,t_p_fin):

# Parameters


# Plateau Density variation
# Initial time (s) t_p_ini
# Rise time (s) t_p_rise
# Decrease time (s) t_p_dec
# Final time (s) t_p_fin


# Measurement time
# Initial time (s) t_mes_ini
# Final time (s) t_mes_fin


# Phase (1, 2 or 3) Phase_SMART
# Number of Samples N_t

# Angular frequency
    omega = 2*constants.pi*constants.c/(lmbd*1e-6)
    # Wave number
    K = 2*constants.pi/(lmbd*1e-6)
    # Cut-off density
    n_c = (constants.m_e*constants.epsilom_0*omega**2)/(constants.q_e**2)

# Write all inputs into an array
    labels = ['lambda = ','t_mes_ini = ','t_mes_fin = ','N_t = ','Phase_SMART = ','t_p_ini = ','t_p_rise = ','t_p_dec = ','t_p_fin = ','Quick = ']
    inputs = [lmbd,t_mes_ini,t_mes_fin,N_t,Phase_SMART,t_p_ini,t_p_rise,t_p_dec,t_p_fin,0]
    
    Parameters = zip(labels,inputs)
## Check if the parameters have been change to not repeat the process everytime
    
    data_old_inputs = pd.read_csv('SMART_phase_selection.txt',sep='\s+',header=None)
    old_inputs_df = pd.DataFrame(data_old_inputs)  
    old_inputs = old_inputs_df.to_numpy().T
    
    

    Check_repeat = old_inputs == inputs 
    
    if Check_repeat.all() == False:
        
        
    
    
    # Load Astra profiles
    
        data_1 = pd.read_csv('ne_fase1.txt',sep='\s+',header=None)
        data_1 = pd.DataFrame(data_1)
        data_2 = pd.read_csv('ne_fase2.txt',sep='\s+',header=None)
        data_2 = pd.DataFrame(data_2)
        data_3 = pd.read_csv('ne_fase3.txt',sep='\s+',header=None)
        data_3 = pd.DataFrame(data_3)
    
        n_e_1 = np.multiply(1e19,data_1[1])
        r_1 = data_1[0]
        n_e_2 = np.multiply(1e19,data_2[1])
        r_2 = data_2[0]
        n_e_3 = np.multiply(1e19,data_3[1])
        r_3 = data_3[0]
    
    
    
    # Average density
    
        if Phase_SMART == 1:
            n_e_av = np.mean(n_e_1)
        if Phase_SMART == 2:
            n_e_av = np.mean(n_e_2)
        if Phase_SMART == 3:
            n_e_av = np.mean(n_e_3)
    
    
    # Time array 
        t_mes = np.linspace(t_mes_ini,t_mes_fin,num = int(N_t))
    
    
    
    # Plateau time evolution 
    
    
        if Phase_SMART == 1:
            n_e_time = np.zeros((n_e_1.size,int(N_t)))
            for i,s in enumerate(t_mes):
                if t_mes[i] < (t_p_ini + t_p_rise):
                    n_e_time[:,i] = np.multiply(n_e_1[:],t_mes[i]/t_p_rise)
                if t_mes[i] > (t_p_ini + t_p_rise):
                    if t_mes[i] < (t_p_fin - t_p_dec):
                        n_e_time[:,i] = n_e_1[:]
                    if t_mes[i] > (t_p_fin - t_p_dec):
                        n_e_time[:,i] = np.add(np.multiply(n_e_1[:],(-1)*t_mes[i]/t_p_dec),np.multiply(t_p_fin/t_p_dec,n_e_1[:]))  
        if Phase_SMART == 2:
            n_e_time = np.zeros((n_e_2.size,int(N_t)))
            for i,s in enumerate(t_mes):
                if t_mes[i] < (t_p_ini + t_p_rise):
                    n_e_time[:,i] = np.multiply(n_e_2[:],t_mes[i]/t_p_rise)
                if t_mes[i] > (t_p_ini + t_p_rise):
                    if t_mes[i] < (t_p_fin - t_p_dec):
                        n_e_time[:,i] = n_e_2[:]
                    if t_mes[i] > (t_p_fin - t_p_dec):
                        n_e_time[:,i] = np.add(np.multiply(n_e_2[:],(-1)*t_mes[i]/t_p_dec),np.multiply(t_p_fin/t_p_dec,n_e_2[:]))  
        if Phase_SMART == 3:
            n_e_time = np.zeros((n_e_3.size,int(N_t)))
            for i,s in enumerate(t_mes):
                if t_mes[i] < (t_p_ini + t_p_rise):
                    n_e_time[:,i] = np.multiply(n_e_2[:],t_mes[i]/t_p_rise)
                if t_mes[i] > (t_p_ini + t_p_rise):
                    if t_mes[i] < (t_p_fin - t_p_dec):
                        n_e_time[:,i] = n_e_3[:]
                    if t_mes[i] > (t_p_fin - t_p_dec):
                        n_e_time[:,i] = np.add(np.multiply(n_e_3[:],(-1)*t_mes[i]/t_p_dec),np.multiply(t_p_fin/t_p_dec,n_e_3[:]))  
        
        
        # Refractive index
        n_refractive = np.sqrt(1-np.multiply(1.0/n_c,n_e_time))
        
        Phase_nkl = np.zeros(int(N_t))
        
        for i,s in enumerate(t_mes):
            if Phase_SMART == 1:
                Phase_nkl[i] = np.multiply(K,np.trapz(n_refractive[:,i],r_1))
            if Phase_SMART == 2:
                Phase_nkl[i] = np.multiply(K,np.trapz(n_refractive[:,i],r_2))
            if Phase_SMART == 3:
                Phase_nkl[i] = np.multiply(K,np.trapz(n_refractive[:,i],r_3))
                
                
                
                
                
                
        np.save(open('n_e_time.npy','wb'),n_e_time)
        np.save(open('Phase_nkl.npy','wb'),Phase_nkl)



    if Check_repeat.all() == True:  
        n_e_time = np.load(open('n_e_time.npy','rb'))
        Phase_nkl = np.load(open('Phase_nkl.npy','rb'))
        
        
        
    with open('SMART_phase_selection.txt','w') as f:
        for i,s in enumerate(inputs):
            f.write(str(inputs[i]))
            f.write('\n')
        
    
    
            
    return Phase_nkl,n_e_time

def time_density_evolution_quick(lmbd,t_mes_ini,t_mes_fin,N_t,Phase_SMART,t_p_ini,t_p_rise,t_p_dec,t_p_fin):

    # Parameters
    
    
    # Plateau Density variation
    # Initial time (s) t_p_ini
    # Rise time (s) t_p_rise
    # Decrease time (s) t_p_dec
    # Final time (s) t_p_fin
    
    
    # Measurement time
    # Initial time (s) t_mes_ini
    # Final time (s) t_mes_fin
    
    
    # Phase (1, 2 or 3) Phase_SMART
    # Number of Samples N_t
    
    
    # Osillations in density 
    # Amplitud (m-3)
    n_e_os = 1.5e16
    # Frequency
    f_os = 1e5
    
    # Shaping
    # Gaussian
    # Center (m)
    r_os = 0.1 
    # Dispersion
    sigma_os = 0.01

# Angular frequency
    omega = 2*constants.pi*constants.c/(lmbd*1e-6)
    omega_os = 2*constants.pi*f_os
    # Wave number
    K = 2*constants.pi/(lmbd*1e-6)
    # Cut-off density
    n_c = (constants.m_e*constants.epsilom_0*omega**2)/(constants.q_e**2)

# Write all inputs into an array
    labels = ['lambda = ','t_mes_ini = ','t_mes_fin = ','N_t = ','Phase_SMART = ','t_p_ini = ','t_p_rise = ','t_p_dec = ','t_p_fin = ','Quick = ', 'n_e_os = ', 'f_os']
    inputs = [lmbd,t_mes_ini,t_mes_fin,N_t,Phase_SMART,t_p_ini,t_p_rise,t_p_dec,t_p_fin,1,n_e_os,f_os]
    
    Parameters = zip(labels,inputs)
## Check if the parameters have been change to not repeat the process everytime
    
    data_old_inputs = pd.read_csv('SMART_phase_selection.txt',sep='\s+',header=None)
    old_inputs_df = pd.DataFrame(data_old_inputs)  
    old_inputs = old_inputs_df.to_numpy().T
    
    

    Check_repeat = old_inputs == inputs 
    
    if Check_repeat.all() == False:
        print('Generating density time evolution')
        print('\n')
        for i,s in enumerate(labels):
            print(labels[i],inputs[i])
        print('\n')
    # Load Astra profiles
        if Phase_SMART == 1:
            data = pd.read_csv('ne_fase1.txt',sep='\s+',header=None)
            data = pd.DataFrame(data)
        if Phase_SMART == 2:
            data = pd.read_csv('ne_fase2.txt',sep='\s+',header=None)
            data = pd.DataFrame(data)
        if Phase_SMART == 3:
            data = pd.read_csv('ne_fase3.txt',sep='\s+',header=None)
            data = pd.DataFrame(data)
    
        n_e_array = np.multiply(1e19,data[1])
        r_array = data[0]

    
    
    # Time array 
        t_mes = np.linspace(t_mes_ini,t_mes_fin,num = int(N_t))
    
    
    # Plateau time evolution 
    
        # Oscillations profile
        n_e_os_gauss = np.multiply(n_e_os,np.exp(np.multiply((-1),np.power(np.divide(np.subtract(r_array,r_os),sigma_os),2))))
        
        n_e_time = np.zeros(n_e_array.size)
        n_refractive = np.zeros(n_e_array.size) 
        Phase_nkl = np.zeros(int(N_t))
        n_e_lin = np.zeros(int(N_t))
        t_dens = np.zeros(int(N_t))
        t_refr = np.zeros(int(N_t))
        t_phase = np.zeros(int(N_t))
        t_ne_lin = np.zeros(int(N_t))
        progress_old = 0
        print('Progress:')
        print('0 %')
        for i,s in enumerate(t_mes):
            t0 = tm.time()
            # n_e_time[:] = np.add(np.multiply((t_mes[i] < (t_p_ini + t_p_rise)),np.add(np.multiply(n_e_array[:],t_mes[i]/t_p_rise), np.multiply(n_e_os_gauss,np.cos(omega_os*t_mes[i])))),
            #                       np.multiply((t_mes[i] > (t_p_ini + t_p_rise)),np.add(np.multiply((t_mes[i] < (t_p_fin - t_p_dec)),np.add(n_e_array[:], np.multiply(n_e_os_gauss,np.cos(omega_os*t_mes[i])))),
            #                                                                           np.multiply((t_mes[i] > (t_p_fin - t_p_dec)),np.add(np.add(np.multiply(n_e_array[:],(-1)*t_mes[i]/t_p_dec),np.multiply(t_p_fin/t_p_dec,n_e_array[:])), np.multiply(n_e_os_gauss,np.cos(omega_os*t_mes[i])))))))
            
            if t_mes[i] < (t_p_ini + t_p_rise):
                n_e_time[:] = np.add(np.multiply(n_e_array[:],t_mes[i]/t_p_rise), np.multiply(n_e_os_gauss,np.cos(omega_os*t_mes[i])))
            if t_mes[i] > (t_p_ini + t_p_rise):
                if t_mes[i] < (t_p_fin - t_p_dec):
                    n_e_time[:] = np.add(n_e_array[:], np.multiply(n_e_os_gauss,np.cos(omega_os*t_mes[i])))
                if t_mes[i] > (t_p_fin - t_p_dec):
                    n_e_time[:] = np.add(np.add(np.multiply(n_e_array[:],(-1)*t_mes[i]/t_p_dec),np.multiply(t_p_fin/t_p_dec,n_e_array[:])), np.multiply(n_e_os_gauss,np.cos(omega_os*t_mes[i])))             
            t1 = tm.time()
            n_refractive[:] = np.sqrt(1-np.multiply(1.0/n_c,n_e_time))
            t2 = tm.time()
            Phase_nkl[i] = np.multiply(K,np.trapz(n_refractive[:],r_array))
            t3 = tm.time()
            n_e_lin[i] = np.trapz(n_e_time,r_array)
            t4 = tm.time()
            
            progress = i*100/N_t
            
            if  int(progress) ==  int(progress_old + 10):
                print(int(progress),'%')
                progress_old = int(progress)
            
            
            
            t_dens[i] = t1-t0
            t_refr[i] = t2-t1
            t_phase[i] = t3-t2
            t_ne_lin[i] = t4-t3
        
        t_dens_av = np.mean(t_dens)
        t_refr_av = np.mean(t_refr)
        t_phase_av = np.mean(t_phase)
        t_ne_lin_av = np.mean(t_ne_lin)
        
        print('Performance:')
        print('Density evolution: ', t_dens_av*1000, 'ms')
        print('Refractive index: ', t_refr_av*1000, 'ms')
        print('Phase integral: ', t_phase_av*1000, 'ms')
        print('Density integral: ', t_ne_lin_av*1000, 'ms')
        print('\n')
        
        
        # np.save(open('n_e_time.npy','wb'),n_e_time)
        print('Saving density time evolution to files')
        t_save1 = tm.time()
        np.save(open('n_e_lin.npy','wb'),n_e_lin)
        np.save(open('Phase_nkl.npy','wb'),Phase_nkl)
        t_save2 = tm.time()
        t_save = t_save2-t_save1
        print('time = ',t_save*1000,'ms')


    if Check_repeat.all() == True:  
        # n_e_time = np.load(open('n_e_time.npy','rb'))
        print('Loading density time evolution from files')
        print('\n')
        for i,s in enumerate(labels):
            print(labels[i],inputs[i])
        print('\n')    
            
        n_e_lin = np.load(open('n_e_lin.npy','rb'))
        Phase_nkl = np.load(open('Phase_nkl.npy','rb'))
        
        
        
    with open('SMART_phase_selection.txt','w') as f:
        for i,s in enumerate(inputs):
            f.write(str(inputs[i]))
            f.write('\n')
        
    
    
            
    return Phase_nkl,n_e_lin




def time_density_evolution_downsampled(lmbd,n_e_array,r_array,t_p_ini,t_p_rise,t_p_dec,t_p_fin,N_t):

    # Parameters
    
    
    # Plateau Density variation
    # Initial time (s) t_p_ini
    # Rise time (s) t_p_rise
    # Decrease time (s) t_p_dec
    # Final time (s) t_p_fin
       
    
    # Phase (1, 2 or 3) Phase_SMART
    # Number of Samples N_t
    
    
    # Osillations in density 
    # Amplitud (m-3)
    # n_e_os = 1.5e16
    # Frequency
    # f_os = 1e5
    
    # Shaping
    # Gaussian
    # Center (m)
    # r_os = 0.1 
    # Dispersion
    # sigma_os = 0.01
    
    # Angular frequency
    omega = 2*constants.pi*constants.c/(lmbd*1e-6)
    # omega_os = 2*constants.pi*f_os
    # Wave number
    K = 2*constants.pi/(lmbd*1e-6)
    # Cut-off density
    n_c = (constants.m_e*constants.epsilom_0*omega**2)/(constants.q_e**2)
    
    # Time array
    t_mes = np.linspace(t_p_ini,t_p_fin,num = int(N_t))
    
    n_e_lin_max = np.trapz(n_e_array,r_array)

    # n_e_array = np.multiply(1e19,data[1])
    # r_array = data[0]


# Write all inputs into an array
    labels = ['lambda = ','N_t = ','ne_av = ','t_p_ini = ','t_p_rise = ','t_p_dec = ','t_p_fin = ','Quick = ']
    inputs = [lmbd,N_t,n_e_lin_max,t_p_ini,t_p_rise,t_p_dec,t_p_fin,1]
    
    Parameters = zip(labels,inputs)
## Check if the parameters have been change to not repeat the process everytime
    
    data_old_inputs = pd.read_csv('SMART_phase_selection.txt',sep='\s+',header=None)
    old_inputs_df = pd.DataFrame(data_old_inputs)  
    old_inputs = old_inputs_df.to_numpy().T
    
    

    Check_repeat = old_inputs == inputs 
    
    if Check_repeat.all() == False:
        print('Generating density time evolution')
        print('\n')
        for i,s in enumerate(labels):
            print(labels[i],inputs[i])
        print('\n')

    
    
    # Time array 
        t_mes = np.linspace(t_p_ini,t_p_fin,num = int(N_t))
    
    
    # Plateau time evolution 
    
        # Oscillations profile
        # n_e_os_gauss = np.multiply(n_e_os,np.exp(np.multiply((-1),np.power(np.divide(np.subtract(r_array,r_os),sigma_os),2))))
        
        n_e_time = np.zeros(n_e_array.size)
        n_refractive = np.zeros(n_e_array.size) 
        Phase_nkl = np.zeros(int(N_t))
        n_e_lin = np.zeros(int(N_t))
        t_dens = np.zeros(int(N_t))
        t_refr = np.zeros(int(N_t))
        t_phase = np.zeros(int(N_t))
        t_ne_lin = np.zeros(int(N_t))
        progress_old = 0
        print('Progress:')
        print('0 %')
        for i,s in enumerate(t_mes):
            t0 = tm.time()
            # n_e_time[:] = np.add(np.multiply((t_mes[i] < (t_p_ini + t_p_rise)),np.add(np.multiply(n_e_array[:],t_mes[i]/t_p_rise), np.multiply(n_e_os_gauss,np.cos(omega_os*t_mes[i])))),
            #                       np.multiply((t_mes[i] > (t_p_ini + t_p_rise)),np.add(np.multiply((t_mes[i] < (t_p_fin - t_p_dec)),np.add(n_e_array[:], np.multiply(n_e_os_gauss,np.cos(omega_os*t_mes[i])))),
            #                                                                           np.multiply((t_mes[i] > (t_p_fin - t_p_dec)),np.add(np.add(np.multiply(n_e_array[:],(-1)*t_mes[i]/t_p_dec),np.multiply(t_p_fin/t_p_dec,n_e_array[:])), np.multiply(n_e_os_gauss,np.cos(omega_os*t_mes[i])))))))
            
            if t_mes[i] < (t_p_ini + t_p_rise):
                n_e_time[:] = np.multiply(n_e_array[:],t_mes[i]/t_p_rise)
            if t_mes[i] > (t_p_ini + t_p_rise):
                if t_mes[i] < (t_p_fin - t_p_dec):
                    n_e_time[:] = n_e_array[:]
                if t_mes[i] > (t_p_fin - t_p_dec):
                    n_e_time[:] = np.add(np.multiply(n_e_array[:],(-1)*t_mes[i]/t_p_dec),np.multiply(t_p_fin/t_p_dec,n_e_array[:]))            
            t1 = tm.time()
            n_refractive[:] = np.sqrt(1-np.multiply(1.0/n_c,n_e_time))
            t2 = tm.time()
            Phase_nkl[i] = np.multiply(K,np.trapz(n_refractive[:],r_array))
            t3 = tm.time()
            n_e_lin[i] = np.trapz(n_e_time,r_array)
            t4 = tm.time()
            
            progress = i*100/N_t
            
            if  int(progress) ==  int(progress_old + 10):
                print(int(progress),'%')
                progress_old = int(progress)
            
            
            
            t_dens[i] = t1-t0
            t_refr[i] = t2-t1
            t_phase[i] = t3-t2
            t_ne_lin[i] = t4-t3
        
        t_dens_av = np.mean(t_dens)
        t_refr_av = np.mean(t_refr)
        t_phase_av = np.mean(t_phase)
        t_ne_lin_av = np.mean(t_ne_lin)
        
        print('Performance:')
        print('Density evolution: ', t_dens_av*1000, 'ms')
        print('Refractive index: ', t_refr_av*1000, 'ms')
        print('Phase integral: ', t_phase_av*1000, 'ms')
        print('Density integral: ', t_ne_lin_av*1000, 'ms')
        print('\n')
        
        
        # np.save(open('n_e_time.npy','wb'),n_e_time)
        print('Saving density time evolution to files')
        t_save1 = tm.time()
        np.save(open('n_e_lin.npy','wb'),n_e_lin)
        np.save(open('Phase_nkl.npy','wb'),Phase_nkl)
        t_save2 = tm.time()
        t_save = t_save2-t_save1
        print('time = ',t_save*1000,'ms')


    if Check_repeat.all() == True:  
        # n_e_time = np.load(open('n_e_time.npy','rb'))
        print('Loading density time evolution from files')
        print('\n')
        for i,s in enumerate(labels):
            print(labels[i],inputs[i])
        print('\n')    
            
        n_e_lin = np.load(open('n_e_lin.npy','rb'))
        Phase_nkl = np.load(open('Phase_nkl.npy','rb'))
        
        
        
    with open('SMART_phase_selection.txt','w') as f:
        for i,s in enumerate(inputs):
            f.write(str(inputs[i]))
            f.write('\n')
        
    
    plt.figure()
    plt.plot(t_mes,n_e_lin)
    plt.xlabel('t (s)')
    plt.ylabel(r'$n_el\;(m^{-2})$')
            
    return Phase_nkl,n_e_lin,t_mes

def density_oscillation(lmbd, t_mes, r_array, n_os, f_os, r_os, sigma_os):
    
    # Angular frequency
    omega = 2*constants.pi*constants.c/(lmbd*1e-6)
    # Wave number
    K = 2*constants.pi/(lmbd*1e-6)
    # Cut-off density
    n_c = (constants.m_e*constants.epsilom_0*omega**2)/(constants.q_e**2)

    
    # Oscillations profile
    n_e_os_gauss = np.multiply(n_os,np.exp(np.multiply((-1),np.power(np.divide(np.subtract(r_array,r_os),sigma_os),2))))
    # Line integral over profile
    n_e_lin = np.trapz(n_e_os_gauss,r_array) 
    # Oscillation
    n_e_lin_t = np.multiply(n_e_lin,np.cos(np.multiply(2*constants.pi*f_os,t_mes)))
    
    Phase_nkl = np.multiply(K/(2*n_c),n_e_lin_t)
    
    
    return Phase_nkl, n_e_lin_t
