# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:20:16 2022

@author: jesus
"""
import numpy as np


def plateau(t_ini=0.0, t_rise=0.1, t_dec=0.1, t_fin=0.5, N=10000):

    t = np.linspace(t_ini, t_fin, int(N))
    f = np.zeros(int(N))

    for i, s in enumerate(t):

        if t[i] < (t_ini + t_rise):
            f[i] = t[i]/t_rise
        if t[i] >= (t_ini + t_rise):
            if t[i] < (t_fin - t_dec):
                f[i] = 1
            if t[i] >= (t_fin - t_dec):
                f[i] = t_fin/t_dec-t[i]/t_dec

    return f, t


def plateau_exp(t_ini=0.0, t_fin=0.5, C=40, N_t=10000):

    t_mid = (t_fin-t_ini)/2

    t_dens = np.linspace(t_ini, t_fin, int(N_t))

    dens_ev = np.subtract(1, np.exp(-C*(t_dens-t_ini)))*(t_dens <= t_mid) \
        + np.subtract(1, np.exp(C*(t_dens-t_fin)))*(t_dens > t_mid)

    return dens_ev, t_dens
