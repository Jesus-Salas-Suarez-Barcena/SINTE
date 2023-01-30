# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:27:11 2022

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_torus(precision, c, a, b):
    U = np.linspace(0, 2*np.pi, precision)
    V = np.linspace(0, 2*np.pi, precision)
    U, V = np.meshgrid(U, V)
    X = (c+a*np.cos(V))*np.cos(U)
    Y = (c+b*np.cos(V))*np.sin(U)
    Z = a*np.sin(V)
    return X, Y, Z




