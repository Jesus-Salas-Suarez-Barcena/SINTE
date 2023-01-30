# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:37:20 2023

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import pandas as pd
from numpy import random as rd
import constants
import scipy as sc
import scipy.signal as sig
import csv as csv
import time as tm
import xlwt as xl


def rot_matrix(phi, psi, theta):

    r11 = np.cos(psi)*np.cos(theta)
    r12 = np.sin(phi)*np.sin(psi)*np.cos(theta)-np.cos(phi)*np.sin(theta)
    r13 = np.cos(phi)*np.sin(psi)*np.cos(theta)+np.sin(phi)*np.sin(theta)
    r21 = np.cos(psi)*np.sin(theta)
    r22 = np.sin(phi)*np.sin(psi)*np.sin(theta)+np.cos(phi)*np.cos(theta)
    r23 = np.cos(phi)*np.sin(psi)*np.sin(theta)-np.sin(phi)*np.cos(theta)
    r31 = -np.sin(psi)
    r32 = np.sin(phi)*np.cos(psi)
    r33 = np.cos(phi)*np.cos(psi)


    rot = np.array([[r11, r12, r13],
                    [r21, r22, r23],
                    [r31, r32, r33]])

    return rot