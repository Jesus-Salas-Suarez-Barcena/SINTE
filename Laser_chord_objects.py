# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:57:58 2022

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import pandas as pd
from numpy import random as rd
import constants
# from Chord_path import *
from Plateau import *
from Interferometer_Signals import *
import scipy as sc
import scipy.signal as sig
from Chord_path3D import *
from geometries import *

# %% Plasma object
class Plasma:
    def __init__(self, file_R, file_Z, file_ne, file_Te, file_Br, file_Bz,
                 file_q, Bt, R0, density_fluctuations, objects):
        print('Loading RZ distributions... \n')
    ## Load R Z distributions

        f = open(file_ne,'r') # 1e19 m-3
        self.n_e = np.genfromtxt(f)
        f.close()

        f = open(file_Te,'r') #  keV
        self.T_e = np.genfromtxt(f)
        f.close()

        f = open(file_Br,'r') # T
        self.Br = np.genfromtxt(f)
        f.close()

        f = open(file_Bz,'r') # T
        self.Bz = np.genfromtxt(f)
        f.close()

        f = open(file_R,'r')
        self.R = np.genfromtxt(f)
        f.close()

        f = open(file_Z,'r')
        self.Z = np.genfromtxt(f)
        f.close()

        f = open(file_q, 'r')
        self.q = np.genfromtxt(f)

        self.Bt = Bt
        self.R0 = R0

        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)

        # Alfvén frequency but for reference only
        RR, ZZ = np.meshgrid(self.R, self.Z)
        self.Bt_RZ = np.divide(Bt*R0, RR)
        self.B_total = np.array([self.Br, self.Bt_RZ, self.Bz])
        self.B_norm = np.linalg.norm(self.B_total, axis=0)
        self.v_a = np.divide(self.B_norm,
                             np.sqrt(constants.mu_0\
                                     *(constants.m_i + constants.m_e)*\
                                         self.n_e*1e19))*(self.n_e > 0)
        self.v_a[np.isnan(self.v_a)] = 0
        self.f_a = self.v_a/(4*np.pi*self.q*R0)*(self.n_e > 0)
        self.f_a[np.isnan(self.f_a)] = 0

        # Density flutuatuins
        self.f_os = density_fluctuations['f_os']
        self.n_os = density_fluctuations['n_os']
        self.r_os = density_fluctuations['r_os']
        self.z_os = density_fluctuations['z_os']
        self.shape_dist = density_fluctuations['Shape'][0]
        self.shape_param = density_fluctuations['Shape'][1:]
        self.type = density_fluctuations['Type']


        # Load the geometrical model
        self.obj = geo_model(objects)

        self.tag = np.array([])
        self.tag_tri = np.array([])
        xx = np.array([])
        yy = np.array([])
        zz = np.array([])
        for i, obj in enumerate(self.obj):
            xx = np.concatenate((xx, obj['rr'][0, :]))
            yy = np.concatenate((yy, obj['rr'][1, :]))
            zz = np.concatenate((zz, obj['rr'][2, :]))
            self.tag = np.concatenate((self.tag, obj['tag']))
            self.tag_tri = np.concatenate((self.tag_tri, obj['tag_tri']))
            if i == 0:
                rr_tri = obj['rr_tri']
            else:
                rr_tri = np.concatenate((rr_tri, obj['rr_tri']),
                                        axis=1)
        self.rr_tri = rr_tri
        self.rr = np.array([xx, yy, zz])






    def plot(self,ax):
        """


        Parameters
        ----------
        ax : subplot where to plot the plasma
            Plots a color map of the plasma electron density (n_e (m-3)).

        Returns
        -------
        pmap : returns object of plot.
            DESCRIPTION.

        """
        RR, ZZ = np.meshgrid(self.R, self.Z)
        pmap = ax.pcolormesh(RR, ZZ, self.n_e*1e19,shading = 'auto',)
        return pmap

# %% Chord object

class Chord:

    def __init__(self, name: str = None, r_ini: np.ndarray = None,
                 k_dir: np.ndarray = None, s_max: float = 2.0,
                 l_probe: float = 0.0, l_refer: float = 0.0,
                 l_source: float = 0.0):
        """
        Elemento de cuerda.

        Parameters
        ----------
        name : str, optional
            Name of the chord. The default is None.
        r_ini : np.ndarray, optional
            Point where the chord starts. The default is None.
        r_fin : np.ndarray, optional
            Point where the chord ends. The default is None.
        l_probe : float, optional
            Length of the probing path. The default is 0.0.
        l_refer : float, optional
            Length of the reference path. The default is 0.0.
        l_source : float, optional
            Length of the Source path. The default is 0.0.

        Returns
        -------
        None.

        """
        self.r_ini = r_ini
        self.k_dir = k_dir
        self.l_probe = l_probe
        self.l_refer = l_refer
        self.l_source = l_source
        self.s_max = s_max
        self.name = name

# %% Source object

class Source:
    N_cells = 1e3
    Computed = False
    def __init__(self,lmbd,deltaf,deltaf_st,deltaPP):
        self.lmbd = lmbd
        self.deltaf = deltaf
        self.deltaf_st = deltaf_st
        self.deltaPP = deltaPP
        self.k = 2*constants.pi/(lmbd*1e-6)
        self.omega =  2*constants.pi*constants.c/(lmbd*1e-6)
        self.n_c = (constants.m_e*constants.epsilom_0*self.omega**2)/(constants.q_e**2)

    def __call__(self, chordName):
        return self._getChordData(chordName)

    def _getChordData(self, chordName: str):
        """
        Get a given Chord of the laser

        Parameters
        ----------
        chordName : str
            Name of the LOS to return.

        Returns
        -------
        LOS : dict
            Contain data from the Chord
        """
        flag = self.name_chords == chordName
        if flag.sum() == 0:
            raise Exception('Chord not found')
        data = {
            'n_e': self.n_e[..., flag],
            's': self.s[..., flag],
            'x': self.x[..., flag],
            'y': self.y[..., flag],
            'z': self.z[..., flag],
            'P_loss': self.P_loss[flag],
            'theta': self.theta[flag],
            'Array_length': self.array_length[flag],
            'n_fluc': self.n_fluc[..., flag],
            'l_probe': self.l_probe[flag],
            'l_refer': self.l_refer[flag],
            'l_source': self.l_source[flag],
            'n_e_l': self.n_e_l[flag],
            'lmbd': self.lmbd,
        }
        return data


    def compute(self, Plasma, Chords):
        """
        Computes a ray optics simulation with the parameters given.

        Parameters
        ----------
        Plasma : Object
            _Contains al the info on the plasma parameters.
        Chords : Object
            Geometry of the lines of sight of the interferometer.

        Returns
        -------
        None.

        """
        self.N_chords = len(Chords)
        n_e_chords = np.zeros((int(self.N_cells),self.N_chords))
        n_fluc_chords = np.zeros((int(self.N_cells),self.N_chords))
        s_chords = np.zeros((int(self.N_cells) ,self.N_chords))
        x_chords = np.zeros((int(self.N_cells), self.N_chords))
        y_chords = np.zeros((int(self.N_cells), self.N_chords))
        z_chords = np.zeros((int(self.N_cells), self.N_chords))
        x_s_chords = np.zeros((int(self.N_cells), self.N_chords))
        y_s_chords = np.zeros((int(self.N_cells), self.N_chords))
        z_s_chords = np.zeros((int(self.N_cells), self.N_chords))
        q_chords = np.zeros((int(self.N_cells),self.N_chords))
        B_norm_chords = np.zeros((int(self.N_cells),self.N_chords))
        v_a_chords = np.zeros((int(self.N_cells),self.N_chords))
        f_a_chords = np.zeros((int(self.N_cells),self.N_chords))
        touch_r_chords = np.zeros((int(self.N_cells),self.N_chords), str)
        touch_s_chords = np.zeros((int(self.N_cells),self.N_chords), str)
        P_loss_chords = np.array([])
        theta_chords = np.array([])
        array_length = np.array([])
        l_probe = np.array([])
        l_refer = np.array([])
        l_source = np.array([])
        r_x_ini = np.array([])
        r_y_ini = np.array([])
        r_z_ini = np.array([])
        # r_x_fin = np.array([])
        # r_y_fin = np.array([])
        # r_z_fin = np.array([])
        n_e_l = np.array([])
        self.name_chords = []

        for i,chord in enumerate(Chords):
            data_chord = Chord_path3D(Plasma, chord, self)
            n_e_array = data_chord['n_e']
            s_array = data_chord['s']
            r_array = data_chord['r']
            r_s_array = data_chord['r_s']
            P_loss = data_chord['P_loss']
            theta = data_chord['theta']
            n_fluc_array = data_chord['n_fluc']
            q_array = data_chord['q']
            B_norm_array = data_chord['B_norm']
            v_a_array = data_chord['v_a']
            f_a_array = data_chord['f_a']
            touch_r_array = data_chord['touch_r']
            touch_s_array = data_chord['touch_s']
            n_e_l_step = np.trapz(n_e_array, s_array)
            N_step = int(n_e_array.size)

            n_e_chords[0:N_step, i] = n_e_array
            n_fluc_chords[0:N_step, i] = n_fluc_array
            s_chords[0:N_step, i] = s_array
            x_chords[0:N_step, i] = r_array[:, 0]
            y_chords[0:N_step, i] = r_array[:, 1]
            z_chords[0:N_step, i] = r_array[:, 2]
            x_s_chords[0:N_step, i] = r_s_array[:, 0]
            y_s_chords[0:N_step, i] = r_s_array[:, 1]
            z_s_chords[0:N_step, i] = r_s_array[:, 2]
            q_chords[0:N_step, i] = q_array
            B_norm_chords[0:N_step, i] = B_norm_array
            v_a_chords[0:N_step, i] = v_a_array
            f_a_chords[0:N_step, i] = f_a_array
            touch_r_chords[0:N_step, i] = touch_r_array
            touch_s_chords[0:N_step, i] = touch_s_array

            P_loss_chords = np.concatenate((P_loss_chords, np.array([P_loss])))
            theta_chords = np.concatenate((theta_chords, np.array([theta])))
            array_length = np.concatenate((array_length, np.array([N_step])))
            l_probe = np.concatenate((l_probe, np.array([chord.l_probe])))
            l_refer = np.concatenate((l_refer, np.array([chord.l_refer])))
            l_source = np.concatenate((l_source, np.array([chord.l_source])))
            r_x_ini = np.concatenate((r_x_ini, np.array([chord.r_ini[0]])))
            r_y_ini = np.concatenate((r_y_ini, np.array([chord.r_ini[1]])))
            r_z_ini = np.concatenate((r_z_ini, np.array([chord.r_ini[2]])))
            # r_x_fin = np.concatenate((r_x_fin, np.array([chord.r_fin[0]]))) # Be careful with this (mustr be changed)
            # r_y_fin = np.concatenate((r_y_fin, np.array([chord.r_fin[1]])))
            # r_z_fin = np.concatenate((r_z_fin, np.array([chord.r_fin[2]])))
            n_e_l = np.concatenate((n_e_l, np.array([n_e_l_step])))

            self.name_chords.append(chord.name)
        self.name_chords = np.array(self.name_chords)
        self.n_e = n_e_chords
        self.n_fluc = n_fluc_chords
        self.f_os = Plasma.f_os
        self.type_os = Plasma.type
        self.s = s_chords
        self.x = x_chords
        self.y = y_chords
        self.z = z_chords
        self.x_s = x_s_chords
        self.y_s = y_s_chords
        self.z_s = z_s_chords
        self.P_loss = P_loss_chords
        self.theta = theta_chords
        self.Computed = True
        self.array_length = array_length
        self.l_probe = l_probe
        self.l_refer = l_refer
        self.l_source = l_source
        self.r_x_ini = r_x_ini
        self.r_y_ini = r_y_ini
        self.r_z_ini = r_z_ini
        # self.r_x_fin = r_x_fin
        # self.r_y_fin = r_y_fin
        # self.r_z_fin = r_z_fin
        self.n_e_l = n_e_l
        self.q = q_chords
        self.B_norm = B_norm_chords
        self.v_a = v_a_chords
        self.f_a = f_a_chords
        self.touch_r = touch_r_chords
        self.touch_s = touch_s_chords

    def plot_chords(self, ax, color='tab:orange'):
        """
        Plot the chords in a given figure

        Parameters
        ----------
        ax : Subplot object
            Where to plot the chords.

        Returns
        -------
        chp : Oject plot
            Plot info.

        """

        if self.Computed:
            chp = []
            chpl = []
            for i in range(self.N_chords):
                touch_r_array = self.touch_r[:, i]
                touch_s_array = self.touch_r[:, i]
                x_chord = self.x[:, i]
                y_chord = self.y[:, i]
                z_chord = self.z[:, i]
                x_s_chord = self.x_s[:, i]
                y_s_chord = self.y_s[:, i]
                z_s_chord = self.z_s[:, i]
                chpl.append(ax.plot3D(x_s_chord[x_s_chord != 0],
                                     y_s_chord[x_s_chord != 0],
                                     z_s_chord[x_s_chord != 0],
                                     color='k')[0])
                ax.text(self.r_x_ini[i]-0.05, self.r_y_ini[i]-0.05,
                        self.r_z_ini[i]-0.05, self.name_chords[i],
                        color='k', fontsize=11)
                chp.append(ax.plot3D(x_chord[x_chord != 0],
                                     y_chord[x_chord != 0],
                                     z_chord[x_chord != 0],
                                     color=color)[0])
                ax.scatter(x_chord[touch_r_array=='mirror'],
                           y_chord[touch_r_array=='mirror'],
                           z_chord[touch_r_array=='mirror'],
                           color='g')
                ax.scatter(x_s_chord[touch_r_array=='mirror'],
                           y_s_chord[touch_r_array=='mirror'],
                           z_s_chord[touch_r_array=='mirror'],
                           color='g')
        if not self.Computed:
            raise Exception('Chords not computed')
            chp = 'None'
            chpl = 'None'
        return chp, chpl

# %% Signal object

class Signal:
    def __init__(self, t_ini, t_fin, fs, A_P=1, A_R=1, A_S=1):
        self.t_ini = t_ini
        self.t_fin = t_fin
        self.fs = fs
        self.Ns = int((t_fin-t_ini)*fs)
        self.t = np.linspace(t_ini, t_fin, self.Ns)
        self.A_P = A_P
        self.A_R = A_R
        self.A_S = A_S

    def __call__(self, chordName):
        return self._getChordData(chordName)

    def _getChordData(self, chordName: str):
        """
        Get a given Chord signal of the laser

        Parameters
        ----------
        chordName : str
            Name of the LOS to return.

        Returns
        -------
        LOS : dict
            Contain signal from the Chord
        """
        flag = self.name_chords == chordName
        if flag.sum() == 0:
            raise Exception('Chord not found')
        data = {
            'n_e_lin': self.n_e_lin[..., flag],
            'phase_nkl': self.phase_nkl[..., flag],
            't_dens': self.t_dens[:],
            'P': self.P[..., flag],
            'R': self.R[..., flag],
            'phase': self.phase[..., flag],
            't_mes': self.t,
            'fs': self.fs,
            'phase_ref': self.phase_ref[flag]
        }
        return data

    def make(self, Source, Chords = np.array(['All']), density_evolution=None,
             t_dens=None, vibrations=None):
        """


        Parameters
        ----------
        Source : Object
            Includes info on the chords.
        Chords : Array of str.
            Selects the chords to create their signals. The default is 'All'.
        density_evolution : Array, optional
            Array of the evolution normalized to the peak value. The default is None.
        t_dens : Array. optional
            Time array of the evolution profile. The default is None.
        vibrations : Dicctionary, optional
            Contains the amplitud of the vibration and its frequency. The default is None.

        Returns
        -------
        None.

        """
        self.name_chords = np.array([])
        N_d = density_evolution.size
        if Chords[0] == 'All':
            N_chords = Source.N_chords
            self.n_e_lin = np.zeros((N_d, N_chords))
            self.phase_nkl = np.zeros((N_d, N_chords))
            self.P = np.zeros((self.Ns, N_chords))
            self.R = np.zeros((self.Ns, N_chords))
            self.phase = np.zeros((self.Ns, N_chords))
            self.t_dens = t_dens
            self.phase_ref = np.array([])

            for i, chord in enumerate(Source.name_chords):

                data_chord = Source(chord)
                n_e_array = data_chord['n_e']
                n_fluc_array = data_chord['n_fluc']
                s_array = data_chord['s']
                N_array = int(data_chord['Array_length'][0])
                l_probe = data_chord['l_probe'][0]
                l_refer = data_chord['l_refer'][0]
                l_source = data_chord['l_source'][0]
                density_evolution_array = np.array([density_evolution])
                n_e_time = np.dot(density_evolution_array.T, n_e_array.T)
                n_ref_array = np.sqrt(1-np.multiply(1.0/Source.n_c, n_e_time))
                print('Size n_ref_array Narr: N = ',
                      len(n_ref_array[10, 0:N_array]))
                Phase_nkl = np.zeros(N_d)
                n_e_lin = np.zeros(N_d)
                n_fluc_lin = np.trapz(n_fluc_array[0:N_array].T,s_array[0:N_array].T)
                Phase_ref = np.amax(s_array)*Source.k
                print('length of chord:', np.amax(s_array))
                for j in range(N_d):
                    Phase_nkl[j] = np.multiply(Source.k,np.trapz(n_ref_array[j, 0:N_array], s_array[0:N_array].T))
                    n_e_lin[j] = np.trapz(n_e_time[j, 0:N_array], s_array[0:N_array].T)

                if Source.type_os == 'Cosine':
                    n_fluc_time = np.multiply(n_fluc_lin, np.cos(np.multiply(2*constants.pi*Source.f_os, self.t)))
                elif Source.type_os == 'Sawtooth':
                    n_fluc_time = np.multiply(n_fluc_lin, sig.sawtooth(np.multiply(2*constants.pi*Source.f_os, self.t)))
                else:
                    raise Exception('Shape not found')
                phase_fluc = np.multiply(Source.k/Source.n_c/2, n_fluc_time)

                P, R, phase = Interferometer_Signals_Preamp(self.t, Source.lmbd, self.A_P, self.A_R,
                               self.A_S, Source.deltaf, Source.deltaf_st,
                               Source.deltaPP, l_probe, l_refer, l_source, phase_fluc,
                               Phase_nkl, t_dens, Phase_ref, vibrations)


                self.name_chords = np.concatenate((self.name_chords,np.array([chord])))
                self.phase_ref = np.concatenate((self.phase_ref,np.array([Phase_ref])))
                self.n_e_lin[:,i] = n_e_lin
                self.phase_nkl[:,i] = Phase_nkl
                self.P[:,i] = P
                self.R[:,i] = R
                self.phase[:,i] = phase

        else:
            N_chords = len(Chords)
            self.n_e_lin = np.zeros((N_d,N_chords))
            self.phase_nkl = np.zeros((N_d,N_chords))
            self.P = np.zeros((self.Ns,N_chords))
            self.R = np.zeros((self.Ns,N_chords))
            self.phase = np.zeros((self.Ns,N_chords))
            self.t_dens = t_dens
            self.phase_ref = np.array([])
            for i,chord in enumerate(Chords):


                data_chord = Source(chord)
                n_e_array = data_chord['n_e']
                n_fluc_array = data_chord['n_fluc']
                s_array = data_chord['s']
                N_array = int(data_chord['Array_length'][0])
                l_probe = data_chord['l_probe'][0]
                l_refer = data_chord['l_refer'][0]
                l_source = data_chord['l_source'][0]
                density_evolution_array = np.array([density_evolution])
                n_e_time = np.dot(density_evolution_array.T,n_e_array.T)
                n_ref_array = np.sqrt(1-np.multiply(1.0/Source.n_c,n_e_time))
                print('Size s_array Narr: N = ',s_array[0:N_array].size)
                Phase_nkl = np.zeros(N_d)
                n_e_lin = np.zeros(N_d)
                n_fluc_lin = np.trapz(n_fluc_array[0:N_array].T,s_array[0:N_array].T)
                Phase_ref = np.amax(s_array)*Source.k
                print('length of chord:', np.amax(s_array))
                for j in range(N_d):
                    Phase_nkl[j] = np.multiply(Source.k,np.trapz(n_ref_array[j, 0:N_array], s_array[0:N_array].T))
                    n_e_lin[j] = np.trapz(n_e_time[j, 0:N_array], s_array[0:N_array].T)
                if Source.type_os == 'Cosine':
                    n_fluc_time = np.multiply(n_fluc_lin, np.cos(np.multiply(2*constants.pi*Source.f_os, self.t)))
                elif Source.type_os == 'Sawtooth':
                    n_fluc_time = np.multiply(n_fluc_lin, sig.sawtooth(np.multiply(2*constants.pi*Source.f_os, self.t)))
                else:
                    raise Exception('Shape not found')
                phase_fluc = np.multiply(Source.k/Source.n_c/2, n_fluc_time)
                for j in range(N_d):
                    Phase_nkl[j] = np.multiply(Source.k,np.trapz(n_ref_array[j,0:N_array], s_array[0:N_array].T))
                    n_e_lin[j] = np.trapz(n_e_time[j,0:N_array], s_array[0:N_array].T)

                phase_fluc = np.multiply(Source.k/Source.n_c/2, n_fluc_time)
                P, R, phase = Interferometer_Signals_Preamp(self.t,Source.lmbd, self.A_P, self.A_R,
                                self.A_S, Source.deltaf, Source.deltaf_st,
                                Source.deltaPP, l_probe, l_refer, l_source, phase_fluc,
                                Phase_nkl, t_dens, Phase_ref, vibrations)



                self.name_chords = np.concatenate((self.name_chords,np.array([chord])))
                self.phase_ref = np.concatenate((self.phase_ref,np.array([Phase_ref])))
                self.n_e_lin[:,i] = n_e_lin
                self.phase_nkl[:,i] = Phase_nkl
                self.P[:,i] = P
                self.R[:,i] = R
                self.phase[:,i] = phase






