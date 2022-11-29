#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:05:43 2022

@author: yichenzhang
"""

import numpy as np
from ArpesCNN.Gen_data import simulation
import matplotlib.pyplot as plt
import os, inspect

PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.mkdir(PATH + '/Parameters')
os.mkdir(PATH + '/Visualizations')

cube_num = 5
for i in range(cube_num):
# Step 1. Set up the basis dictionary
    a,c = 3.66,6.51
    avec = np.array([[a,0,0],
    [0,a,0],
    [0,0,c]])

    Fe = np.array([0.5*a,0.5*a,0.0])
    Te = np.array([0.0,0.5*a,0.72*c])

    #spin = {'bool':True, 
    #'soc':True,    
    #'lam':{0:0.1,1:0.55}}

    basis = {'atoms':[0,1], 
    'Z':{0:26,1:52},    
    'orbs':[['32xy','32yz','32xz','32ZR','32XY','40'],['51x','51y','51z']], 
    'pos':[Fe,Te]}
    #'spin':spin} 


# Step 2. Set up the Hamiltonian dictionary
    Param_list = []
    E_Fe_3d = np.random.randint(-3,2) + np.random.rand(1)
    Param_list.append(E_Fe_3d)
    E_Fe_4s = np.random.randint(-3,2) + np.random.rand(1)
    Param_list.append(E_Fe_4s)
    E_Te_5p = np.random.randint(-3,2) + np.random.rand(1)
    Param_list.append(E_Te_5p)
    V_4s_5p = np.random.randint(-2,1) + np.random.rand(1)
    Param_list.append(V_4s_5p)
    V_3d_5p_S = np.random.randint(-2,1) + np.random.rand(1)
    Param_list.append(V_3d_5p_S)
    V_3d_5p_P = np.random.randint(-2,1) + np.random.rand(1)
    Param_list.append(V_3d_5p_P)
    V_4s_4s = np.random.randint(-2,1) + np.random.rand(1)
    Param_list.append(V_4s_4s)
    V_5p_5p_S = -np.random.randint(-2,1) + np.random.rand(1)
    Param_list.append(V_5p_5p_S)
    V_5p_5p_P = np.random.randint(-2,1) + np.random.rand(1)
    Param_list.append(V_5p_5p_P)
    V_3d_3d_S = np.random.randint(-2,1) + np.random.rand(1)
    Param_list.append(V_3d_3d_S)
    V_3d_3d_P = np.random.randint(-2,1) + np.random.rand(1)
    Param_list.append(V_3d_3d_P)
    V_3d_3d_D = np.random.randint(-2,1) + np.random.rand(1)
    Param_list.append(V_3d_3d_D)

    V1 = {'032':E_Fe_3d,'040':E_Fe_4s,'151':E_Te_5p,'014501S':V_4s_5p,'013521S':V_3d_5p_S,'013521P':V_3d_5p_P}
    V2 = {'004400S':V_4s_4s/a,'115511S':V_5p_5p_S/a,'115511P':V_5p_5p_P/a,'003322S':V_3d_3d_S,'003322P':V_3d_3d_P,'003322D':V_3d_3d_D}
    VSK = [V1,V2]

    cutoff = [0.8*a,1.5*a]

    hamiltonian = {'type':'SK',    
            'V':VSK,          
            'avec':avec,    
            'cutoff':cutoff, 
            'renorm':1.0,     
            'offset':0.0,   
            'tol':1e-4}       
            #'spin':spin}

# Step 3. Set up the ARPES dictionary
    k_bound = np.pi/a
    SE = (0.01 + 0.035 * np.random.rand(1)) * 1j
    Param_list.append(SE)
    arpes = {'cube':{'X':[-k_bound,k_bound,300],'Y':[-k_bound,k_bound,300],'E':[-4.5,0.2,300],'kz':0.0}, 
        'hv':100,                          
        'T':10,                     
        'pol':np.array([1,0,-1]),           
        'SE':['constant',SE],            
        'resolution':{'E':0.12,'k':0.04}}

# Step 4. Feed the basis, hamiltonian, and ARPES dictionaries to the Simulation class
    sim = simulation(basis, hamiltonian, arpes)

    I = sim.TB_cube()
    NSR = 0.5
    #Param_list.append(NSR)
    Ig = sim.ARPESraw_cube(noise=True, NSR=NSR)
    slice_num = 16
    sim.npy_from_cube(I, i, slice_num, -k_bound, 0., 'TB', path=PATH)
    sim.npy_from_cube(Ig, i, slice_num, -k_bound, 0., 'ARPES', path=PATH, noise=True, NSR=NSR)
    np.savetxt('%s/Parameters/Param_list_%d.txt' %(PATH, i), Param_list)
    for j in range(slice_num):
        a = np.load('%s/TB_simulation/TB_sim_%d_%d.npy' %(PATH, i, j))
        b = np.load('%s/ARPES_simulation/ARPES_sim_%d_%d.npy' %(PATH, i, j))

        plt.subplot(121)
        plt.imshow(a, cmap='Blues')
        plt.subplot(122)
        plt.imshow(b, cmap='Blues')
        plt.savefig('%s/Visualizations/Sim_%d_%d.png' %(PATH, i, j))
        plt.close()