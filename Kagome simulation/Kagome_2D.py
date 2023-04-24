#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 11:15:40 2023

@author: yichenzhang
"""

import numpy as np
import chinook.build_lib as build_lib
import chinook.operator_library as operators
from ArpesCNN.Gen_data import simulation
import os, inspect
import matplotlib.pyplot as plt
#from chinook.ARPES_lib import experiment

PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.mkdir(PATH + '/Parameters')
os.mkdir(PATH + '/Visualizations')

cube_tot = 1
cube_training_num = 1
cube_validation_num = 0
cube_testing_num = 0
slice_num = 10

if cube_training_num + cube_validation_num + cube_testing_num != cube_tot:
    raise Exception('Sum of the number of training, validation and testing cubes has to equal to the total number of cubes')


for i in range(cube_tot):
  # Step 1. Set up the basis dictionary
  a,c = 8.0,12.0
  avec = np.array([[a,0,0],[a/2,a*np.sqrt(3)/2,0],[0,0,c]])

  Z_A = np.array([0.0,0.0,0.0])
  Z_B = np.array([a/2,0.0,0.0])
  Z_C = np.array([a/4,a*np.sqrt(3)/4,0.0])

  basis = {'atoms':[0,1,2], 
  'Z':{0:26,1:26,2:26},    
  'orbs':[['32xy','32yz','32xz','32ZR','32XY'],['32xy','32yz','32xz','32ZR','32XY'],
          ['32xy','32yz','32xz','32ZR','32XY']],
  'pos':[Z_A,Z_B,Z_C]}

  basis_object = build_lib.gen_basis(basis)

  # Step 2. Define k-path
  kpoints = np.array([[0.0,0.0,0.0],[2/3,1/3,0.0],[0.5,0.0,0.0],[0.0,0.0,0.0]])
  labels = np.array(['$\\Gamma$','$K$','$M$','$\\Gamma$'])



  kdict = {'type':'F',
  'avec':avec,
  'pts':kpoints,
  'grain':200,
  'labels':labels}

  k_object = build_lib.gen_K(kdict)

  # Step 3. Set up the Hamiltonian dictionary
  E_A = 0
  E_B = 0
  E_C = 0

  t_ddS = -0.325
  t_ddP = 0.25
  t_ddD = -0.2

  VSK = {'032':E_A,'132':E_B,'232':E_C,'013322S':t_ddS,'013322P':t_ddP,'013322D':t_ddD,
         '023322S':t_ddS,'023322P':t_ddP,'023322D':t_ddD,
         '123322S':t_ddS,'123322P':t_ddP,'123322D':t_ddD}

  cutoff = 0.51 * a

  hamiltonian = {'type':'SK',    
              'V':VSK,          
              'avec':avec,    
              'cutoff':cutoff, 
              'renorm':1.0,     
              'offset':0.0,   
              'tol':1e-4} 

  #TB = build_lib.gen_TB(basis_object,hamiltonian,k_object)

  #TB.Kobj = k_object
  #TB.solve_H()
  #TB.plotting(win_min=-1.2,win_max=1.2)

  #TB.print_basis_summary()
  #sigma_bands = operators.fatbs(proj=[0,3,4,5,8,9,10,13,14],TB=TB,degen=False)
  #pi_bands = operators.fatbs(proj=[1,2,6,7,11,12],TB=TB,degen=False)

  # Step 4. Set up the ARPES dictionary
  k_bound = np.pi/a

  arpes = {'cube':{'X':[-1.5 * k_bound,1.5 * k_bound,300],'Y':[-k_bound,k_bound,300],'E':[-1.25,0.0,300],'kz':0.0}, 
          'hv':80,                          
          'T':10,                     
          'pol':np.array([1,0,-1]),           
          'SE':['poly',0.05,0,0.05],            
          'resolution':{'E':0.03,'k':0.01}}

  # Step 5. Feed the basis, hamiltonian, and ARPES dictionaries to the Simulation class
  sim = simulation(basis, hamiltonian, arpes)

  I = sim.TB_cube()
  NSR = 0.2
  Ig = sim.ARPESraw_cube(noise=True, NSR=NSR)

  if i < (cube_training_num): 
        sim.npy_from_cube(I, i, slice_num, -k_bound, 0., 'TB_train', path=PATH)
        sim.npy_from_cube(Ig, i, slice_num, -k_bound, 0., 'ARPES_train', path=PATH, noise=True, NSR=NSR)
        #np.savetxt('%s/Parameters/Param_list_%d.txt' %(PATH, i), Param_list)
        for j in range(slice_num):
            a = np.load('%s/train_hr/TB_sim_%d_%d.npy' %(PATH, i, j))
            b = np.load('%s/train_lr/ARPES_sim_%d_%d.npy' %(PATH, i, j))

            plt.subplot(121)
            plt.imshow(a, cmap='Blues')
            plt.subplot(122)
            plt.imshow(b, cmap='Blues')
            plt.savefig('%s/Visualizations/Sim_%d_%d.png' %(PATH, i, j))
            plt.close()




