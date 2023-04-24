#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 22:34:49 2023

@author: yichenzhang
"""

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
  a,c = 3.78,13.18
  avec = np.array([[a,0,0],[0,a,0],[0,0,c]])


  Cu = np.array([0.0,0.0,0.0])
  O_X = np.array([a/2,0.0,0.0])
  O_Y = np.array([0.0,a/2,0.0])
  O_a = np.array([0.0,0.0,0.64*a])
  O_b = np.array([0.0,0.0,-0.64*a])
  O_b_NN = np.array([a/2,a/2,1.105*a])
  O_a_NN = np.array([a/2,a/2,-1.105*a])

  #spin = {'bool':True, 
  #'soc':True,    
  #'lam':{0:0.1,1:0.55}}

  basis = {'atoms':[0,1,2,3,4,5,6], 
  'Z':{0:29,1:8,2:8,3:8,4:8,5:8,6:8},    
  'orbs':[['32XY','40'],['21x','21y'],['21x','21y'],['21z'],['21z'],['21z'],['21z']],
  'pos':[Cu,O_X,O_Y,O_a,O_b,O_b_NN,O_a_NN]}
  #'spin':spin} 
  basis_object = build_lib.gen_basis(basis)


  # Step 2. Define k-path
  #kpoints = np.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.5,0.5,0.0],[0.0,0.0,0.0]])
  #labels = np.array(['$\\Gamma$','$X$','$M$','$\\Gamma$'])



  #kdict = {'type':'F',
  #'avec':avec,
  #'pts':kpoints,
  #'grain':200,
  #'labels':labels}

  #k_object = build_lib.gen_K(kdict)

  # Step 3. Set up the Hamiltonian dictionary
  E_d = 0.0 # Onsite term for Cu 3dx^2-y^2, can be viewed as chemical potential tuning parameter
  E_p = -3.5 # Onsite term for O 2px/y
  E_z = -2.6 # Onsite term for O 2pz
  E_s = 6.5 # Onsite term for Cu 4s
    
  t_pd = 1.0
  V_01_pd = t_pd # Hopping between Cu 3dx^2-y^2 and O_X 2px orbital
  V_02_pd = -t_pd # Hopping between Cu 3dx^2-y^2 and O_Y 2py orbital
    
  t_sp = 1.3
  V_01_sp = t_sp # Hopping between Cu 4s and O_X 2px orbital
  V_02_sp = t_sp # Hopping between Cu 4s and O_Y 2py orbital
    
  t_ss = 0.40
  V_00_ss_N = -t_ss # Nearest hopping between Cu 4s and Cu 4s
    
  t_ss_NN = 0.10 #t_ss_prime
  V_00_ss_NN = -t_ss_NN # Next nearest hopping between Cu 4s and Cu 4s
    
  t_sigma_1 = 0.13
  V_11_pp_N_S = t_sigma_1 # Nearest sigma bonding between O_X 2p
  V_22_pp_N_S = t_sigma_1 # Nearest sigma bonding between O_y 2p
    
  t_pi_1 = 0.0325
  V_11_pp_N_P = -t_pi_1 # Nearest pi bonding between O_X 2p
  V_22_pp_N_P = -t_pi_1 # Nearest pi bonding between O_Y 2p
    
  t_sigma = 0.95 # Constructed under rotated basis. 
  t_pi = 0.2375 ## Constructed under rotated basis. 
  t_pp = (t_sigma + t_pi) / 2 # Sigma component of Sigma bonding component between O_X and O_Y 2p orbitals under natural basis
  t_pp_2 = (t_sigma - t_pi) / 2 # Sigma component of Pi bonding component between O_X and O_Y 2p orbitals under natural basis
  V_12_pp_S = -t_pp
  V_12_pp_P = t_pp_2
  t_sigma_2 = 0.4
  V_11_pp_S_NN = t_sigma_2 / 2 # Next nearest hopping sigma component between O_X 2p orbitals
  V_22_pp_S_NN = t_sigma_2 / 2# Next nearest hopping sigma component between O_Y 2p orbitals
    
  t_spz = 1.4
  V_03_spz = t_spz
  V_04_spz = -t_spz
    
  t_pz = 0.95
  V_31_pp = -t_pz
  V_32_pp = -t_pz
  V_41_pp = t_pz
  V_42_pp = t_pz
    
  t_pz_2 = 0.1
  V_52_pp = t_pz_2
  V_51_pp = t_pz_2
  V_62_pp = -t_pz_2
  V_61_pp = -t_pz_2
    
  t_pz_1 = 0.45
  V_53_pp = t_pz_1
  V_64_pp = t_pz_1
    
  #t_pz_3 = 0

  V1 = {'032':E_d,'040':E_s,'121':E_p,'221':E_p,'321':E_z,'421':E_z,'521':E_z,'621':E_z,
       '013221S':V_01_pd,'023221S':V_02_pd,'014201S':V_01_sp,
       '024201S':V_02_sp,'004400S':V_00_ss_N,'112211S':V_11_pp_N_S,
       '222211S':V_22_pp_N_S,'112211P':V_11_pp_N_P,'222211P':V_22_pp_N_P,
       '122211S':V_12_pp_S,'122211P':V_12_pp_P,'034201S':V_03_spz,
       '044201S':V_04_spz,'312211S':V_31_pp,'322211S':V_32_pp,
       '412211S':V_41_pp,'422211S':V_42_pp,'532211S':t_pz_1,'642211S':t_pz_1}
  V2 = {'004400S':V_00_ss_NN,'112211S':V_11_pp_S_NN,'222211S':V_22_pp_S_NN,
        '522211S':V_52_pp,'512211S':V_51_pp,'622211S':V_62_pp,'612211S':V_61_pp}
  VSK = [V1,V2]

  cutoff = [1.08*a,1.5*a]

  hamiltonian = {'type':'SK',    
              'V':VSK,          
              'avec':avec,    
              'cutoff':cutoff, 
              'renorm':1.0,     
              'offset':1.0,   
              'tol':1e-4}       
              #'spin':spin}
            

  # Step 4. Set up the ARPES dictionary
  k_bound = np.pi/a

  arpes = {'cube':{'X':[-k_bound,k_bound,300],'Y':[-k_bound,k_bound,300],'E':[-1.25,0.0,300],'kz':0.0}, 
          'hv':100,                          
          'T':10,                     
          'pol':np.array([1,0,-1]),           
          'SE':['poly',0.05,0,0.1],            
          'resolution':{'E':0.05,'k':0.02}}

  # Step 5. Feed the basis, hamiltonian, and ARPES dictionaries to the Simulation class
  sim = simulation(basis, hamiltonian, arpes)

  I = sim.TB_cube()
  NSR = 1.5
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




