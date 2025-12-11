#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CURRENT VERSION WRITTEN BY OLIVIER DESTRIAN IN OCTOBER 2025
PAPER "Cytoplasmic crowding acts as a porous medium reducing macromolecule diffusion"
This code allows to plot curves for porous hindrances (nano and micro, tortuous and hydrodynamic)
And to recover the theroretical plots printed in main manuscript Fig 4
"""
import pandas as pd

import copy
import random
import math 
import time
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma
import matplotlib.image as mpl_img


#from P24_expLSM_FCT import *


import matplotlib.font_manager
font = {'family' : 'Arial',
         'weight' : 'bold',
         'size'   : 8}

plt.rc('font', **font)
#sns.set(font='Arial')
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['font.size'] = 8
#sns.set_style( {"grid.color": ".6", "grid.linestyle": ":"})
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = True
plot_borders_width=0.7
plt.rcParams['axes.linewidth'] = plot_borders_width
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['xtick.major.width'] = plot_borders_width
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['ytick.major.width'] = plot_borders_width
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['xtick.minor.width'] = plot_borders_width

plt.rcParams['boxplot.showbox']=False

def unique_string(array):
    L_2=[]
    for x in array:
        if x not in L_2 and pd.isnull(x)==0:
            L_2.append(x)
    return np.array(L_2)

L_color=['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'] #color blind colors
A=1/9 #quadratic coefficient for porous hydrodynamic  hindrance



#%%
"========================================================================================="
"=========================BASIC FUNCTION FOR PLOTTING POROUS HINDRANCES for 2D-CD and 3D-CFC ========================="
"========================================================================================="

subpath_effects='porous_hindrances_updated_2024_03_28/'
L_R_partic=[2.3]#gfp radius
L_R_solid=[12.5]#nano or micro-obstacle radius
L_vf=['occupied']#x-coordinate: volume fraction ['excluded' or 'occupied']
dimension=3#2 or 3

for i_vf in range(len(L_vf)):
    plt.figure(dpi=400)#,figsize=[9,6]
    vol_fraction=L_vf[i_vf]
    for i_R_partic in range(len(L_R_partic)):
        R_partic=L_R_partic[i_R_partic]#rayon particule diffusive
        for i_R_solid in range(len(L_R_solid)):
            R_solid=L_R_solid[i_R_solid]#rayon de l'obstacle solide immobile
         
            "1 - fichiers loadés : rayon réel=200nm"
            R_gfp=2.3#DO NOT TOUCH particle radius used for dimensionalisation in numerical computations
            if dimension==2:
                x_t=np.load(subpath_effects+'x_t_act_mesh0p05_refine1_2024_03_28.npy')#2023_12_11
                y_t=np.load(subpath_effects+'y_t_act_mesh0p05_refine1_2024_03_28.npy')#2023_12_11
                x_perm=np.load(subpath_effects+'x_perm_act_mesh0p05_refine1_2024_03_28.npy')#2023_12_11
                y_perm=np.load(subpath_effects+'y_perm_act_mesh0p05_refine1_2024_03_28.npy')#2023_12_11
                R_results=4#DO NOT TOUCH obstacle radius used for dimensionalisation in numerical computations
            elif dimension==3:
                x_t=np.load(subpath_effects+'x_t_orga_cfc_msh0p05_refine_1s2_2024_03_28.npy')
                y_t=np.load(subpath_effects+'y_t_orga_cfc_msh0p05_refine_1s2_2024_03_28.npy')
                x_perm=np.load(subpath_effects+'x_perm_orga_cfc_msh0p05_refine_1s2_2024_03_28.npy')
                y_perm=np.load(subpath_effects+'y_perm_orga_cfc_msh0p05_refine_1s2_2024_03_28.npy')
                R_results=200#DO NOT TOUCH obstacle radius used for dimensionalisation in numerical computations
            
            #tortuous recomputation
            if vol_fraction=='occupied': 
                x_t_excluded_results=((R_results+R_gfp)/R_results)**dimension*x_t
                x_t_excluded_2=x_t_excluded_results
                x_t_occupied_2=x_t_excluded_2/((R_solid+R_partic)/R_solid)**dimension
                x_t=x_t_occupied_2
            elif vol_fraction=='excluded':
                x_t_excluded_results=((R_results+R_gfp)/R_results)**dimension*x_t
                x_t=x_t_excluded_results
                
            #porous hydrodynamic drag recomputation
            y_perm_2=y_perm*(R_solid/R_results)**2 #re-dimensionalization [m²]
            y_perm=y_perm_2
            y_hd=[]; x_hd=[]
            for i in range(len(x_perm)):
                Perm=y_perm[i]
                C_HD=R_partic/(Perm**0.5)      
                Ratio_hd=(1+C_HD+A*C_HD**2)**(-1)
                y_hd.append(1-Ratio_hd)
                x_hd.append(x_perm[i])
            y_hd=np.array(y_hd)
            x_hd=np.array(x_hd)
            if vol_fraction=='excluded':
                x_hd_excluded_2=((R_solid+R_partic)/R_solid)**dimension*x_hd
                x_hd=x_hd_excluded_2
                
                
            #computing full model nano-obstacles effect
            lim=min(max(x_t),max(x_hd))
            x_t_shortened=x_t[x_t<=lim]
            x_hd_shortened=x_hd[x_hd<=lim]
            x_cb=np.sort(np.concatenate((x_t_shortened,x_hd_shortened)))
            y_hd_interp=np.interp(x_cb,x_hd,y_hd)
            y_t_interp=np.interp(x_cb,x_t,y_t)
            y_cb=1-(1-y_hd_interp)*(1-y_t_interp)
            

                
            #plot
            L_marker=['s','o','^','x']
            plt.plot(x_t,(1-y_t),label='Tortuosity',color=L_color[0],marker=L_marker[0],markevery=0.05)
            plt.plot(x_hd,(1-y_hd),label='Porous hydro. hind.',color=L_color[1],marker=L_marker[1],markevery=0.05)
            plt.plot(x_cb,(1-y_cb),label='Combination',color=L_color[2],marker=L_marker[2],markevery=0.05)
            
    plt.ylabel(r'Nano-obstacle porous hindrances')
    plt.legend(loc='lower right')
    title=f'Porous hindrances, {vol_fraction}, {dimension}D, Robs={R_solid}nm, Rpartic={R_partic}nm,vers2024 05 30'
    plt.title(title,fontsize='small')
    plt.grid()
    plt.ylim([0,1])
    if vol_fraction=='occupied':
        plt.xlabel(r'Nano-obstacle occupied fraction ')
        plt.xlim([0,0.4])
    elif vol_fraction=='excluded':
        plt.xlabel(r'Nano-obstacle excluded fraction $\epsilon_n$')
        plt.xlim([0,0.8])   
    #plt.savefig(title+".svg")
    plt.show()
        


#%%
"========================================================================================="
"=========================APPLICATioN TO COMPARE GFP MONOMER AND GFP DIMER FOR A GIVEN NANO-OBSTACLE EXCLUDED VOLUME FRACTION as in supplemetary Fig.========================="
"========================================================================================="
L_R_partic=[2.3,3.2]
L_R_solid=[12.5]
L_vf=['excluded']
dimension=3
plt.figure(dpi=400,figsize=[12/2.56,8/2.56])
for i_vf in range(len(L_vf)):
    
    vol_fraction=L_vf[i_vf]
    for i_R_partic in range(len(L_R_partic)):
        R_partic=L_R_partic[i_R_partic]#diffusive particle radius
        for i_R_solid in range(len(L_R_solid)):
            R_solid=L_R_solid[i_R_solid]#rayon de l'obstacle solide immobile
         
            "1 - loaded files: real obstacle size 200nm"
            R_gfp=2.3#DO NOT TOUCH particle radius used for dimensionalisation in numerical computations
            if dimension==2:
                x_t=np.load(subpath_effects+'x_t_act_mesh0p05_refine1_2024_03_28.npy')#2023_12_11
                y_t=np.load(subpath_effects+'y_t_act_mesh0p05_refine1_2024_03_28.npy')#2023_12_11
                x_perm=np.load(subpath_effects+'x_perm_act_mesh0p05_refine1_2024_03_28.npy')#2023_12_11
                y_perm=np.load(subpath_effects+'y_perm_act_mesh0p05_refine1_2024_03_28.npy')#2023_12_11
                R_results=4#DO NOT TOUCH particle radius used for dimensionalisation in numerical computations
            elif dimension==3:
                x_t=np.load(subpath_effects+'x_t_orga_cfc_msh0p05_refine_1s2_2024_03_28.npy')#'x_t_orga_mesh0p05_refine1_2023_12_11.npy')
                y_t=np.load(subpath_effects+'y_t_orga_cfc_msh0p05_refine_1s2_2024_03_28.npy')#'y_t_orga_mesh0p05_refine1_2023_12_11.npy'
                x_perm=np.load(subpath_effects+'x_perm_orga_cfc_msh0p05_refine_1s2_2024_03_28.npy')#'x_perm_orga_mesh0p05_refine1_2023_12_11.npy'
                y_perm=np.load(subpath_effects+'y_perm_orga_cfc_msh0p05_refine_1s2_2024_03_28.npy')#'y_perm_orga_mesh0p05_refine1_2023_12_11.npy'
                R_results=200##DO NOT TOUCH particle radius used for dimensionalisation in numerical computations
            #Hydrodynamic results are recomputed from permeability
            
            "2 - Recomputing tortuosity"
            if vol_fraction=='occupied': #corrected on 2024 02 12
                x_t_excluded_results=((R_results+R_gfp)/R_results)**dimension*x_t
                x_t_excluded_2=x_t_excluded_results #tortuosity is identical for identical excluded fraction
                x_t_occupied_2=x_t_excluded_2/((R_solid+R_partic)/R_solid)**dimension
                x_t=x_t_occupied_2
            elif vol_fraction=='excluded':
                x_t_excluded_results=((R_results+R_gfp)/R_results)**dimension*x_t
                x_t=x_t_excluded_results #tortusoity is identifical for identical excluded fraction
                
            "3 - Recomputation hydrodynamic hindrance (HD)"
            y_perm_2=y_perm*(R_solid/R_results)**2 #re-dimensionalization of permeability (in m2)
            y_perm=y_perm_2
            y_hd=[]; x_hd=[]
            for i in range(len(x_perm)):
                Perm=y_perm[i]
                C_HD=R_partic/(Perm**0.5)      
                Ratio_hd=(1+C_HD+A*C_HD**2)**(-1)
                y_hd.append(1-Ratio_hd)
                x_hd.append(x_perm[i])
            y_hd=np.array(y_hd)
            x_hd=np.array(x_hd)
            if vol_fraction=='excluded':
                x_hd_excluded_2=((R_solid+R_partic)/R_solid)**dimension*x_hd
                x_hd=x_hd_excluded_2
            "4 - construction of the curve for the combined hydro and tortuosity"  
            lim=min(max(x_t),max(x_hd))
            x_t_shortened=x_t[x_t<=lim]
            x_hd_shortened=x_hd[x_hd<=lim]
            x_cb=np.sort(np.concatenate((x_t_shortened,x_hd_shortened)))
            y_hd_interp=np.interp(x_cb,x_hd,y_hd)
            y_t_interp=np.interp(x_cb,x_t,y_t)
            y_cb=1-(1-y_hd_interp)*(1-y_t_interp)
            

                
            "5 - print"
            L_marker=['s','o','^','x']
            plt.plot(x_t,(1-y_t),'k:',label=r'$T_\alpha^{-1}$',color=L_color[i_R_partic],marker=L_marker[2*i_R_partic],markevery=0.2)
            plt.plot(x_hd,(1-y_hd),'k--',label=r'$H_\alpha^{-1}$',color=L_color[i_R_partic],marker=L_marker[2*i_R_partic],markevery=0.08)
            plt.plot(x_cb,(1-y_cb),'k',label=r'$T_\alpha^{-1} \times H_\alpha^{-1}$',color=L_color[i_R_partic],marker=L_marker[2*i_R_partic],markevery=0.2)
            
    plt.ylabel(r'Nano-obstacle hindrances')
    #plt.legend(loc='lower left')
    title=f'{vol_fraction}, {dimension}D, Robs={R_solid}nm, Rpartic={R_partic}nm,vers2024 05 30'
    plt.title(title,fontsize='x-small')
    #plt.grid()
    plt.ylim([0,1])
    if vol_fraction=='occupied':
        plt.xlim([0,0.5])
    elif vol_fraction=='excluded':
        plt.xlabel(r'Nano-obstacle fraction $\epsilon_n$')
        plt.xlim([0,0.6])   
    #plt.savefig(title+".svg")
    plt.show()
        





#%%
"Fig 4 plots"
path=''
file='results_3D_aleatoire_2024_04_29'

D2_solid='2d_r4nm'  #'2d_r4nm'=F-actine   ; '2d_r12p5nm'=microtubules ; '3d_r12p5nm'=ribosomes
D3_solid='3d_r12p5nm'

subpath_effects=path+'porous_hindrances_updated_2024_03_28/'#'porous_hindrances_2024_02_12/'#'porous_hindrances_2024_02_12/'#'hindrance_effects_vers2023_12_12/' #dossier avec les résultats des calculs théoriques T,HD,CB issus des problèmes de fermeture

"loading results from simulations"
x_t_ex_simple_2D=np.load(subpath_effects+'x_t_ex_'+D2_solid+'.npy') #excluded vol
y_t_ex_simple_2D=np.load(subpath_effects+'y_t_ex_'+D2_solid+'.npy') #excluded vol
x_t_ex_simple_3D=np.load(subpath_effects+'x_t_ex_'+D3_solid+'.npy') #excluded vol
y_t_ex_simple_3D=np.load(subpath_effects+'y_t_ex_'+D3_solid+'.npy') #excluded vol

"""conversion"""
R_gfp=2.3 #GFP radius (2.3nm)
R_act=4.0 #2D nano-obstacle radius (F-actin 4 [nm])
R_rib=12.5 #3D nano-obstacle radius (ribosome 12.5[nm])
x_t_oc_simple_2D=x_t_ex_simple_2D*(R_act/(R_act+R_gfp))**2
x_t_oc_simple_3D=x_t_ex_simple_3D*(R_rib/(R_rib+R_gfp))**3


"1 - loading files (real radius 200Nm)"
for dimension in [2,3]:
    if dimension==2:
        x_perm=np.load(subpath_effects+'x_perm_act_mesh0p05_refine1_2024_03_28.npy')
        y_perm=np.load(subpath_effects+'y_perm_act_mesh0p05_refine1_2024_03_28.npy')
        R_results=4#do not modify, radius used for numerical 2D simulations 4nm
        R_solid=4#new obstacle radius
        y_perm_2=y_perm*(R_solid/R_results)**2 #re-dimensionalisation permeability in [m²]
        y_perm=y_perm_2
        x_perm_oc_simple_2D=x_perm
        y_perm_oc_simple_2D=y_perm_2
        x_perm_ex_simple_2D=x_perm_oc_simple_2D*((R_act+R_gfp)/R_act)**2
        
    elif dimension==3:
        x_perm=np.load(subpath_effects+'x_perm_orga_cfc_msh0p05_refine_1s2_2024_03_28.npy')
        y_perm=np.load(subpath_effects+'y_perm_orga_cfc_msh0p05_refine_1s2_2024_03_28.npy')
        R_results=200#do not modify, radius used for numerical 3D simulations 200nm
        R_solid=12.5#new obstacle radius
        y_perm_2=y_perm*(R_solid/R_results)**2 #re-dimensionalisation permeability in [m²]
        x_perm_oc_simple_3D=x_perm
        y_perm_oc_simple_3D=y_perm_2
        x_perm_ex_simple_3D=x_perm_oc_simple_3D*((R_rib+R_gfp)/R_rib)**3
        



"===================IMPORTATION -- 3D FIBRES --- TORT ================"
df_tort_import=pd.read_excel (path+file+'.xlsx',sheet_name='tort_fibres_2024_05_06') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
df_tort_import=df_tort_import[['Porosite','TH_x','TH_y','TH_z','R_ex','N_obs']]
df_tort_import['Porosite']=1-df_tort_import['Porosite']
df_tort_import=df_tort_import.rename({'Porosite':'fs_ex'},axis=1)

"=================== IMPORTATION -- 3D FIBRES --- PERM ================"
df_perm_import=pd.read_excel (path+file+'.xlsx',sheet_name='perm_fibres_2024_04_29') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
df_perm_import=df_perm_import[['Porosite','Perm_adim','R_ex','N_obs']]
df_perm_import['Porosite']=1-df_perm_import['Porosite']
R_nano=4 #en nm
df_perm_import['Perm_adim']=df_perm_import['Perm_adim']*(R_nano/df_perm_import['R_ex'])**2
df_perm_import=df_perm_import.rename({'Porosite':'fs_oc','R_ex':'R_oc','Perm_adim':'Perm_nm2'},axis=1)

"======================== LIEN TORTUOSITE ET PERM======================"
"TORT - Binning by radius and number of obstacles"
array_radius_num=df_tort_import['R_ex']
array_N_obs=df_tort_import['N_obs']

L_R_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std=[],[],[],[],[],[]
array_unique_radius_num=unique_string(array_radius_num)
for i in range(array_unique_radius_num.size):
    
        array_N_obs_local=array_N_obs[array_radius_num==array_unique_radius_num[i]]
        array_unique_N_obs_local=unique_string(array_N_obs_local)
        for j in range(array_unique_N_obs_local.size):
            L_R_num_bin.append(array_unique_radius_num[i])
            L_N_obs_bin.append(array_unique_N_obs_local[j])
            condition=(array_radius_num==array_unique_radius_num[i])*(array_N_obs==array_unique_N_obs_local[j])
            L_fs_oc_bin.append(np.mean(df_tort_import['fs_ex'][condition]))
            L_perm_bin.append(np.mean(df_tort_import[['TH_x','TH_y','TH_z']][condition]))
            L_fs_oc_bin_std.append(np.std(df_tort_import['fs_ex'][condition]))
            L_perm_bin_std.append(np.std(np.array(df_tort_import[['TH_x','TH_y','TH_z']][condition])))
        
L_R_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std=np.array(L_R_num_bin),np.array(L_N_obs_bin),np.array(L_fs_oc_bin),np.array(L_perm_bin),np.array(L_fs_oc_bin_std),np.array(L_perm_bin_std)      
       
charray_tort_results_fibers=np.array([L_R_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std])
charray_tort_results_fibers=np.transpose(charray_tort_results_fibers)
df_tort_results_fibers= pd.DataFrame(charray_tort_results_fibers, columns=['R_ex','N_obs','fs_ex','tort','fs_ex_std','tort_std'])

"permeability - binning by radius and number of obstacles"
array_radius_num=df_perm_import['R_oc']
array_N_obs=df_perm_import['N_obs']

L_R_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std=[],[],[],[],[],[]
array_unique_radius_num=unique_string(array_radius_num)
for i in range(array_unique_radius_num.size):
    
        array_N_obs_local=array_N_obs[array_radius_num==array_unique_radius_num[i]]
        array_unique_N_obs_local=unique_string(array_N_obs_local)
        for j in range(array_unique_N_obs_local.size):
            L_R_num_bin.append(array_unique_radius_num[i])
            L_N_obs_bin.append(array_unique_N_obs_local[j])
            condition=(df_perm_import['R_oc']==array_unique_radius_num[i])*(df_perm_import['N_obs']==array_unique_N_obs_local[j])
            L_fs_oc_bin.append(np.mean(df_perm_import['fs_oc'][condition]))
            L_perm_bin.append(np.mean(df_perm_import['Perm_nm2'][condition]))
            L_fs_oc_bin_std.append(np.std(df_perm_import['fs_oc'][condition]))
            L_perm_bin_std.append(np.std(df_perm_import['Perm_nm2'][condition]))
        
L_R_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std=np.array(L_R_num_bin),np.array(L_N_obs_bin),np.array(L_fs_oc_bin),np.array(L_perm_bin),np.array(L_fs_oc_bin_std),np.array(L_perm_bin_std)      
       
charray_perm_results_fibers=np.array([L_R_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std])
charray_perm_results_fibers=np.transpose(charray_perm_results_fibers)
df_perm_results_fibers= pd.DataFrame(charray_perm_results_fibers, columns=['R_oc','N_obs','fs_oc','perm_nm2','fs_oc_std','perm_nm2_std'])

"===============Fusion of dataframes for tortuosity and permeability==============="
R_GFP=2.3
R_act=4.0
R_ex_oc_ratio=(R_GFP+R_act)/R_act

"---charray_combined creation from TORT et PERM results---"
N_c_tort=charray_tort_results_fibers.shape[1]
N_c_perm=charray_perm_results_fibers.shape[1]-1 #-1 to avoid putting twice the number of obstacles
charray_combined_results_fiber=np.zeros((0,N_c_tort+N_c_perm),dtype=np.float64) # charray combined definition

array_radius_num=df_perm_results_fibers['R_oc']
for i_R_oc in range (array_radius_num.size):
    R_oc=array_radius_num[i_R_oc]    
    array_N_obs_local=np.array(df_perm_results_fibers['N_obs'][array_radius_num==R_oc])
    
    R_ex=R_oc*R_ex_oc_ratio
    
    for i_N_obs in range(array_N_obs_local.size) :
        N_obs=array_N_obs_local[i_N_obs]
                    
        "recherche d'une correspondance entre les datasets tort et perm"
        i_correspondance_tort=np.array((abs(df_tort_results_fibers['R_ex']-R_ex)<0.001)*(df_tort_results_fibers['N_obs']==N_obs))
        i_correspondance_perm=np.array((df_perm_results_fibers['R_oc']==R_oc)*(df_perm_results_fibers['N_obs']==N_obs))
        "si pas de correspondance : les données ne sont pas récupérées"
        if True in np.array(i_correspondance_tort):   
            charray_combined_results_fiber_sub=np.concatenate((charray_tort_results_fibers[i_correspondance_tort,:],charray_perm_results_fibers[i_correspondance_perm,:1],charray_perm_results_fibers[i_correspondance_perm,2:]),axis=1)
            charray_combined_results_fiber=np.concatenate((charray_combined_results_fiber,charray_combined_results_fiber_sub),axis=0)

df_combined_results_fibers= pd.DataFrame(charray_combined_results_fiber, columns=['R_ex','N_obs','fs_ex','tort','fs_ex_std','tort_std','R_oc','fs_oc','perm_nm2','fs_oc_std','perm_nm2_std'])

figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
fig,ax=plt.subplots(figsize=figsize,dpi=800)
# ax.plot(df_combined_results_fibers['fs_oc'],df_combined_results_fibers['fs_ex'],'x')
# ax.plot([0,1],[0,1],'k--')
# #ax.grid()
# ax.set_xlim([0,0.9])
plt.xticks(np.arange(0,1,0.1))#ax.set_xlim([0,0.9])
plt.xticks(np.arange(0,1,0.1))
# ax.set_ylim([0,1])
# # ax.set_aspect('equal')
# plt.xlabel(r'Occupied fiber fraction ')
# plt.ylabel(r'Excluded fiber fraction $\epsilon_n$')





#%%

# 

'======================== 3D SPHERES + FIBRES ======================'
"=================== IMPORTATION --- 3D MIXED --- TORT ================"
df_tort=pd.read_excel (path+file+'.xlsx',sheet_name='tort_f_and_s_2024_04_29') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
array_tort=pd.DataFrame(df_tort).to_numpy()
X_fs=1-array_tort[:,0]
Y_fn=np.mean(array_tort[:,1:4],axis=1)


"===================IMPORTATION -- 3D FIBRES --- TORT ================"
df_tort_import=pd.read_excel (path+file+'.xlsx',sheet_name='tort_f_and_s_2024_04_29') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
df_tort_import=df_tort_import[['Porosite','TH_x','TH_y','TH_z','[R_act,R_rib]','[N_act,N_rib]']]
df_tort_import['Porosite']=1-df_tort_import['Porosite']
df_tort_import=df_tort_import.rename({'Porosite':'fs_ex'},axis=1)

"=================== IMPORTATION -- 3D FIBRES --- PERM ================"
df_perm_import=pd.read_excel (path+file+'.xlsx',sheet_name='perm_f_and_s_2024_04_29') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
df_perm_import=df_perm_import[['Porosite','Perm_adim','[R_act,R_rib]','[N_act,N_rib]']]
df_perm_import['Porosite']=1-df_perm_import['Porosite']
R_nano=4 #en nm
R_num_oc=0.06
df_perm_import['Perm_adim']=df_perm_import['Perm_adim']*(R_nano/R_num_oc)**2
df_perm_import=df_perm_import.rename({'Porosite':'fs_oc','R_ex':'R_oc','Perm_adim':'Perm_nm2'},axis=1)

"======================== 3D random MIXED - TORTUOSITY and PERM======================"
"TORT - Binning by radius and number of obstacles"
array_radius_num=df_tort_import['[R_act,R_rib]']
array_N_obs=df_tort_import['[N_act,N_rib]']

L_R_act_num_bin,L_R_rib_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std=[],[],[],[],[],[],[]
array_unique_radius_num=unique_string(array_radius_num)
    
array_N_obs_local=array_N_obs
array_unique_N_obs_local=unique_string(array_N_obs_local)
for j in range(array_unique_N_obs_local.size):
    L_R_act_num_bin.append(np.fromstring(array_unique_radius_num[0],sep=', ')[0])
    L_R_rib_num_bin.append(np.fromstring(array_unique_radius_num[0],sep=', ')[1])
    L_N_obs_bin.append(array_unique_N_obs_local[j])
    condition=(array_N_obs==array_unique_N_obs_local[j])
    L_fs_oc_bin.append(np.mean(df_tort_import['fs_ex'][condition]))
    L_perm_bin.append(np.mean(df_tort_import[['TH_x','TH_y','TH_z']][condition]))
    L_fs_oc_bin_std.append(np.std(df_tort_import['fs_ex'][condition]))
    L_perm_bin_std.append(np.std(np.array(df_tort_import[['TH_x','TH_y','TH_z']][condition])))


L_R_act_num_bin,L_R_rib_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std=np.array(L_R_act_num_bin),np.array(L_R_rib_num_bin),np.array(L_N_obs_bin),np.array(L_fs_oc_bin),np.array(L_perm_bin),np.array(L_fs_oc_bin_std),np.array(L_perm_bin_std)      
       
charray_tort_results_mixed=np.array([L_R_act_num_bin,L_R_rib_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std])
charray_tort_results_mixed=np.transpose(charray_tort_results_mixed)
df_tort_results_mixed= pd.DataFrame(charray_tort_results_mixed, columns=['R_act_ex','R_rib_ex','N_obs','fs_ex','tort','fs_ex_std','tort_std'])

"PERM - Binning by radius and number of obstacles "
array_radius_num=df_perm_import['[R_act,R_rib]']
array_N_obs=df_perm_import['[N_act,N_rib]']

L_R_act_num_bin,L_R_rib_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std=[],[],[],[],[],[],[]
array_unique_radius_num=unique_string(array_radius_num)
    
array_N_obs_local=array_N_obs
array_unique_N_obs_local=unique_string(array_N_obs_local)
for j in range(array_unique_N_obs_local.size):
    L_R_act_num_bin.append(np.fromstring(array_unique_radius_num[0],sep=', ')[0])
    L_R_rib_num_bin.append(np.fromstring(array_unique_radius_num[0],sep=', ')[1])
    L_N_obs_bin.append(array_unique_N_obs_local[j])
    condition=(array_N_obs==array_unique_N_obs_local[j])
    L_fs_oc_bin.append(np.mean(df_perm_import['fs_oc'][condition]))
    L_perm_bin.append(np.mean(df_perm_import['Perm_nm2'][condition]))
    L_fs_oc_bin_std.append(np.std(df_perm_import['fs_oc'][condition]))
    L_perm_bin_std.append(np.std(df_perm_import['Perm_nm2'][condition]))

L_R_act_num_bin,L_R_rib_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std=np.array(L_R_act_num_bin),np.array(L_R_rib_num_bin),np.array(L_N_obs_bin),np.array(L_fs_oc_bin),np.array(L_perm_bin),np.array(L_fs_oc_bin_std),np.array(L_perm_bin_std)      
       
charray_perm_results_mixed=np.array([L_R_act_num_bin,L_R_rib_num_bin,L_N_obs_bin,L_fs_oc_bin,L_perm_bin,L_fs_oc_bin_std,L_perm_bin_std])
charray_perm_results_mixed=np.transpose(charray_perm_results_mixed)
df_perm_results_mixed= pd.DataFrame(charray_perm_results_mixed, columns=['R_act_oc','R_rib_oc','N_obs','fs_oc','perm_nm2','fs_oc_std','perm_nm2_std'])
       
"===============3D MIXED - FUSION of dataframes for TORT ET PERM==============="
"---harray_combined creation from TORT and PERM results---"
N_c_tort=charray_tort_results_mixed.shape[1]
N_c_perm=charray_perm_results_mixed.shape[1]-1 #-1 to avoid putting twice the number of obstacles
charray_combined_results_mixed=np.zeros((0,N_c_tort+N_c_perm),dtype=np.float64) #définition charray combined

array_N_obs=df_perm_results_mixed['N_obs']
for i_N_obs in range (array_N_obs.size):
    N_obs=array_N_obs[i_N_obs]    

    "recherche d'une correspondance entre les datasets tort et perm"
    i_correspondance_tort=np.array((df_tort_results_mixed['N_obs']==N_obs))
    i_correspondance_perm=np.array((df_perm_results_mixed['N_obs']==N_obs))
    "si pas de correspondance : les données ne sont pas récupérées"
    if True in np.array(i_correspondance_tort):   
        charray_combined_results_mixed_sub=np.concatenate((charray_tort_results_mixed[i_correspondance_tort,:],charray_perm_results_mixed[i_correspondance_perm,:2],charray_perm_results_mixed[i_correspondance_perm,3:]),axis=1)
        charray_combined_results_mixed=np.concatenate((charray_combined_results_mixed,charray_combined_results_mixed_sub),axis=0)

df_combined_results_mixed= pd.DataFrame(charray_combined_results_mixed, columns=['R_act_ex','R_rib_ex','N_obs','fs_ex','tort','fs_ex_std','tort_std','R_act_oc','R_rib_oc','fs_oc','perm_nm2','fs_oc_std','perm_nm2_std'])










" LINK OCCUPIED AND EXCLUDED NANO-obstacle FRACTIONS"
figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
fig,ax=plt.subplots(figsize=figsize,dpi=800)
ax.plot([0,1],[0,1],'k:',label='Infinitely large solids')
L_sort=np.argsort(df_combined_results_fibers['fs_oc'])
ax.errorbar(df_combined_results_fibers['fs_oc'][L_sort],df_combined_results_fibers['fs_ex'][L_sort],yerr=df_combined_results_fibers['fs_ex_std'][L_sort],xerr=df_combined_results_fibers['fs_oc_std'][L_sort],label='3D-random F-actin',color=L_color[2],capsize=3)
L_sort=np.argsort(df_combined_results_mixed['fs_oc'])
ax.errorbar(df_combined_results_mixed['fs_oc'][L_sort],df_combined_results_mixed['fs_ex'][L_sort],yerr=df_combined_results_mixed['fs_ex_std'][L_sort],xerr=df_combined_results_mixed['fs_oc_std'][L_sort],label='3D-random Mixed',color=L_color[3],capsize=3)

ax.plot([0,1],[0,(6.3/4)**2],'-',label='2D-CD F-actin',color=L_color[0])
ax.plot([0,1],[0,(14.8/12.5)**3],'-',label='3D-CFC Ribosomes',color=L_color[1])


#ax.grid()
ax.set_xlim([0,0.9])
plt.xticks(np.arange(0,1,0.1))#ax.set_xlim([0,0.9])
ax.set_ylim([0,1])
#ax.set_aspect('equal')
plt.xlabel(r'Nano-obstacle occupied fraction ',font=font)
plt.ylabel(r'Nano-obstacle excluded fraction   $\epsilon_n$',font=font)
title='f_oc_and_f_ex_allgeometries_2025_05_06'
plt.title(title,fontsize='small')
# plt.legend(title='Geometry',fontsize='small')
# plt.savefig(title+".svg")
plt.show()

# "TORTUOSITY - OCCUPIED VOLUME (NANO OBSTACLES) "
# figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
# fig,ax=plt.subplots(figsize=figsize,dpi=800)
# L_sort=np.argsort(df_combined_results_fibers['fs_oc'])
# ax.errorbar(df_combined_results_fibers['fs_oc'][L_sort],df_combined_results_fibers['tort'][L_sort],yerr=df_combined_results_fibers['tort_std'][L_sort],xerr=df_combined_results_fibers['fs_oc_std'][L_sort],label='3D-random F-actin',color=L_color[2],capsize=3)
# L_sort=np.argsort(df_combined_results_mixed['fs_oc'])
# ax.errorbar(df_combined_results_mixed['fs_oc'][L_sort],df_combined_results_mixed['tort'][L_sort],yerr=df_combined_results_mixed['tort_std'][L_sort],xerr=df_combined_results_mixed['fs_oc_std'][L_sort],label='3D-random Mixed',color=L_color[3],capsize=3)

# ax.plot(x_t_oc_simple_2D,1-y_t_ex_simple_2D,'-',label='2D-CD F-actin',color=L_color[0])
# ax.plot(x_t_oc_simple_3D,1-y_t_ex_simple_3D,'-',label='3D-CFC Ribosomes',color=L_color[1])

# #ax.grid()
# ax.set_xlim([0,0.9])
# plt.xticks(np.arange(0,1,0.1))#ax.set_xlim([0,0.5])
# ax.set_ylim([0,1])
# ax.set_box_aspect(1)
# plt.xlabel(r'Nano-obstacle occupied fraction   ',font=font)
# plt.ylabel(r'Relative diffusivity   $D_\gamma^{TH}$/$D_\alpha$',font=font)
# title='TH_oc_allgeometries_2025_05_06'
# plt.title(title,fontsize='small')
# plt.legend(title='Geometry',fontsize='small')
# plt.savefig(title+".svg")
# plt.show()




"SEPARATE LEGEND PRINT"
figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
fig,ax=plt.subplots(figsize=figsize,dpi=800)
ax.errorbar([1],[1],yerr=[1],xerr=[1],label='3D-random F-actin',color=L_color[2],capsize=3,linewidth=1,marker='<',ms=3,zorder=1,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])
ax.errorbar([1],[1],yerr=[1],xerr=[1],label='3D-random Mixed',color=L_color[3],capsize=3,linewidth=1,marker='s',ms=2.5,zorder=0,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])
ax.plot([1],[1],'-',label='2D-CD F-actin',color=L_color[0],linewidth=1,marker='o',markevery=10,ms=3,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])#,capsize=3,linewidth=1,marker='<',ms=5,zorder=1),capsize=3,linewidth=1,marker='s',ms=4,zorder=0),linewidth=1,marker='o',markevery=10,ms=5),linewidth=1,marker='^',markevery=3,ms=5)
ax.plot([1],[1],'-',label='3D-CFC Ribosomes',color=L_color[1],linewidth=1,marker='^',markevery=3,ms=3,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])
ax.set_xlim([0,0.9])
ax.set_ylim([0,1])
ax.set_box_aspect(1)
plt.savefig(title+".svg")
plt.legend()#fontsize='small'
plt.show()


"TORTUOSITY - EXCLUDED VOLUME (NANO OBSTACLES) "
figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
fig,ax=plt.subplots(figsize=figsize,dpi=800)
L_sort=np.argsort(df_combined_results_fibers['fs_oc'])
ax.errorbar(df_combined_results_fibers['fs_ex'][L_sort],df_combined_results_fibers['tort'][L_sort],yerr=df_combined_results_fibers['tort_std'][L_sort],xerr=df_combined_results_fibers['fs_ex_std'][L_sort],label='3D-random F-actin',color=L_color[2],capsize=3,linewidth=1,marker='<',ms=3,zorder=1,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])
L_sort=np.argsort(df_combined_results_mixed['fs_oc'])
ax.errorbar(df_combined_results_mixed['fs_ex'][L_sort],df_combined_results_mixed['tort'][L_sort],yerr=df_combined_results_mixed['tort_std'][L_sort],xerr=df_combined_results_mixed['fs_ex_std'][L_sort],label='3D-random Mixed',color=L_color[3],capsize=3,linewidth=1,marker='s',ms=2.5,zorder=0,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])

ax.plot(x_t_ex_simple_2D,1-y_t_ex_simple_2D,'-',label='2D-CD F-actin',color=L_color[0],linewidth=1,marker='o',markevery=10,ms=3,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])#,capsize=3,linewidth=1,marker='<',ms=5,zorder=1),capsize=3,linewidth=1,marker='s',ms=4,zorder=0),linewidth=1,marker='o',markevery=10,ms=5),linewidth=1,marker='^',markevery=3,ms=5)
ax.plot(x_t_ex_simple_3D,1-y_t_ex_simple_3D,'-',label='3D-CFC Ribosomes',color=L_color[1],linewidth=1,marker='^',markevery=3,ms=3,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])

#ax.grid()
ax.set_xlim([0,0.9])
plt.xticks(np.arange(0,1,0.2))#ax.set_xlim([0,0.9])
ax.set_ylim([0,1])
ax.set_box_aspect(1)
plt.xlabel(r'Nano-obs excluded fraction   $\epsilon_n$',font=font)
plt.ylabel(r'Nano-obs tortuosity',font=font)
# title='TH_ex_allgeometries_2025_05_06'
# plt.title(title,fontsize='x-small')
#plt.legend(title='Geometry',fontsize='small')
plt.savefig(title+".svg")
plt.show()


"TORTUOSITY - MICRO-OBSTACLES"
figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
fig,ax=plt.subplots(figsize=figsize,dpi=800)
L_sort=np.argsort(df_combined_results_fibers['fs_oc'])
#ax.errorbar(df_combined_results_fibers['fs_ex'][L_sort],df_combined_results_fibers['tort'][L_sort],yerr=df_combined_results_fibers['tort_std'][L_sort],xerr=df_combined_results_fibers['fs_ex_std'][L_sort],label='3D-random F-actin',color=L_color[2],capsize=3)
L_sort=np.argsort(df_combined_results_mixed['fs_oc'])
#ax.errorbar(df_combined_results_mixed['fs_ex'][L_sort],df_combined_results_mixed['tort'][L_sort],yerr=df_combined_results_mixed['tort_std'][L_sort],xerr=df_combined_results_mixed['fs_ex_std'][L_sort],label='3D-random Mixed',color=L_color[3],capsize=3)

#ax.plot(x_t_ex_simple_2D,1-y_t_ex_simple_2D,'-',label='2D-CD Fibers (any radius)',color=L_color[0])
ax.plot(x_t_ex_simple_3D,1-y_t_ex_simple_3D,'-',label='3D-CFC Spherical micro-obstacles',color='k',linewidth=1)#,color=L_color[0]

#ax.grid()
ax.set_xlim([0,0.9])
plt.xticks(np.arange(0,1,0.2))#ax.set_xlim([0,0.9])
ax.set_ylim([0,1])
plt.xlabel(r'Micro-obstacle fraction   $\Phi_m$',font=font)
plt.ylabel(r'Micro-obs tortuosity',font=font)
# title='TH_micro_2025_05_06'
# plt.title(title,fontsize='small')
#plt.legend(title='Geometry',fontsize='small')
plt.savefig(title+".svg")
plt.show()
#,capsize=3,linewidth=1,marker='<',ms=5,zorder=1),capsize=3,linewidth=1,marker='s',ms=4,zorder=0),linewidth=1,marker='o',markevery=10,ms=5),linewidth=1,marker='^',markevery=3,ms=5)


"PERMEABILITY - occupied fraction "
# figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
# fig,ax=plt.subplots(figsize=figsize,dpi=800)
# L_sort=np.argsort(df_combined_results_fibers['fs_oc'])
# ax.errorbar(df_combined_results_fibers['fs_oc'][L_sort],df_combined_results_fibers['perm_nm2'][L_sort],yerr=df_combined_results_fibers['perm_nm2_std'][L_sort],xerr=df_combined_results_fibers['fs_oc_std'][L_sort],label='3D-random F-actin',color=L_color[2],capsize=3)
# L_sort=np.argsort(df_combined_results_mixed['fs_oc'])
# ax.errorbar(df_combined_results_mixed['fs_oc'][L_sort],df_combined_results_mixed['perm_nm2'][L_sort],yerr=df_combined_results_mixed['perm_nm2_std'][L_sort],xerr=df_combined_results_mixed['fs_oc_std'][L_sort],label='3D-random Mixed',color=L_color[3],capsize=3)

# ax.plot(x_perm_oc_simple_2D,y_perm_oc_simple_2D,'-',label='2D-CD F-actin',color=L_color[0])
# ax.plot(x_perm_oc_simple_3D,y_perm_oc_simple_3D,'-',label='3D-CFC Ribosomes',color=L_color[1])

# #ax.grid()
# ax.set_xlim([0,0.9])
# plt.xticks(np.arange(0,1,0.1))#ax.set_xlim([0,0.5])
# ax.set_ylim([5e-2,5e4])
# ax.set_box_aspect(1)
# plt.xlabel(r'Nano-obstacle occupied fraction   ',font=font)
# plt.ylabel(r'Nano-scale permeability   $K_\alpha$ (nm²)',font=font)
# title='perm_oc_allgeometries_2025_05_06'
# plt.yscale('log')
# plt.title(title,fontsize='small')
# plt.legend(title='Geometry',fontsize='small')
# plt.savefig(title+".svg")
# plt.show()


"PERMEABILITY - excluded fraction"
# figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
# fig,ax=plt.subplots(figsize=figsize,dpi=800)
# L_sort=np.argsort(df_combined_results_fibers['fs_oc'])
# ax.errorbar(df_combined_results_fibers['fs_ex'][L_sort],df_combined_results_fibers['perm_nm2'][L_sort],yerr=df_combined_results_fibers['perm_nm2_std'][L_sort],xerr=df_combined_results_fibers['fs_ex_std'][L_sort],label='3D-random F-actin',color=L_color[2],capsize=3)
# L_sort=np.argsort(df_combined_results_mixed['fs_oc'])
# ax.errorbar(df_combined_results_mixed['fs_ex'][L_sort],df_combined_results_mixed['perm_nm2'][L_sort],yerr=df_combined_results_mixed['perm_nm2_std'][L_sort],xerr=df_combined_results_mixed['fs_ex_std'][L_sort],label='3D-random Mixed',color=L_color[3],capsize=3)

# ax.plot(x_perm_ex_simple_2D,y_perm_oc_simple_2D,'-',label='2D-CD F-actin',color=L_color[0])
# ax.plot(x_perm_ex_simple_3D,y_perm_oc_simple_3D,'-',label='3D-CFC Ribosomes',color=L_color[1])

# #ax.grid()
# ax.set_xlim([0,0.9])
# plt.xticks(np.arange(0,1,0.1))#ax.set_xlim([0,0.9])
# ax.set_ylim([5e-2,5e4])
# ax.set_box_aspect(1)
# plt.xlabel(r'Nano-obstacle excluded fraction   $\epsilon_n$',font=font)
# plt.ylabel(r'Nano-scale permeability   $K_\alpha$ (nm²)',font=font)
# title='perm_ex_allgeometries_2025_05_06'
# plt.yscale('log')

# plt.title(title,fontsize='small')
# plt.legend(title='Geometry',fontsize='small')
# plt.savefig(title+".svg")
# plt.show()

"POROUS HYDRODYNAMIC HINDRANCE - OCCUPIED fraction "
# figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
# fig,ax=plt.subplots(figsize=figsize,dpi=800)
# A=1/9
# R_partic=2.3

# "2D CD F-actin"
# C_HD=R_partic/(y_perm_oc_simple_2D**0.5) 
# Ratio_hd_2DCD=(1+C_HD+A*C_HD**2)**(-1) 
# ax.plot(x_perm_oc_simple_2D,Ratio_hd_2DCD,label='2D-CD F-actin',color=L_color[0])

# "3D CFC Ribosomes"
# C_HD=R_partic/(y_perm_oc_simple_3D**0.5) 
# Ratio_hd_3DCFC=(1+C_HD+A*C_HD**2)**(-1) 
# ax.plot(x_perm_oc_simple_3D,Ratio_hd_3DCFC,label='3D-CFC Ribosomes',color=L_color[1])

# "3D-random F-actin"
# L_sort=np.argsort(df_combined_results_fibers['fs_oc'])
# C_HD=R_partic/(df_combined_results_fibers['perm_nm2']**0.5)      
# Ratio_hd_fibers=(1+C_HD+A*C_HD**2)**(-1)
# #std composée
# C_HD_low=R_partic/((df_combined_results_fibers['perm_nm2']-df_combined_results_fibers['perm_nm2_std'])**0.5)      
# Ratio_hd_fibers_low=(1+C_HD_low+A*C_HD_low**2)**(-1)
# C_HD_high=R_partic/((df_combined_results_fibers['perm_nm2']+df_combined_results_fibers['perm_nm2_std'])**0.5)      
# Ratio_hd_fibers_high=(1+C_HD_high+A*C_HD_high**2)**(-1)
# Ratio_hd_fibers_std=(Ratio_hd_fibers_high-Ratio_hd_fibers_low)/2
# #affichage
# ax.errorbar(df_combined_results_fibers['fs_oc'][L_sort],Ratio_hd_fibers[L_sort],yerr=Ratio_hd_fibers_std[L_sort],xerr=df_combined_results_fibers['fs_oc_std'][L_sort],label='3D-random F-actin',color=L_color[2],capsize=3)

# "3D-random Mixed"
# L_sort=np.argsort(df_combined_results_mixed['fs_oc'])
# C_HD=R_partic/(df_combined_results_mixed['perm_nm2']**0.5)      
# Ratio_hd_mixed=(1+C_HD+A*C_HD**2)**(-1)
# #std composée
# C_HD_low=R_partic/((df_combined_results_mixed['perm_nm2']-df_combined_results_mixed['perm_nm2_std'])**0.5)      
# Ratio_hd_mixed_low=(1+C_HD_low+A*C_HD_low**2)**(-1)
# C_HD_high=R_partic/((df_combined_results_mixed['perm_nm2']+df_combined_results_mixed['perm_nm2_std'])**0.5)      
# Ratio_hd_mixed_high=(1+C_HD_high+A*C_HD_high**2)**(-1)
# Ratio_hd_mixed_std=(Ratio_hd_mixed_high-Ratio_hd_mixed_low)/2
# #affichage
# ax.errorbar(df_combined_results_mixed['fs_oc'][L_sort],Ratio_hd_mixed[L_sort],yerr=Ratio_hd_mixed_std[L_sort],xerr=df_combined_results_mixed['fs_oc_std'][L_sort],label='3D-random Mixed',color=L_color[3],capsize=3)
# #ax.grid()
# ax.set_xlim([0,0.9])
# plt.xticks(np.arange(0,1,0.1))#ax.set_xlim([0,0.5])
# ax.set_ylim([0,1])
# ax.set_box_aspect(1)

# plt.xlabel(r'Nano-obstacle occupied fraction   ',font=font)
# plt.ylabel(r'Relative diffusivity $D_\gamma^{HDH}$/$D_\alpha$',font=font)
# title='HDH_oc_allgeometries_2025_05_06'
# plt.title(title,fontsize='small')
# plt.legend(title='Geometry',fontsize='small')
# plt.savefig(title+".svg")
# plt.show()

"POROUS HYDRODYNAMIC HINDRANCE - EXCLUDED fraction "
figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
fig,ax=plt.subplots(figsize=figsize,dpi=800)
A=1/9
R_partic=2.3

#,capsize=3,linewidth=1,marker='<',ms=5,zorder=1),capsize=3,linewidth=1,marker='s',ms=4,zorder=0),linewidth=1,marker='o',markevery=10,ms=5),linewidth=1,marker='^',markevery=3,ms=5)


"2D CD F-actin"
C_HD=R_partic/(y_perm_oc_simple_2D**0.5) 
Ratio_hd_2DCD=(1+C_HD+A*C_HD**2)**(-1) 
ax.plot(x_perm_ex_simple_2D,Ratio_hd_2DCD,label='2D-CD F-actin',color=L_color[0],linewidth=1,marker='o',markevery=10,ms=3,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])

"3D CFC Ribosomes"
C_HD=R_partic/(y_perm_oc_simple_3D**0.5) 
Ratio_hd_3DCFC=(1+C_HD+A*C_HD**2)**(-1) 
ax.plot(x_perm_ex_simple_3D,Ratio_hd_3DCFC,label='3D-CFC Ribosomes',color=L_color[1],linewidth=1,marker='^',markevery=3,ms=3,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])

"3D-random F-actin"
L_sort=np.argsort(df_combined_results_fibers['fs_ex'])
C_HD=R_partic/(df_combined_results_fibers['perm_nm2']**0.5)      
Ratio_hd_fibers=(1+C_HD+A*C_HD**2)**(-1)
#std composée
C_HD_low=R_partic/((df_combined_results_fibers['perm_nm2']-df_combined_results_fibers['perm_nm2_std'])**0.5)      
Ratio_hd_fibers_low=(1+C_HD_low+A*C_HD_low**2)**(-1)
C_HD_high=R_partic/((df_combined_results_fibers['perm_nm2']+df_combined_results_fibers['perm_nm2_std'])**0.5)      
Ratio_hd_fibers_high=(1+C_HD_high+A*C_HD_high**2)**(-1)
Ratio_hd_fibers_std=(Ratio_hd_fibers_high-Ratio_hd_fibers_low)/2
#affichage
ax.errorbar(df_combined_results_fibers['fs_ex'][L_sort],Ratio_hd_fibers[L_sort],yerr=Ratio_hd_fibers_std[L_sort],xerr=df_combined_results_fibers['fs_ex_std'][L_sort],label='3D-random F-actin',color=L_color[2],capsize=3,linewidth=1,marker='<',ms=3,zorder=1,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])

"3D-random Mixed"
L_sort=np.argsort(df_combined_results_mixed['fs_ex'])
C_HD=R_partic/(df_combined_results_mixed['perm_nm2']**0.5)      
Ratio_hd_mixed=(1+C_HD+A*C_HD**2)**(-1)
#std composée
C_HD_low=R_partic/((df_combined_results_mixed['perm_nm2']-df_combined_results_mixed['perm_nm2_std'])**0.5)      
Ratio_hd_mixed_low=(1+C_HD_low+A*C_HD_low**2)**(-1)
C_HD_high=R_partic/((df_combined_results_mixed['perm_nm2']+df_combined_results_mixed['perm_nm2_std'])**0.5)      
Ratio_hd_mixed_high=(1+C_HD_high+A*C_HD_high**2)**(-1)
Ratio_hd_mixed_std=(Ratio_hd_mixed_high-Ratio_hd_mixed_low)/2
#affichage
ax.errorbar(df_combined_results_mixed['fs_ex'][L_sort],Ratio_hd_mixed[L_sort],yerr=Ratio_hd_mixed_std[L_sort],xerr=df_combined_results_mixed['fs_ex_std'][L_sort],label='3D-random Mixed',color=L_color[3],capsize=3,linewidth=1,marker='s',ms=2.5,zorder=0,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])
#ax.grid()
ax.set_xlim([0,0.9])
plt.xticks(np.arange(0,1,0.2))#ax.set_xlim([0,0.9])
ax.set_ylim([0,1])
ax.set_box_aspect(1)


plt.xlabel(r'Nano-obs excluded fraction   $\epsilon_n$',font=font)
plt.ylabel(r'Nano-obs Porous. Hydro. Hind. ',font=font)
# title = 'HDH_ex_allgeometries_2025_05_06'
# plt.title(title, fontsize='small')
#plt.legend(title='Geometry',fontsize='small')
#plt.savefig(title+".svg")
plt.show()


"NANO-obstacle HINDRANCE - OCCUPIED fraction"
# figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
# fig,ax=plt.subplots(figsize=figsize,dpi=800)

# "2D CD F-actin"
# x_t,y_t,x_hd,y_hd=x_t_oc_simple_2D,1-y_t_ex_simple_2D,x_perm_oc_simple_2D,Ratio_hd_2DCD
# lim=min(max(x_t),max(x_hd))
# x_t_shortened=x_t[x_t<=lim]
# x_hd_shortened=x_hd[x_hd<=lim]
# x_cb=np.sort(np.concatenate((x_t_shortened,x_hd_shortened)))
# y_hd_interp=np.interp(x_cb,x_hd,y_hd)
# y_t_interp=np.interp(x_cb,x_t,y_t)
# y_cb=y_hd_interp*y_t_interp
# ax.plot(x_cb,y_cb,color=L_color[0],label="2D-CD F-actin")

# "3D CFC Ribosomes"
# x_t,y_t,x_hd,y_hd=x_t_oc_simple_3D,1-y_t_ex_simple_3D,x_perm_oc_simple_3D,Ratio_hd_3DCFC
# lim=min(max(x_t),max(x_hd))
# x_t_shortened=x_t[x_t<=lim]
# x_hd_shortened=x_hd[x_hd<=lim]
# x_cb=np.sort(np.concatenate((x_t_shortened,x_hd_shortened)))
# y_hd_interp=np.interp(x_cb,x_hd,y_hd)
# y_t_interp=np.interp(x_cb,x_t,y_t)
# y_cb=y_hd_interp*y_t_interp
# ax.plot(x_cb,y_cb,color=L_color[1],label="3D-CFC Ribosomes")

# "3D-random F-actin"
# L_sort=np.argsort(df_combined_results_fibers['fs_oc'])
# y_cb_fibers=df_combined_results_fibers['tort'][L_sort]*Ratio_hd_fibers[L_sort]
# #std composée
# y_cb_fibers_high=(df_combined_results_fibers['tort']+df_combined_results_fibers['tort_std'])*Ratio_hd_fibers_high
# y_cb_fibers_low=(df_combined_results_fibers['tort']-df_combined_results_fibers['tort_std'])*Ratio_hd_fibers_low
# y_cb_fibers_std=(y_cb_fibers_high-y_cb_fibers_low)/2
# #affichage
# ax.errorbar(df_combined_results_fibers['fs_oc'][L_sort],y_cb_fibers[L_sort],yerr=y_cb_fibers_std[L_sort],xerr=df_combined_results_fibers['fs_oc_std'][L_sort],label='3D-random F-actin',color=L_color[2],capsize=3)

# "3D-random Mixed"
# L_sort=np.argsort(df_combined_results_mixed['fs_oc'])
# y_cb_mixed=df_combined_results_mixed['tort'][L_sort]*Ratio_hd_mixed[L_sort]
# #std composée
# y_cb_mixed_high=(df_combined_results_mixed['tort']+df_combined_results_mixed['tort_std'])*Ratio_hd_mixed_high
# y_cb_mixed_low=(df_combined_results_mixed['tort']-df_combined_results_mixed['tort_std'])*Ratio_hd_mixed_low
# y_cb_mixed_std=(y_cb_mixed_high-y_cb_mixed_low)/2
# #affichage
# ax.errorbar(df_combined_results_mixed['fs_oc'][L_sort],y_cb_mixed[L_sort],yerr=y_cb_mixed_std[L_sort],xerr=df_combined_results_mixed['fs_oc_std'][L_sort],label='3D-random Mixed',color=L_color[3],capsize=3)

# #ax.grid()
# ax.set_xlim([0,0.9])
# plt.xticks(np.arange(0,1,0.1))#ax.set_xlim([0,0.5])
# ax.set_ylim([0,1])
# ax.set_box_aspect(1)

# # ax.set_aspect('equal')
# plt.xlabel(r'Nano-obstacle occupied fraction   ',font=font)
# plt.ylabel(r'Relative diffusivity $D_\gamma$/$D_\alpha$',font=font)
# title='CB_oc_allgeometries_2025_05_06'
# plt.title(title,fontsize='small')
# plt.legend(title='Geometry',fontsize='small')
# plt.savefig(title+".svg")



"NANO-obstacle HINDRANCE - EXCLUDED fraction"
#,capsize=3,linewidth=1,marker='<',ms=5,zorder=1),capsize=3,linewidth=1,marker='s',ms=4,zorder=0),linewidth=1,marker='o',markevery=10,ms=5),linewidth=1,marker='^',markevery=3,ms=5)
figsize=[5/2.54,5/2.54]#[3/2.54,5/2.54]
fig,ax=plt.subplots(figsize=figsize,dpi=800)

"2D CD F-actin"
x_t,y_t,x_hd,y_hd=x_t_ex_simple_2D,1-y_t_ex_simple_2D,x_perm_ex_simple_2D,Ratio_hd_2DCD
lim=min(max(x_t),max(x_hd))
x_t_shortened=x_t[x_t<=lim]
x_hd_shortened=x_hd[x_hd<=lim]
x_cb_2DCD=np.sort(np.concatenate((x_t_shortened,x_hd_shortened)))
y_hd_interp=np.interp(x_cb_2DCD,x_hd,y_hd)
y_t_interp=np.interp(x_cb_2DCD,x_t,y_t)
y_cb_2DCD=y_hd_interp*y_t_interp
ax.plot(x_cb_2DCD,y_cb_2DCD,color=L_color[0],label="2D-CD F-actin",linewidth=1,marker='o',markevery=10,ms=3,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])

"3D CFC Ribosomes"
x_t,y_t,x_hd,y_hd=x_t_ex_simple_3D,1-y_t_ex_simple_3D,x_perm_ex_simple_3D,Ratio_hd_3DCFC
lim=min(max(x_t),max(x_hd))
x_t_shortened=x_t[x_t<=lim]
x_hd_shortened=x_hd[x_hd<=lim]
x_cb_3DCFC=np.sort(np.concatenate((x_t_shortened,x_hd_shortened)))
y_hd_interp=np.interp(x_cb_3DCFC,x_hd,y_hd)
y_t_interp=np.interp(x_cb_3DCFC,x_t,y_t)
y_cb_3DCFC=y_hd_interp*y_t_interp
ax.plot(x_cb_3DCFC,y_cb_3DCFC,color=L_color[1],label="3D-CFC Ribosomes",linewidth=1,marker='^',markevery=3,ms=3,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])

"3D-random F-actin"
L_sort=np.argsort(df_combined_results_fibers['fs_ex'])
y_cb_fibers=df_combined_results_fibers['tort'][L_sort]*Ratio_hd_fibers[L_sort]
#std composée
y_cb_fibers_high=(df_combined_results_fibers['tort']+df_combined_results_fibers['tort_std'])*Ratio_hd_fibers_high
y_cb_fibers_low=(df_combined_results_fibers['tort']-df_combined_results_fibers['tort_std'])*Ratio_hd_fibers_low
y_cb_fibers_std=(y_cb_fibers_high-y_cb_fibers_low)/2
#affichage
ax.errorbar(df_combined_results_fibers['fs_ex'][L_sort],y_cb_fibers[L_sort],yerr=y_cb_fibers_std[L_sort],xerr=df_combined_results_fibers['fs_ex_std'][L_sort],label='3D-random F-actin',color=L_color[2],capsize=3,linewidth=1,marker='<',ms=3,zorder=1,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])

"3D-random Mixed"
L_sort=np.argsort(df_combined_results_mixed['fs_ex'])
y_cb_mixed=df_combined_results_mixed['tort'][L_sort]*Ratio_hd_mixed[L_sort]
#std composée
y_cb_mixed_high=(df_combined_results_mixed['tort']+df_combined_results_mixed['tort_std'])*Ratio_hd_mixed_high
y_cb_mixed_low=(df_combined_results_mixed['tort']-df_combined_results_mixed['tort_std'])*Ratio_hd_mixed_low
y_cb_mixed_std=(y_cb_mixed_high-y_cb_mixed_low)/2
#affichage
ax.errorbar(df_combined_results_mixed['fs_ex'][L_sort],y_cb_mixed[L_sort],yerr=y_cb_mixed_std[L_sort],xerr=df_combined_results_mixed['fs_ex_std'][L_sort],label='3D-random Mixed',color=L_color[3],capsize=3,linewidth=1,marker='s',ms=2.5,zorder=0,markeredgewidth=0.6, markeredgecolor=[0.3,0.3,0.3])

#ax.grid()
ax.set_xlim([0,0.9])
plt.xticks(np.arange(0,1,0.2))#ax.set_xlim([0,0.9])
ax.set_ylim([0,1])
# ax.set_aspect('equal')
ax.set_box_aspect(1)

plt.xlabel(r'Nano-obs excluded fraction   $\epsilon_n$',font=font)
plt.ylabel(r'Nano-obs total hindrance',font=font)
# title='CB_ex_allgeometries_2025_05_06'
# plt.title(title,fontsize='small')
#plt.legend(title='Geometry',fontsize='small')
#plt.savefig(title+".svg")
plt.show()

# 
