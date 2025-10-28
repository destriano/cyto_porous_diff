#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CURRENT VERSION WRITTEN BY OLIVIER DESTRIAN IN OCTOBER 2025
PAPER "Cytoplasmic crowding acts as a porous medium reducing macromolecule diffusion"
This code allows to analyse FCS autocorrelation curves obtained from Symphotime
To infer the inverse residence time of GFP in living cells
"""

import pandas as pd
import scipy.optimize as scop
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
from matplotlib.patches import PathPatch
import seaborn as sns


"======================================================="
"=============FUNCTIONS DEFINITION===================="
"======================================================="
def unique_string(array): #Return an array of unique non-NaN values from the input array.
    L_2=[]
    for x in array:
        if x not in L_2 and pd.isnull(x)==0:
            L_2.append(x)
    return np.array(L_2)

def remove_nans(vector,data_type): #Remove NaN values from an array and return a clean array
    L_2=[]
    for x in vector:
        if pd.isnull(x)==0:
            L_2.append(x)
    return np.array(L_2,dtype=data_type)

#function to process the FCS data
def P25_FCS_autocorrelation_averaging_and_fitting(list_fcs_nums,parametres,region_tag,prints=0):
    min_time_range,max_time_range,k0,rho_bd,tau_bd,model=parametres
    if model=='diffusion_simple':
        def function_to_fit(x,rho,tau,cte):
            return cte+rho*((1+x/tau)**-1)*((1+x/(tau*k0**2)))**-0.5
    elif model=='triplet_state':
        
        T_bd=[0,10]
        def function_to_fit(x,rho,tau,cte,T):
            return cte+(1+T*(np.exp(-x/tau_trip)-1))*(rho*((1+x/tau)**-1)*((1+x/(tau*k0**2)))**-0.5)
   
    if prints:
        plt.figure(dpi=400)
    for i_k in range(len(list_fcs_nums)):
        k=list_fcs_nums[i_k]
        x_tag='LSM_'+str(k)+'_Correlation_time_ms'
        y_tag='LSM_'+str(k)+'_G_1e3'
        y_error_tag='LSM_'+str(k)+'_Err_1e3'
        x=df_autocorrelation_curves[x_tag]
        y=df_autocorrelation_curves[y_tag]
        y_error=df_autocorrelation_curves[y_error_tag]
        
        if i_k==0:
            x_moy=copy.deepcopy(x)
            y_moy=copy.deepcopy(y)
            y_error_moy=copy.deepcopy(y_error)
        else:
            x_moy+=x
            y_moy+=y
            y_error_moy+=y_error
        if prints:
            plt.plot(x,y-np.mean(y[(x>5e1)*(x<1e2)]),label=str(k))
    x_moy/=len(list_fcs_nums)
    y_moy/=len(list_fcs_nums)
    y_error_moy/=len(list_fcs_nums)
    
    x_moy_to_fit=x_moy[(x_moy>min_time_range)*(x_moy<max_time_range)]
    y_moy_to_fit=y_moy[(x_moy>min_time_range)*(x_moy<max_time_range)]
    y_error_moy_to_fit=y_error_moy[(x_moy>min_time_range)*(x_moy<max_time_range)]    
    
    x_moy_to_fit=x_moy_to_fit.astype(np.float64)
    y_moy_to_fit=y_moy_to_fit.astype(np.float64)
    y_error_moy_to_fit=y_error_moy_to_fit.astype(np.float64)#*0+1

    "6 - calcul de la fonction fittée"
    if model=='diffusion_simple':
        coefs_fit,pcov=scop.curve_fit(function_to_fit, x_moy_to_fit,y_moy_to_fit,sigma=y_error_moy_to_fit,bounds=([rho_bd[0],tau_bd[0],-1000],[rho_bd[1],tau_bd[1],1000]))
        Y_fit=function_to_fit(x_moy_to_fit,coefs_fit[0],coefs_fit[1],coefs_fit[2])
        
        Y_0=function_to_fit(0,coefs_fit[0],coefs_fit[1],coefs_fit[2])
        Y_inf=function_to_fit(100000,coefs_fit[0],coefs_fit[1],coefs_fit[2])
    elif model=='triplet_state':
        coefs_fit,pcov=scop.curve_fit(function_to_fit, x_moy_to_fit,y_moy_to_fit,sigma=y_error_moy_to_fit,bounds=([rho_bd[0],tau_bd[0],-1000,T_bd[0]],[rho_bd[1],tau_bd[1],1000,T_bd[1]]))
        Y_fit=function_to_fit(x_moy_to_fit,coefs_fit[0],coefs_fit[1],coefs_fit[2],coefs_fit[3])
       
        Y_0=function_to_fit(0,coefs_fit[0],coefs_fit[1],coefs_fit[2],coefs_fit[3])
        Y_inf=function_to_fit(100000,coefs_fit[0],coefs_fit[1],coefs_fit[2],coefs_fit[3])
    tau_ms=coefs_fit[1]

    return tau_ms, x_moy_to_fit,y_moy_to_fit,y_error_moy_to_fit,Y_fit,Y_0,Y_inf

#%%

"=========================================================================="
"===============METADATA===================="
"=========================================================================="
path=''
file="METADATA_FCS_EXAMPLE" #example is one burst of 5 repetitions from a LP region of a single cell
min_file=6 ; max_file=10 #files present in the metadata excel to be processed

#metadata
df_0=pd.read_excel (path+file+'.xlsx',sheet_name='data') 
array_0=pd.DataFrame(df_0).to_numpy()

#autocorrelation curves extracted from symphotime
df_autocorrelation_curves=pd.read_excel (path+file+'.xlsx',sheet_name='Autocorrelation_curves_all')
array_autocorrelation_curves=pd.DataFrame(df_autocorrelation_curves).to_numpy()  

#autocorrelation curves extraction from excel
k=0
for i in range(min_file,max_file+1):
    name_to_add='LSM_'+str(i)
    
    array_autocorrelation_curves[0,k]=name_to_add+'_Correlation_time_ms'
    k+=1
    
    if array_autocorrelation_curves[0,k]=='G(t)[]':
        a=copy.deepcopy(array_autocorrelation_curves[1:,k])
        array_autocorrelation_curves[1:,k]*=1000
    array_autocorrelation_curves[0,k]=name_to_add+'_G_1e3'
    k+=1
    
    if array_autocorrelation_curves[0,k]=='±Err []':
        array_autocorrelation_curves[1:,k]*=1000    
    array_autocorrelation_curves[0,k]=name_to_add+'_Err_1e3'
    k+=1
    
df_autocorrelation_curves=pd.DataFrame(data=array_autocorrelation_curves[1:,:],columns=array_autocorrelation_curves[0,:])





#%%
#FCS numerical parameters
tau_trip=20e-3 #in ms, only taken into account if triplet sate method enabled
min_time_range=1e-2 #in ms, min time delay to be processed
max_time_range=1e2 #in ms, max time delay to be processed
model='diffusion_simple'#diffusion_simple or triplet_state model

k0=3.6#confocal volume axial to transversal ratio
rho_bd=[0,100] #fit parameters boundaries
tau_bd=[0.01,2]#fit parameters boundaries

parametres=(min_time_range,max_time_range,k0,rho_bd,tau_bd,model)

prints=1

"=========================================================================="
"====================== FCS ANALYSIS ==============================="
"=========================================================================="
L_effet,L_cellule,L_zone,L_iteration,L_tau,L_tau_error=[],[],[],[],[],[]

#Import data parameters
array_effet=pd.DataFrame(df_0['effet']).to_numpy() #list of effects to be analysed
array_unique_effet=unique_string(array_effet) #unique list of effects to be analysed
array_cell=pd.DataFrame(df_0['cell']).to_numpy() #list of cells to be analysed
array_unique_cell=remove_nans(unique_string(array_cell),'int')
array_zone=pd.DataFrame(df_0['zone']).to_numpy() #list of intracellular regions to be analysed
array_nom=pd.DataFrame(df_0['nom']).to_numpy() #list of names to be analysed
array_config=pd.DataFrame(df_0['config']).to_numpy() #list of configs to be analysed
array_num=pd.DataFrame(df_0['num']).to_numpy()  #list of file numbers to be analysed
array_type=pd.DataFrame(df_0['type']).to_numpy()  #list of data types to be analysed

for i_effet in range (len(array_unique_effet)):
    effet=array_unique_effet[i_effet] 
    for k_cell in array_unique_cell[:]:
        #local metadata extraction
        array_local_zone=array_zone[(array_cell==k_cell)*(array_effet==effet)]
        array_unique_zone=unique_string(array_local_zone)
        array_local_num=array_num[(array_cell==k_cell)*(array_effet==effet)]
        array_local_nom=array_nom[(array_cell==k_cell)*(array_effet==effet)]
        array_local_type=array_type[(array_cell==k_cell)*(array_effet==effet)]

        if prints:
            i_zone=0
            plt.figure(figsize=[5/2.54,5/2.54],dpi=400)
        for k_zone in array_unique_zone:
            
            print('cell :'+str(k_cell)+', zone : '+str(k_zone))
            array_local_num=array_num[(array_cell==k_cell)*(array_effet==effet)*(array_zone==k_zone)*(array_type=='FCS')]
            
            region_tag=str(k_zone)
            
            tau_diff, x_moy_to_fit,y_moy_to_fit,y_error_moy_to_fit,Y_fit,Y_0,Y_inf=P25_FCS_autocorrelation_averaging_and_fitting(array_local_num,parametres,region_tag,prints=0)
        
            if prints: 
                if str(k_zone)=='LP':
                    i_zone=1
                    tau_lp=tau_diff
                elif str(k_zone)=='HP':
                    tau_hp=tau_diff
                    i_zone=0
                if str(k_zone)=='LP' or str(k_zone)=='HP':
                    L_color_local=['#00ABF0','#FE351A']
                    plt.plot(x_moy_to_fit,(y_moy_to_fit-Y_inf)/(Y_0-Y_inf),'.',label=region_tag+' - Exp',color=L_color_local[i_zone],ms=2.5,zorder=0,alpha=0.6)
                    plt.plot(x_moy_to_fit,(Y_fit-Y_inf)/(Y_0-Y_inf),label=region_tag+' - Fit',color=L_color_local[i_zone],linewidth=0.9,alpha=1.0)
                    plt.xlabel('Correlation delay (ms)')
                    plt.ylabel('Normalized autocorrelation value')
                    plt.xscale('log')
                    plt.ylim([-0.1,1.5])   
                    plt.xlim([1e-2,1e2])
                    plt.legend(fontsize='small')
                        
                        
            L_tau.append(float(tau_diff))
            L_effet.append(effet[0])
            L_cellule.append(k_cell[0])
            L_zone.append(str(k_zone))
                  
                    
