#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CURRENT VERSION WRITTEN BY OLIVIER DESTRIAN IN OCTOBER 2025
PAPER "Cytoplasmic crowding acts as a porous medium reducing macromolecule diffusion"
This code allows to compare experimental data to model predictions
It uses experimental FRAP and segmentation results, and finite element simulations results

Full data used for the study is here used (not a test sample)
"""

import pandas as pd
pd.options.mode.chained_assignment = None

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma

import scipy.optimize as scop
import scipy
from scipy.optimize import curve_fit
from scipy.ndimage import convolve1d

from scipy import interpolate
from scipy import stats

import copy

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import PathPatch

import seaborn as sns

import matplotlib.font_manager
font = {'family' : 'Arial',
         'weight' : 'bold',
         'size'   : 8}

plt.rc('font', **font)
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

"======================================================="
"=============definition des fonctions===================="
"======================================================="

def build_num(num): #sert juste à rajouter un 0 si y'en a pas
    if num<10:
        return '0'+str(int(num))
    else : 
        return str(int(num))
    
def unique_string(array): # permet, dans un char_array (ou pas), de récupérer un charray de valeurs uniques
    L_2=[]
    for x in array:
        if x not in L_2 and pd.isnull(x)==0:
            L_2.append(x)
    return np.array(L_2)
        
def remove_nans(vector,data_type): #permet de faire un charray sans les nans 
    L_2=[]
    for x in vector:
        if pd.isnull(x)==0:
            L_2.append(x)
    return np.array(L_2,dtype=data_type)

L_color=['#377eb8', '#ff7f00', '#4daf4a',
      '#f781bf', '#a65628', '#984ea3',
      '#999999', '#e41a1c', '#dede00'] #color blind colors
L_marker=['s','o','^','x','<','*']
def P25_print_text_statistical_t_test(a,b,text,k):
    _,p=scipy.stats.ttest_ind(a,b,equal_var=False)   
    if p<0.0001:
        plt.text(0.1,0.9-0.05*k,text+str(p),fontsize='x-small')
    elif p<0.001: 
        plt.text(0.1,0.9-0.05*k,text+str(p),fontsize='x-small',color='tab:blue')
    elif p<0.01: 
        plt.text(0.1,0.9-0.05*k,text+str(p),fontsize='x-small',color='tab:green')
    elif p<0.05: 
        plt.text(0.1,0.9-0.05*k,text+str(p),fontsize='x-small',color='tab:orange')
    else :
        plt.text(0.1,0.9-0.05*k,text+str(p),fontsize='x-small',color='tab:red')
def adjust_box_widths(ax, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """
    # iterating through axes artists:
    for c in ax.get_children():

        # searching for PathPatches
        if isinstance(c, PathPatch):
            # getting current width of box:
            p = c.get_path()
            verts = p.vertices
            verts_sub = verts[:-1]
            xmin = np.min(verts_sub[:, 0])
            xmax = np.max(verts_sub[:, 0])
            xmid = 0.5*(xmin+xmax)
            xhalf = 0.5*(xmax - xmin)

            # setting new width of box
            xmin_new = xmid-fac*xhalf
            xmax_new = xmid+fac*xhalf
            verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
            verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

            # setting new width of median line
            for l in ax.lines:
                if np.all(l.get_xdata() == [xmin, xmax]):
                    l.set_xdata([xmin_new, xmax_new])
                    
                    
def P26_boxplot_and_violinplot(array_unique_effet,array_unique_zone,df_combined_results,x_var,y_var,hue_var,hue_var_label,title,xlim,ylim,xlabel,ylabel,figsize,bars_mode,restriction=None,L_color=L_color):
    if restriction!=None : 
        df_combined_results_mod=df_combined_results[np.in1d(np.array(df_combined_results[restriction[0]]), restriction[1])]
    else :
        df_combined_results_mod=df_combined_results
    if hue_var_label is not None:
        df_combined_results_mod=df_combined_results_mod.rename(columns={hue_var: hue_var_label})
        hue_var_ttest='Condition'
    else : 
        hue_var_ttest='cond'
    
    "1===Affichage boxplots==="
    fig, axg = plt.subplots(figsize=figsize,dpi=800)
    plt.title(title,font=font)
    
    if x_var=='region':
        array_order=array_unique_zone
    elif x_var=='cond':
        array_order=array_unique_effet
        
    sns.swarmplot(data=df_combined_results_mod,x=x_var,y=y_var,palette=L_color,alpha=0.8, dodge=False,size=3,legend=0,hue=hue_var_label,order=array_order,zorder=0)   
    for i in range(len(array_order)): 
        group=array_order[i]
        values=df_combined_results_mod[y_var][df_combined_results_mod[x_var]==group]
        if bars_mode=='q1q2q3': #on affiche le point median et les quartiles 1 et 3
            low_bar = np.percentile(values, 25)
            med_bar = np.percentile(values, 50)
            high_bar = np.percentile(values, 75)
        elif bars_mode=='meanstd': #on affiche la moyenne et +- standard deviation --> plus cohérent avec la partie modèle
            med_bar=np.mean(values)
            low_bar=med_bar-np.std(values)
            high_bar=med_bar+np.std(values)        
        plt.vlines(x=i, ymin=low_bar, ymax=high_bar, color="black", linewidth=1)
        plt.hlines(y=low_bar, xmin=i-0.2, xmax=i+0.2, color="black", linewidth=1)
        plt.hlines(y=high_bar, xmin=i-0.2, xmax=i+0.2, color="black", linewidth=1)
        plt.hlines(y=med_bar, xmin=i-0.4, xmax=i+0.4, color="black", linewidth=1.5)
            
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.ylabel(ylabel,font=font)
    plt.xlabel(xlabel,font=font)   
  
    "3===Affichage t-tests==="
    plt.figure(dpi=400)
    plt.title(title)
    k=1
    "AFFICHAGE ISO-CONDITION"
    for i_cond in range(len(array_unique_effet)) :
        if len(array_unique_zone)==2:
            a=df_combined_results_mod[y_var][(df_combined_results_mod[hue_var_ttest]==array_unique_effet[i_cond])*(df_combined_results_mod['region']=='HP')]
            b=df_combined_results_mod[y_var][(df_combined_results_mod[hue_var_ttest]==array_unique_effet[i_cond])*(df_combined_results_mod['region']=='LP')]
            P25_print_text_statistical_t_test(a,b,array_unique_effet[i_cond]+', HP VS LP, p=',k)      
            k+=1        
        else :
            print("Problème dans 2===Affichage t-tests=== : le nombre de régions n'est pas de 2")
    "AFFICHAGE ISO-REGION"
    k+=1 
    for i_region in range(len(array_unique_zone)):
        for i_cond in range(len(array_unique_effet)) :
            a=df_combined_results_mod[y_var][(df_combined_results_mod[hue_var_ttest]==array_unique_effet[i_cond])*(df_combined_results_mod['region']==array_unique_zone[i_region])]
            b=df_combined_results_mod[y_var][(df_combined_results_mod[hue_var_ttest]==array_unique_effet[i_cond-1])*(df_combined_results_mod['region']==array_unique_zone[i_region])]
            P25_print_text_statistical_t_test(a,b,array_unique_zone[i_region]+', '+array_unique_effet[i_cond]+' VS '+array_unique_effet[i_cond-1]+', p=',k)
            k+=1
    "AFFICHAGE ISO-REGION - comparé au control"        
    k+=1 
    for i_region in range(len(array_unique_zone)):
        for i_cond in range(1,len(array_unique_effet)) :
            a=df_combined_results_mod[y_var][(df_combined_results_mod[hue_var_ttest]==array_unique_effet[i_cond])*(df_combined_results_mod['region']==array_unique_zone[i_region])]
            b=df_combined_results_mod[y_var][(df_combined_results_mod[hue_var_ttest]==array_unique_effet[0])*(df_combined_results_mod['region']==array_unique_zone[i_region])]
            P25_print_text_statistical_t_test(a,b,array_unique_zone[i_region]+', '+array_unique_effet[i_cond]+' VS '+array_unique_effet[0]+', p=',k)
            k+=1
    plt.ylim([-0.5,1])
    plt.xlabel('black****, blue***, green**, orange*, red n.s.')
    plt.show()
   
def P26_boxplot_and_violinplot_ratioLPHP(array_unique_effet,array_unique_zone,df_combined_results,x_var,y_var,hue_var,title,xlabel,ylabel,hue_var_label,ylim,figsize,bars_mode,L_color):
    m_ratio=np.zeros((0,2),dtype='object')
    for i_cond in range(len(array_unique_effet)) :
        num=np.array(df_combined_results[y_var][(df_combined_results['cond']==array_unique_effet[i_cond])*(df_combined_results['region']=='LP')])
        denom=np.array(df_combined_results[y_var][(df_combined_results['cond']==array_unique_effet[i_cond])*(df_combined_results['region']=='HP')])
        ratio=np.divide(num,denom)
        m_to_add=np.array(np.zeros((ratio.shape[0],2)),dtype='object')
        m_to_add[:,1]=ratio
        m_to_add[:,0]=array_unique_effet[i_cond]
        m_ratio=np.concatenate((m_ratio,m_to_add),axis=0)    
    df_ratio=pd.DataFrame(columns=['cond','ratio'],data=m_ratio)
    df_ratio['ratio']=df_ratio['ratio'].apply(pd.to_numeric)

    "1===Affichage boxplots==="
    fig, axg = plt.subplots(figsize=figsize,dpi=800)
    plt.title(title,font=font)

    if x_var=='region':
        array_order=array_unique_zone
    elif x_var=='cond':
        array_order=array_unique_effet
        
    sns.swarmplot(data=df_ratio,x='cond',y='ratio',palette=L_color,alpha=0.8, dodge=False,size=3,legend=0,hue=hue_var_label,order=array_order,zorder=0)
    for i in range(len(array_order)):
        group=array_order[i]
        values=df_ratio['ratio'][df_ratio['cond']==group]     
        if bars_mode=='q1q2q3': #on affiche le point median et les quartiles 1 et 3
            low_bar = np.percentile(values, 25)
            med_bar = np.percentile(values, 50)
            high_bar = np.percentile(values, 75)
        elif bars_mode=='meanstd': #on affiche la moyenne et +- standard deviation --> plus cohérent avec la partie modèle
            med_bar=np.mean(values)
            low_bar=med_bar-np.std(values)
            high_bar=med_bar+np.std(values)         
        plt.vlines(x=i, ymin=low_bar, ymax=high_bar, color="black", linewidth=1)
        plt.hlines(y=low_bar, xmin=i-0.2, xmax=i+0.2, color="black", linewidth=1)
        plt.hlines(y=high_bar, xmin=i-0.2, xmax=i+0.2, color="black", linewidth=1)
        plt.hlines(y=med_bar, xmin=i-0.4, xmax=i+0.4, color="black", linewidth=1.5)

    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.ylabel(ylabel,font=font)
    plt.xlabel(xlabel,font=font)   
 
    "AFFICHAGE cond VS cond-1"
    plt.figure(dpi=400)
    plt.title(title)
    k=1
    for i_cond in range(len(array_unique_effet)) :
        a=df_ratio['ratio'][(df_ratio['cond']==array_unique_effet[i_cond])]
        b=df_ratio['ratio'][(df_ratio['cond']==array_unique_effet[i_cond-1])]
        P25_print_text_statistical_t_test(a,b,array_unique_effet[i_cond]+' VS '+array_unique_effet[i_cond-1]+', p=',k)
        k+=1
    "AFFICHAGE par rapport à ISO"
    k+=1
    for i_cond in range(len(array_unique_effet)) :
        a=df_ratio['ratio'][(df_ratio['cond']==array_unique_effet[i_cond])]
        b=df_ratio['ratio'][(df_ratio['cond']=='300')]
        P25_print_text_statistical_t_test(a,b,array_unique_effet[i_cond]+' VS ISO, p=',k)
        k+=1
    plt.ylim([-0.5,1])
    plt.xlabel('black****, blue***, green**, orange*, red n.s.')
    plt.show()

def P25_boxplot_and_violinplot_diffLPHP(array_unique_effet,array_unique_zone,df_combined_results,x_var,y_var,hue_var,title,xlabel,ylabel,hue_var_label,ylim):
    m_ratio=np.zeros((0,2),dtype='object')
    for i_cond in range(len(array_unique_effet)) :
        num=np.array(df_combined_results[y_var][(df_combined_results['cond']==array_unique_effet[i_cond])*(df_combined_results['region']=='LP')])
        denom=np.array(df_combined_results[y_var][(df_combined_results['cond']==array_unique_effet[i_cond])*(df_combined_results['region']=='HP')])
        ratio=np.subtract(num,denom)
        m_to_add=np.array(np.zeros((ratio.shape[0],2)),dtype='object')
        m_to_add[:,1]=ratio
        m_to_add[:,0]=array_unique_effet[i_cond]
        m_ratio=np.concatenate((m_ratio,m_to_add),axis=0)    
    df_ratio=pd.DataFrame(columns=['cond','ratio'],data=m_ratio)
    df_ratio['ratio']=df_ratio['ratio'].apply(pd.to_numeric)
 
    "affichage"
    fig, ax = plt.subplots(figsize=[9,6],dpi=400)
    ax.set_axisbelow(True)
    plt.grid('horizontal')
    ax2=sns.boxplot(data=df_ratio,x='cond',y='ratio',showfliers=False,meanline=True, showmeans=True,boxprops={"alpha": .85},meanprops={"color": "black", "linewidth": 2},medianprops={"color": "black", "linewidth": 1},order=array_unique_effet)
    sns.swarmplot(data=df_ratio,x='cond',y='ratio',color='k', dodge=True,size=5,legend=0,alpha=0.9,order=array_unique_effet)#,size=5)
    ax2.set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)
    plt.ylim(ylim)
    
    
    "2===Application d'un pattern pour colorblinds==="
    # hatches must equal the number of hues (3 in this case)
    global_hatches=['','/', 'x',"\ ",'//','xx','\\']
    if hue_var_label is not None:
        hatches = global_hatches[:unique_string(df_combined_results[hue_var_label]).size]
    else :
        hatches = global_hatches[:unique_string(df_combined_results[x_var]).size]
    
    # select the correct patches
    patches = [patch for patch in ax2.patches if type(patch) == mpl.patches.PathPatch]
    # the number of patches should be evenly divisible by the number of hatches
    h = hatches * (len(patches) // len(hatches))
    # iterate through the patches for each subplot
    for patch, hatch in zip(patches, h):
        patch.set_hatch(hatch)
        patch.set_edgecolor('darkslategray')
    
    "AFFICHAGE cond VS cond-1"
    plt.figure(dpi=400)
    plt.title(title)
    k=1
    for i_cond in range(len(array_unique_effet)) :
        a=df_ratio['ratio'][(df_ratio['cond']==array_unique_effet[i_cond])]
        b=df_ratio['ratio'][(df_ratio['cond']==array_unique_effet[i_cond-1])]
        P25_print_text_statistical_t_test(a,b,array_unique_effet[i_cond]+' VS '+array_unique_effet[i_cond-1]+', p=',k)
        k+=1
    "AFFICHAGE par rapport à ISO"
    k+=1
    for i_cond in range(len(array_unique_effet)) :
        a=df_ratio['ratio'][(df_ratio['cond']==array_unique_effet[i_cond])]
        b=df_ratio['ratio'][(df_ratio['cond']=='300')]
        P25_print_text_statistical_t_test(a,b,array_unique_effet[i_cond]+' VS ISO, p=',k)
        k+=1
    plt.ylim([-0.5,1])
    plt.xlabel('black****, blue***, green**, orange*, red n.s.')        



#%%
"file loading"
"results from FRAP processing, stored in a char array"
charray_frap_results_import=np.load('charray_frap_all_results_2025_10_29.npy')
#charray_frap_results_import format
#line 0 : osmolarity
#line 1 : cell unique number
#line 2 : intracellular region type
#line 3 : gfp diffusivity in um2s-1


"results from SEGM processing, stored in a char array"
charray_segm_results_import=np.load('charray_segm_all_results_2025_10_29.npy')
#charray_segm_results_import format
#line 0 : osmolarity
#line 1 : cell unique number
#line 2 : intracellular region type (if cell: it means only cell volume computation was performed,
#which was always the case for measuring cell volume before osmotic shock for cells going to be studied after osmotic shock)
#line 3 : cell volume in um3
#line 5: OBSOLETE 
#line 6: OBSOLETE 
#line 7: Micro-obstacle fraction 
#line 9: Intermediate variable for porosity calculation (in accessible volume fraction)
#line 10: OBSOLETE
#line 11: Intermediate variable for nano-obstacle excluded volume computation

"---création du charray_combined qui combine les résultats FRAP et SEGM---"
N_l_frap=charray_frap_results_import.shape[0]
N_l_segm=charray_segm_results_import.shape[0]-2#-3 pour enlever la redondance des infos cellule : effet, n_cell, zone mais 1+ pour rajouter 'cell_vol'
charray_combined_results=np.zeros((N_l_frap+N_l_segm,0),dtype='object') #définition du charray combined

"liste des effets à étudier dans les corrélations et les fits théoriques"
array_unique_effet_tostudy=np.array(['300','450','600','900'])
"liste des effets/cell/zones présents dans les datasets"
array_effet=charray_frap_results_import[0,:]
array_cell=charray_frap_results_import[1,:]
array_zone=charray_frap_results_import[2,:]
M_cell_vol=charray_segm_results_import[3,:]

for i_effet in range (len(array_unique_effet_tostudy)):
    effet=array_unique_effet_tostudy[i_effet]    
    array_local_cell_unique=unique_string(array_cell[(array_effet==effet)])
      
    for i_cell in range(len(array_local_cell_unique)) :
        k_cell=array_local_cell_unique[i_cell]
        array_local_zone_unique=unique_string(array_zone[(array_cell==k_cell)*(array_effet==effet)])
        
        for i_zone in range(len(array_local_zone_unique)) :
            k_zone=array_local_zone_unique[i_zone]
            
            "recherche d'une correspondance entre le dataset (cellule,zone,effet) du charray_frap et celui du charray_segm"
            i_correspondance_frap=(charray_frap_results_import[0,:]==effet)*(charray_frap_results_import[1,:]==str(int(k_cell)))*(charray_frap_results_import[2,:]==k_zone)
            i_correspondance_segm=(charray_segm_results_import[0,:]==effet)*(charray_segm_results_import[1,:]==str(int(k_cell)))*(charray_segm_results_import[2,:]==k_zone)
            "si pas de correspondance : les données ne sont pas récupérées"
            if True in i_correspondance_frap:
                charray_combined_results_sub=np.concatenate((charray_frap_results_import[:,i_correspondance_frap],charray_segm_results_import[2:,i_correspondance_segm]))
                charray_combined_results_sub[N_l_frap,0]='rel_cell_vol'
                if effet=='300': #pas de calcul du volume relatif
                    charray_combined_results_sub[5,0]=1.0
                else : #récupération du volume pour la condition ISO et l'autre condition. [0] car ya deux zones : LP/HP (même cellule donc même volume)
                    if True in (charray_segm_results_import[0,:]=='300')*(charray_segm_results_import[1,:]==str(int(k_cell))) :
                        vol_iso=np.float64(M_cell_vol[(charray_segm_results_import[0,:]=='300')*(charray_segm_results_import[1,:]==str(int(k_cell)))][0])

                    vol_noniso=np.float64(M_cell_vol[(charray_segm_results_import[0,:]==effet)*(charray_segm_results_import[1,:]==str(int(k_cell)))][0])                 
                    charray_combined_results_sub[5,0]=vol_noniso/vol_iso
                #on ajoute à chaque fois les nouvelles données au charray_combined
                charray_combined_results=np.concatenate((charray_combined_results,charray_combined_results_sub),axis=1)


"=========================================================================="
"=============== DATAFRAME CREATION ======================"
"=========================================================================="  

"création d'une version dataframe"
df_combined_results= pd.DataFrame(np.transpose(charray_combined_results), columns=['cond','cell','region','diff','','vol_r','','bin_poro','bin_nano','bin_micro','','prop_poro','prop_nano','prop_micro'])
df_combined_results[['cell','diff','vol_r','bin_poro','bin_nano','bin_micro','prop_poro','prop_nano','prop_micro']] = df_combined_results[['cell','diff','vol_r','bin_poro','bin_nano','bin_micro','prop_poro','prop_nano','prop_micro']].apply(pd.to_numeric) 

  
"===AJOUT DE NOUVELLES VARIABLES ====="
"OBSOLETE NANO-OBSTACLE COMPUTATION METHOD CALLED PROPORTIONAL"
variable=np.array(df_combined_results['prop_nano'],dtype=float)+np.array(df_combined_results['prop_micro'],dtype=float)
df_variable=pd.DataFrame(columns=['prop_s_tot'], data=variable)
df_combined_results=pd.concat([df_combined_results,df_variable],axis=1)


"OBSOLETE TOTAL OBTACLE FRACTION COMPUTATION METHOD CALLED PROPORTIONAL"
variable=np.array(df_combined_results['bin_nano'],dtype=float)+np.array(df_combined_results['bin_micro'],dtype=float)
df_variable=pd.DataFrame(columns=['bin_s_tot'], data=variable)
df_combined_results=pd.concat([df_combined_results,df_variable],axis=1)

"GOOD COMPUTATION METHOD"
variable=1-np.array(df_combined_results['prop_poro'],dtype=float)-np.array(df_combined_results['bin_micro'],dtype=float)
df_variable=pd.DataFrame(columns=['bin_cor_nano'], data=variable)
df_combined_results=pd.concat([df_combined_results,df_variable],axis=1)


'ratio nano-total - proportional'
variable=np.divide(np.array(df_combined_results['prop_nano'],dtype=float),np.array(df_combined_results['prop_s_tot'],dtype=float))
df_variable=pd.DataFrame(columns=['prop_ratio_nano_tot'], data=variable)
df_combined_results=pd.concat([df_combined_results,df_variable],axis=1)

'ratio nano-total - binary'
variable=np.divide(np.array(df_combined_results['bin_nano'],dtype=float),np.array(df_combined_results['bin_s_tot'],dtype=float))
df_variable=pd.DataFrame(columns=['bin_ratio_nano_tot'], data=variable)
df_combined_results=pd.concat([df_combined_results,df_variable],axis=1)

'ratio nano-total - binary corrected'
variable=np.divide(np.array(df_combined_results['bin_cor_nano'],dtype=float),np.array(df_combined_results['prop_s_tot'],dtype=float))
df_variable=pd.DataFrame(columns=['bin_cor_ratio_nano_tot'], data=variable)
df_combined_results=pd.concat([df_combined_results,df_variable],axis=1)

"evolution viscosité cytosolic modèle solvant idéal"
L_k=[0,0.5,1,1.5,2]#[0,0.5,1,1.5,2,-0.5,-1,-2]
def relative_cytosolic_viscosity(rel_vol,k):
    return (1+k*(1/rel_vol))/(1+k)
for k in L_k:
    variable=relative_cytosolic_viscosity(np.array(df_combined_results['vol_r'],dtype=float),k)
    df_variable=pd.DataFrame(columns=['eta_r_k='+str(k)], data=variable)
    df_combined_results=pd.concat([df_combined_results,df_variable],axis=1)

    
sorting='zone' #'zone' manière d'agencer les boites à moustaches sur le BOXPLOT
array_unique_effet=['300','450','600','900']
array_unique_zone=['HP','LP']

"=========================================="
"D GFP boxplot des données frap "
"=========================================="
bars_mode='meanstd' #horizontal bars in data plotting correspond to mean value and +- standard deviations

figsize=[5/2.54,5/2.54]
x_var='cond'; y_var='diff' ; hue_var=None
xlabel='Osmolarity [mOsm]'; ylabel=r'GFP diffusivity [$\mu m^2 s^{-1}$] '  ; hue_var_label=None
xlim=[-0.5,3.5];ylim=[-0,30]
title='d gfp absolu LP'+bars_mode
restriction=('region',['LP'])
L_color_local=['#FE351A','#EF1D01','#C71901','#9F1401']
P26_boxplot_and_violinplot(array_unique_effet,array_unique_zone,df_combined_results,x_var,y_var,hue_var,hue_var_label,title,xlim,ylim,xlabel,ylabel,figsize,bars_mode,restriction=restriction,L_color=L_color_local)

figsize=[5/2.54,5/2.54]
x_var='cond'; y_var='diff' ; hue_var=None
xlabel='Osmolarity [mOsm]'; ylabel=r'GFP diffusivity [$\mu m^2 s^{-1}$] '  ; hue_var_label=None
xlim=[-0.5,3.5];ylim=[-0,30]
title='d gfp absolu HP'+bars_mode
restriction=('region',['HP'])
L_color_local=['#00ABF0','#008FC8','#0072A0','#005678']
P26_boxplot_and_violinplot(array_unique_effet,array_unique_zone,df_combined_results,x_var,y_var,hue_var,hue_var_label,title,xlim,ylim,xlabel,ylabel,figsize,bars_mode,restriction=restriction,L_color=L_color_local)

"=========================================="
"D GFP RATIO affichage ratio pour chaque cellule LP/HP"
"=========================================="
"median et q1q3"
x_var='cond'; y_var='diff' ; hue_var='' ; 
xlabel='Osmolarity [mOsm]'; ylabel='Relative diffusivity'  ; hue_var_label=None
xlim=[-0.5,3.5];ylim=[0.4,1.4]
L_color_local=['#8C8C8C','#787878','#646464','#505050']
title='d gfp relatif LP/HP'+bars_mode
P26_boxplot_and_violinplot_ratioLPHP(array_unique_effet,array_unique_zone,df_combined_results,x_var,y_var,hue_var,title,xlabel,ylabel,hue_var_label,ylim,figsize,bars_mode,L_color=L_color_local)


"t test one variable 2025 04 28"
for cond in np.unique(df_combined_results['cond']):
    num=np.array(df_combined_results['diff'][(df_combined_results['cond']==cond)*(df_combined_results['region']=='LP')])
    denom=np.array(df_combined_results['diff'][(df_combined_results['cond']==cond)*(df_combined_results['region']=='HP')])
    ratio=np.divide(num,denom)
    _,p=scipy.stats.ttest_1samp(ratio,1.0)
    print(cond+'mOsm, 1 sample t-test, p='+str(p))

"=========================================="
"CELLVOL HISTOGRAMME ABSOLU "
"=========================================="
fig, axg = plt.subplots(figsize=[5/2.54,5/2.54],dpi=800)
data_cellvol_iso=np.array(charray_segm_results_import[3,(charray_segm_results_import[0,:]=='300')*(charray_segm_results_import[2,:]!='LP')],dtype=np.float64)
sns.histplot(data=data_cellvol_iso,bins=[0,5000,10000,15000,20000,25000,30000],color='gray')
plt.xlabel(r"Cell volume $[µm^3]$", font=font)
plt.ylim([0,60])
plt.xlim([0,35000])
plt.ylabel('Number of cells',font=font)
plt.title('Cell volume',font=font)
plt.show()

"=========================================="
"=========================================="
figsize=[5/2.54,5/2.54]
x_var='cond'; y_var='vol_r' ; hue_var=None; ratio=0
xlabel='Osmolarity [mOsm]'; ylabel=r'Relative volume $V_r$'  ; hue_var_label=None
xlim=[0.5,3.5];ylim=[0.2,1.0]
title='Rel cell vol'+bars_mode ; restriction=('region',['LP'])
L_color_local=['#8C8C8C','#787878','#646464','#505050']
P26_boxplot_and_violinplot(array_unique_effet,array_unique_zone,df_combined_results,x_var,y_var,hue_var,hue_var_label,title,xlim,ylim,xlabel,ylabel,figsize,bars_mode,restriction=restriction,L_color=L_color_local)

"=========================================="
"SEGM PROP boxplot des données segm : segm prop"
"=========================================="
figsize=[3/2.54,5/2.54]
x_var='region'; y_var='prop_poro' ; hue_var=None
xlabel='Cytoplasm region'; ylabel='Normalized fluorescence'  ; hue_var_label=None
xlim=[-0.5,1.5]; ylim=[.4,1.2]
title='norm fluo HP LP ISO '+bars_mode
restriction=('cond',['300'])
L_color_local=['#00ABF0','#FE351A']
P26_boxplot_and_violinplot(array_unique_effet,array_unique_zone,df_combined_results,x_var,y_var,hue_var,hue_var_label,title,xlim,ylim,xlabel,ylabel,figsize,bars_mode,restriction=restriction,L_color=L_color_local)

figsize=[5/2.54,5/2.54]
x_var='cond'; y_var='prop_poro' ; hue_var=None
xlabel='Osmolarity [mOsm]'; ylabel='Normalized fluorescence'  ; hue_var_label=None
xlim=[-0.5,3.5]; ylim=[.4,1.2]
title='norm fluo HP'+bars_mode
restriction=('region',['HP'])
L_color_local=['#00ABF0','#008FC8','#0072A0','#005678']
P26_boxplot_and_violinplot(array_unique_effet,array_unique_zone,df_combined_results,x_var,y_var,hue_var,hue_var_label,title,xlim,ylim,xlabel,ylabel,figsize,bars_mode,restriction=restriction,L_color=L_color_local)

figsize=[5/2.54,5/2.54]
x_var='cond'; y_var='prop_poro' ; hue_var=None
xlabel='Osmolarity [mOsm]'; ylabel='Normalized fluorescence'  ; hue_var_label=None
xlim=[-0.5,3.5]; ylim=[.4,1.2]
title='norm fluo LP'+bars_mode
restriction=('region',['LP'])
L_color_local=['#FE351A','#EF1D01','#C71901','#9F1401']
P26_boxplot_and_violinplot(array_unique_effet,array_unique_zone,df_combined_results,x_var,y_var,hue_var,hue_var_label,title,xlim,ylim,xlabel,ylabel,figsize,bars_mode,restriction=restriction,L_color=L_color_local)


"=========================================="
"SEGM PROP affichage ratio pour chaque cellule LP/HP"
"=========================================="
x_var='cond'; y_var='prop_poro' ; hue_var=''
xlabel='Osmolarity [mOsm]' ; ylabel='Relative LP/HP fluo' ; hue_var_label=None
ylim=[0.65,1.2]
title='rel norm fluo LP/HP '+bars_mode
P26_boxplot_and_violinplot_ratioLPHP(array_unique_effet,array_unique_zone,df_combined_results,x_var,y_var,hue_var,title,xlabel,ylabel,hue_var_label,ylim,figsize,bars_mode,L_color=L_color_local)

"=========================================================================="
"=========================================================================="
"=============== FIT WITH THEORETICAL MODEL ======================"
"=========================================================================="  
"=========================================================================="        

"USER HAS TO SELECT A CONFIGURATION FOR NANO-OBSTACLES and one for MICRO-OBSTACLES"
nano_solid='3d_r12p5nm' #'3d_r12p5nm': 3D-CFC ribosomes #'2d_r4nm': 2D-CD F-actin
micro_solid='3d_r200nm' #'3d_r200nm'= small granular micro-obstacles
subpath_effects='porous_hindrances_updated_2024_03_28/'


#all nano-obstacles results loaded are expressed as a function of 
#nano-obstacle excluded fraction in micro-scale gel \epsilon_n
x_t_ex_nano=np.load(subpath_effects+'x_t_ex_'+nano_solid+'.npy')
y_t_ex_nano=np.load(subpath_effects+'y_t_ex_'+nano_solid+'.npy')
x_t_ex_micro=np.load(subpath_effects+'x_t_ex_'+micro_solid+'.npy')
y_t_ex_micro=np.load(subpath_effects+'y_t_ex_'+micro_solid+'.npy')

x_hd_ex_nano=np.load(subpath_effects+'x_hd_ex_'+nano_solid+'.npy')
y_hd_ex_nano=np.load(subpath_effects+'y_hd_ex_'+nano_solid+'.npy')
x_hd_ex_micro=np.load(subpath_effects+'x_hd_ex_'+micro_solid+'.npy')
y_hd_ex_micro=np.load(subpath_effects+'y_hd_ex_'+micro_solid+'.npy')*0

x_cb_ex_nano=np.load(subpath_effects+'x_cb_ex_'+nano_solid+'.npy')
y_cb_ex_nano=np.load(subpath_effects+'y_cb_ex_'+nano_solid+'.npy')
x_cb_ex_micro=np.load(subpath_effects+'x_cb_ex_'+micro_solid+'.npy')
y_cb_ex_micro=np.load(subpath_effects+'y_cb_ex_'+micro_solid+'.npy')*0

x_t_ex_nano=np.concatenate((np.array([0]),x_t_ex_nano))
y_t_ex_nano=np.concatenate((np.array([0]),y_t_ex_nano))
x_t_ex_micro=np.concatenate((np.array([0]),x_t_ex_micro))
y_t_ex_micro=np.concatenate((np.array([0]),y_t_ex_micro))

x_hd_ex_nano=np.concatenate((np.array([0]),x_hd_ex_nano))
y_hd_ex_nano=np.concatenate((np.array([0]),y_hd_ex_nano))
x_hd_ex_micro=np.concatenate((np.array([0]),x_hd_ex_micro))
y_hd_ex_micro=np.concatenate((np.array([0]),y_hd_ex_micro))

x_cb_ex_nano=np.concatenate((np.array([0]),x_cb_ex_nano))
y_cb_ex_nano=np.concatenate((np.array([0]),y_cb_ex_nano))
x_cb_ex_micro=np.concatenate((np.array([0]),x_cb_ex_micro))
y_cb_ex_micro=np.concatenate((np.array([0]),y_cb_ex_micro))


"2b - affichage évolution viscosité cytosolique avec le volume"
L_rel_vol=np.arange(0.5,1.01,0.01)

M_rel_viscosity=np.zeros((len(L_k),len(L_rel_vol)))
for i_k in range(len(L_k)):
    k=L_k[i_k]
    for i in range (len(L_rel_vol)):
        rel_vol=L_rel_vol[i]
        M_rel_viscosity[i_k,i]=relative_cytosolic_viscosity(rel_vol,k)


"AFFICHAGE CORRECTION VARIATION DIFFUSIVITE CYTOSOLIQUE"
#plt.figure(dpi=400,figsize=[9,6])
for i_k in range(len(L_k[:3])): 
    k=L_k[i_k+1]
    y_var='eta_r_k='+str(k)
        
    values_mean=[np.mean(df_combined_results[y_var][df_combined_results['cond']=='450'])]
    values_mean.append(np.mean(df_combined_results[y_var][df_combined_results['cond']=='600']))
    values_mean.append(np.mean(df_combined_results[y_var][df_combined_results['cond']=='900']))
    
    values_std=[np.std(df_combined_results[y_var][df_combined_results['cond']=='450'])]
    values_std.append(np.std(df_combined_results[y_var][df_combined_results['cond']=='600']))
    values_std.append(np.std(df_combined_results[y_var][df_combined_results['cond']=='900']))
     
    # plt.errorbar([1,3,5],values_mean,yerr=values_std,capsize=7,marker=L_marker[i_k],color=L_color[i_k],label='k='+str(k))
    # plt.xticks([1,3,5],labels=['450','600','900'])
    # plt.grid()
    # plt.ylabel('Cytosolic diffusivity variation correction')
    # plt.title('Cytosolic diffusivity variation correction (ideal solvent model)')
    # plt.legend()
    

"=========================================================================="  
"3 - définition fonction de fit théorie-expérience"
"=========================================================================="  
"=== APPLICATION CORRECTION VARIATION VISCOSITE CYTOSOLIQUE ==="
def P25_correction_variation_viscosite_cytosolique(df_combined_results,k):
    array_unique_effet=unique_string(df_combined_results['cond'])
    
    L_correction=[]
    m_variable=np.zeros((0))
    for effet in array_unique_effet: 
        mean_vol_r=np.mean(df_combined_results['vol_r'][df_combined_results['cond']==effet])
        
        "modèle solvant idéal (ici k>0) // modèle Kreiger Dougherty (ici k<0)"
        if k>0: #ideal solvent model
            correction=(1+k*(1/mean_vol_r))/(1+k)   
        elif k==0: #no correction
            correction=1
        
        variable=np.array(df_combined_results['diff'][df_combined_results['cond']==effet])*correction
        m_variable=np.concatenate((m_variable,variable))
        L_correction.append(correction)
        
        
        
        
    df_variable=pd.DataFrame(columns=['diff_cor_k='+str(k)], data=m_variable)
    df_combined_results=pd.concat([df_combined_results,df_variable],axis=1) 
    return df_combined_results,L_correction
for k in L_k:   
    df_combined_results,L_correction=P25_correction_variation_viscosite_cytosolique(df_combined_results,k)
    print('k='+str(k)+', L_correction='+str(L_correction))
    

"=== AFFICHAGE OSMOTIC SHIFT REFERENCE POROSITE ==="
def poro_ref(vol_r,poro_ref_iso):
    return 1-vol_r**(-1)*(1-poro_ref_iso)

L_poro_ref_iso=[1,0.95,0.9,0.85,0.8,0.75]
L_vol_r=np.linspace(1,0.5,100)
plt.figure(figsize=[9,6],dpi=400)
for i_poro_ref_iso in range(len(L_poro_ref_iso)):
    poro_ref_iso=L_poro_ref_iso[i_poro_ref_iso]
    
    plt.plot(L_vol_r,poro_ref(L_vol_r,poro_ref_iso),color=L_color[i_poro_ref_iso])#,'k'
    plt.plot(1,poro_ref_iso,marker=L_marker[i_poro_ref_iso],color=L_color[i_poro_ref_iso],label=r'$\epsilon_{ref}^{iso}=$'+str(poro_ref_iso))
        
plt.xlabel(r'Relative cell volume $V_r$')
plt.ylabel('Porosity reference')
plt.title('Evolution poro ref nucléoplasme avec les chocs osmotiques')
plt.xticks(ticks=[0.5,0.6,0.7,0.8,0.9,1], labels=[str(x) for x in [0.5,0.6,0.7,0.8,0.9]]+['Iso-osmotic'])
plt.grid()
plt.legend()

"=== APPLICATION CORRECTION REFERENCE POROSITE ==="
def P25_correction_porosite_reference(df_combined_results,poro_ref_iso,osmotic_ref_shift=True):
    m_variable=np.zeros((0,6))
    for effet in array_unique_effet: 
        if osmotic_ref_shift==False :
            mean_vol_r=1
        else :    
            mean_vol_r=np.mean(df_combined_results['vol_r'][df_combined_results['cond']==effet])
        
        "calcul prop_poro"
        variable_prop_poro=np.array(df_combined_results['prop_poro'][df_combined_results['cond']==effet])*poro_ref(mean_vol_r,poro_ref_iso)
        "calcul prop_s_tot"
        variable_prop_s_tot=1-variable_prop_poro
        "calcul prop_micro"
        f_micro_bin=np.array(df_combined_results['bin_micro'][df_combined_results['cond']==effet])+0.0000001#to avoid division by zero
        variable_prop_micro=f_micro_bin*(1-poro_ref(mean_vol_r,poro_ref_iso)*(1-np.array(df_combined_results['prop_micro'][df_combined_results['cond']==effet])/f_micro_bin))         
        "calcul prop_nano"
        variable_prop_nano=1-variable_prop_micro-variable_prop_poro
        "calcul bin_cor_nano"
        variable_bin_cor_nano=1-f_micro_bin-variable_prop_poro
        
        "calcul bin_cor_nano_OCC-3DCFC-ribosomes"
        variable_bin_cor_nano_occ=variable_bin_cor_nano/1.66 #ratio 1.66 entre volume exclu et occupe pour 3DCFC RIBOSOMES
        
        "variable globale"
        variable_globale=np.zeros((6,variable_prop_poro.shape[0]))
        variable_globale[0,:]=variable_prop_poro
        variable_globale[1,:]=variable_prop_s_tot
        
        variable_globale[2,:]=variable_prop_micro
        variable_globale[3,:]=variable_prop_nano
        variable_globale[4,:]=variable_bin_cor_nano
        variable_globale[5,:]=variable_bin_cor_nano_occ
        variable_globale=np.transpose(variable_globale)
        m_variable=np.concatenate((m_variable,variable_globale),axis=0)
          
    L_names=['prop_poro','prop_s_tot','prop_micro','prop_nano','bin_cor_nano','bin_cor_nano_occ']
    columns=[name+', ref='+str(poro_ref_iso)+', shift='+str(osmotic_ref_shift) for name in L_names]
    df_variable=pd.DataFrame(columns=columns, data=m_variable)
    df_combined_results=pd.concat([df_combined_results,df_variable],axis=1)
    return df_combined_results

"application"
L_osmotic_ref_shift=[True]#False,
for osmotic_ref_shift in L_osmotic_ref_shift:
    for poro_ref_iso in L_poro_ref_iso:   
        df_combined_results=P25_correction_porosite_reference(df_combined_results,poro_ref_iso,osmotic_ref_shift) 
        
       
"=== DEFINITION DE df_combined_results_mean==="
df_combined_results_mean= pd.DataFrame().reindex(columns=df_combined_results.columns)
df_combined_results_std= pd.DataFrame().reindex(columns=df_combined_results.columns)
for condition in unique_string(df_combined_results['cond']):
    for region in unique_string(df_combined_results['region']):      
        df_combined_results_mean.loc[len(df_combined_results_mean)] =df_combined_results[(df_combined_results['cond']==condition)*(df_combined_results['region']==region)].mean(axis=0,numeric_only=True)
        df_combined_results_mean['cond'].loc[len(df_combined_results_mean)-1]=condition
        df_combined_results_mean['region'].loc[len(df_combined_results_mean)-1]=region
        
        df_combined_results_std.loc[len(df_combined_results_std)] =df_combined_results[(df_combined_results['cond']==condition)*(df_combined_results['region']==region)].std(axis=0,numeric_only=True)
        df_combined_results_std['cond'].loc[len(df_combined_results_std)-1]=condition
        df_combined_results_std['region'].loc[len(df_combined_results_std)-1]=region
        
"=== DEFINITION DE df_combined_results_median==="
df_combined_results_median= pd.DataFrame().reindex(columns=df_combined_results.columns)
df_combined_results_q1= pd.DataFrame().reindex(columns=df_combined_results.columns)
df_combined_results_q3= pd.DataFrame().reindex(columns=df_combined_results.columns)
for condition in unique_string(df_combined_results['cond']):
    for region in unique_string(df_combined_results['region']):      
        df_combined_results_median.loc[len(df_combined_results_median)] =df_combined_results[(df_combined_results['cond']==condition)*(df_combined_results['region']==region)].quantile(q=0.5,axis=0,numeric_only=True)
        df_combined_results_median['cond'].loc[len(df_combined_results_median)-1]=condition
        df_combined_results_median['region'].loc[len(df_combined_results_median)-1]=region
        
        df_combined_results_q1.loc[len(df_combined_results_q1)] =df_combined_results[(df_combined_results['cond']==condition)*(df_combined_results['region']==region)].quantile(q=0.25,axis=0,numeric_only=True)
        df_combined_results_q1['cond'].loc[len(df_combined_results_q1)-1]=condition
        df_combined_results_q1['region'].loc[len(df_combined_results_q1)-1]=region
        
        df_combined_results_q3.loc[len(df_combined_results_q3)] =df_combined_results[(df_combined_results['cond']==condition)*(df_combined_results['region']==region)].quantile(q=0.75,axis=0,numeric_only=True)
        df_combined_results_q3['cond'].loc[len(df_combined_results_q3)-1]=condition
        df_combined_results_q3['region'].loc[len(df_combined_results_q3)-1]=region

bars_mode='meanstd'
figsize=[5/2.54,5/2.54]
L_positions=[1,3,5,7]       
L_poro_ref_iso=[0.9,0.85,0.8]
L_osmotic_ref_shift=[True]#porosity reference shift with osmotic condition is activated
L_method=['bin_cor'] #'prop' is obsolete
L_variables=np.array([['prop_poro','prop_nano','prop_micro'], #['prop_poro','prop_nano','prop_micro'] are obsolete variables
                      ['prop_poro','bin_cor_nano','bin_micro']])

L_color_nano=[[154,62,34]]
L_color_nano=np.array(L_color_nano)/255
L_color_nano=list(L_color_nano)

L_color_micro=[[0,0,143]]
L_color_micro=np.array(L_color_micro)/255
L_color_micro=list(L_color_micro)

array_order=np.array(['300','450','600','900'])
volume_type='ex' #ex for excluded or oc for occupied for 3D cfc ribosomes without intersection
"MEDIAN/MEANSTD affichage des jeux de données porosité corrigée pour les différentes hypothèses"

for i_method in range(len(L_method)):
    for osmotic_ref_shift in L_osmotic_ref_shift:
        for poro_ref_iso in L_poro_ref_iso:
            poro_correction=', ref='+str(poro_ref_iso)+', shift='+str(osmotic_ref_shift)
            for i_region in range(2):
                region=['HP','LP'][i_region]
                if i_region==0:
                    L_color_local=['#00ABF0','#008FC8','#0072A0','#005678']
                elif i_region==1:
                    L_color_local=['#FE351A','#EF1D01','#C71901','#9F1401']
                "méthode proportionnelle" 
                for i_variable in range(2):
                    plt.figure(figsize=figsize,dpi=800)
                    plt.title(region+', '+L_method[i_method]+'-ref='+str(poro_ref_iso)+', Osm. shift='+str(osmotic_ref_shift)+', '+bars_mode+'i_var='+str(i_variable),font=font,fontsize='x-small')
                    
                    
                    
                    
                    if i_variable==0 and volume_type=='ex':
                        sns.swarmplot(data=df_combined_results[df_combined_results['region']==region],x='cond',y=L_variables[i_method,1]+poro_correction,palette=L_color_local,alpha=0.8, dodge=False,size=3,legend=0,hue=hue_var_label,order=array_order,zorder=0)#,order=array_order
                        plt.ylabel(r'Nano-obstacle excluded fraction in cytoplasm $\Phi_n$',fontsize='x-small')
                    elif i_variable==0 and volume_type=='oc':
                        sns.swarmplot(data=df_combined_results[df_combined_results['region']==region],x='cond',y='bin_cor_nano_occ'+poro_correction,palette=L_color_local,alpha=0.8, dodge=False,size=3,legend=0,hue=hue_var_label,order=array_order,zorder=0)#,order=array_order
                        plt.ylabel(r'Nano-obstacle occupied fraction in cytoplasm',fontsize='x-small')
                    if i_variable==1:
                        plt.ylabel(r'Micro-obstacle fraction $\Phi_m$',fontsize='x-small')
                        if L_method[i_method]=='prop':
                            sns.swarmplot(data=df_combined_results[df_combined_results['region']==region],x='cond',y='prop_micro',palette=L_color_local,alpha=0.8, dodge=False,size=3,legend=0,hue=hue_var_label,order=array_order,zorder=0,marker='o')#,,size=3.5,marker='^',order=array_order
                        elif L_method[i_method]=='bin_cor':
                            sns.swarmplot(data=df_combined_results[df_combined_results['region']==region],x='cond',y='bin_micro',palette=L_color_local,alpha=0.8, dodge=False,size=3,legend=0,hue=hue_var_label,order=array_order,zorder=0,marker='o')#,marker='^',order=array_order
        
                    
                    for i in range(len(array_order)):
                        group=array_order[i]
                        if 1:
                            "nano"
                            if i_variable==0:
                                if volume_type=='ex':
                                    values=df_combined_results[L_variables[i_method,1]+poro_correction][(df_combined_results['region']==region)*(df_combined_results['cond']==group)]
                                elif volume_type=='oc':
                                    values=df_combined_results['bin_cor_nano_occ'+poro_correction][(df_combined_results['region']==region)*(df_combined_results['cond']==group)]

                                    
                                if bars_mode=='q1q2q3':
                                    low_bar = np.percentile(values, 25)
                                    med_bar = np.percentile(values, 50)
                                    high_bar = np.percentile(values, 75)
                                elif bars_mode=='meanstd':
                                    med_bar = np.mean(values)
                                    low_bar = med_bar-np.std(values)
                                    high_bar = med_bar+np.std(values)
                                
                                plt.vlines(x=i, ymin=low_bar, ymax=high_bar, color="black", linewidth=1)
                                plt.hlines(y=low_bar, xmin=i-0.2, xmax=i+0.2, color="black", linewidth=1)
                                plt.hlines(y=high_bar, xmin=i-0.2, xmax=i+0.2, color="black", linewidth=1)
                                plt.hlines(y=med_bar, xmin=i-0.4, xmax=i+0.4, color="black", linewidth=1.5)
                            
                            
                            elif i_variable==1: 
                                "micro"
                                if L_method[i_method]=='prop':
                                    values=df_combined_results['prop_micro'][(df_combined_results['region']==region)*(df_combined_results['cond']==group)]
                                elif L_method[i_method]=='bin_cor':
                                    values=df_combined_results['bin_micro'][(df_combined_results['region']==region)*(df_combined_results['cond']==group)]
                                
                                if bars_mode=='q1q2q3':
                                    low_bar = np.percentile(values, 25)
                                    med_bar = np.percentile(values, 50)
                                    high_bar = np.percentile(values, 75)
                                elif bars_mode=='meanstd':
                                    med_bar = np.mean(values)
                                    low_bar = med_bar-np.std(values)
                                    high_bar = med_bar+np.std(values)
                                                       
                                plt.vlines(x=i, ymin=low_bar, ymax=high_bar, color="black", linewidth=1)
                                plt.hlines(y=low_bar, xmin=i-0.2, xmax=i+0.2, color="black", linewidth=1)
                                plt.hlines(y=high_bar, xmin=i-0.2, xmax=i+0.2, color="black", linewidth=1)
                                plt.hlines(y=med_bar, xmin=i-0.4, xmax=i+0.4, color="black", linewidth=1.5)
    
                    plt.xlim([-0.5,3.5])
                    plt.ylim([0,0.6])
                    plt.show()
              

    
def P25_fit_exp_model(X_experimental,Y_experimental,model_type,solid_repartition=('% nano',0.7),verbosity=1):
    if solid_repartition[0]=='cte micro':
        #if verbosity:
            #print('IL EST IMPORTANT DE VERIFIER QUE x_var=fraction NANO!!!')
        f_micro=solid_repartition[1]
        
        if model_type=="tortuosity":
            def function_to_fit(F_sol,diff_cytosol):
                F_act=F_sol
                F_orga=f_micro
                effect_diff=diff_cytosol*(1-np.interp(F_act/(1-F_orga),x_t_ex_nano,y_t_ex_nano,left=0))*(1-np.interp(F_orga,x_t_ex_micro,y_t_ex_micro,left=0))
                return effect_diff
        elif model_type=='hydrodynamic':
            def function_to_fit(F_sol,diff_cytosol):
                F_act=F_sol
                F_orga=f_micro
                effect_diff=diff_cytosol*(1-np.interp(F_act/(1-F_orga),x_hd_ex_nano,y_hd_ex_nano,left=0))*(1-np.interp(F_orga,x_hd_ex_micro,y_hd_ex_micro,left=0))
                return effect_diff
        elif model_type=='combined':
            def function_to_fit(F_sol,diff_cytosol):
                F_act=F_sol
                F_orga=f_micro
                effect_diff=diff_cytosol*(1-np.interp(F_act/(1-F_orga),x_t_ex_nano,y_t_ex_nano,left=0))*(1-np.interp(F_orga,x_t_ex_micro,y_t_ex_micro,left=0))*(1-np.interp(F_act/(1-F_orga),x_hd_ex_nano,y_hd_ex_nano,left=0))*(1-np.interp(F_orga,x_hd_ex_micro,y_hd_ex_micro,left=0))
                return effect_diff 
        
            
    "7 - procédure de fit appliquée aux données corrigées (éventuellement binnées)"
    popt, pcov = curve_fit(function_to_fit, X_experimental, Y_experimental)     
    error=np.mean([(Y_experimental[i]-function_to_fit(X_experimental[i], *popt))**2 for i in range(len(X_experimental))])
    return function_to_fit, popt, error


    
def P25_model_fitting_and_plot_for_article(df_combined_results,df_combined_results_mean,x_var,y_var,solid_repartition,figsize=figsize,restrictions_fit=False,restrictions_plot=False,legend=1):
    L_linestyle=['-','--',':']
    L_model=['combined','hydrodynamic','tortuosity']
    L_model_type_short=['Full model,','HDH only,  ','TDH only,  ']
    fig,ax=plt.subplots(dpi=800,figsize=figsize)
    X_fit=np.linspace(0,0.72,1001)
    
    if restrictions_fit!=False : 
        df_combined_results_mean_fit=df_combined_results_mean[df_combined_results_mean[restrictions_fit[0]].isin(restrictions_fit[1])]

    else : 
        df_combined_results_mean_fit=df_combined_results_mean
        
    if restrictions_plot!=False : 
        
        df_combined_results_mean_plot=df_combined_results_mean[df_combined_results_mean[restrictions_plot[0]].isin(restrictions_plot[1])].reset_index()
        df_combined_results_mean_plot_std=df_combined_results_std[df_combined_results_std[restrictions_plot[0]].isin(restrictions_plot[1])].reset_index()
        df_combined_results_plot=df_combined_results[df_combined_results[restrictions_plot[0]].isin(restrictions_plot[1])].reset_index()

    else : 
        df_combined_results_mean_plot=df_combined_results_mean
        df_combined_results_mean_plot_std=df_combined_results_std
        df_combined_results_plot=df_combined_results
    
        

    X_experimental_fit=np.array(df_combined_results_mean_fit[x_var])
    Y_experimental_fit=np.array(df_combined_results_mean_fit[y_var])
    
    
    
    for i_model in range(len(L_model)):   
        model_type=L_model[i_model] ; model_type_short=L_model_type_short[i_model]
        function_to_fit,popt, error=P25_fit_exp_model(X_experimental_fit,Y_experimental_fit,model_type,solid_repartition)
        plt.plot(X_fit, function_to_fit(X_fit, *popt),color='k',linestyle=L_linestyle[i_model],label=model_type_short+r' $D_{\alpha}$='+str(round(popt[0],1))+', RMSE='+str(round(np.sqrt(error),2)))


    l_colors=L_color
    l_markers=('^','o','^','o','^','o','^','o','^','o')
    X_experimental_plot=np.array(df_combined_results_mean_plot[x_var])
    Y_experimental_plot=np.array(df_combined_results_mean_plot[y_var])
    X_experimental_plot_std=np.array(df_combined_results_mean_plot_std[x_var])
    Y_experimental_plot_std=np.array(df_combined_results_mean_plot_std[y_var])
    for i in range(len(X_experimental_plot)):
        if df_combined_results_mean_plot['region'][i]=='HP':
            marker='o'
        else :
            marker='^'
        if df_combined_results_mean_plot['cond'][i]=='300':
            color=l_colors[0]
        elif df_combined_results_mean_plot['cond'][i]=='450':
            color=l_colors[1]
        elif df_combined_results_mean_plot['cond'][i]=='600':
            color=l_colors[2]
        elif df_combined_results_mean_plot['cond'][i]=='900':
            color=l_colors[3]
        plt.errorbar(X_experimental_plot[i], Y_experimental_plot[i], yerr=Y_experimental_plot_std[i],xerr=X_experimental_plot_std[i],marker=marker,markeredgewidth=1,markerfacecolor=color, markersize=6,capsize=5,color='k')#,color=l_colors[i])                
        
    if legend:
        plt.legend(fontsize='x-small')
    
    
    for i in range(df_combined_results_plot.shape[0]):
        x_plot=df_combined_results_plot[x_var][i]
        y_plot=df_combined_results_plot[y_var][i]
        if df_combined_results_plot['region'][i]=='HP':
            marker='o'
        else :
            marker='^'
        if df_combined_results_plot['cond'][i]=='300':
            color=l_colors[0]
        elif df_combined_results_plot['cond'][i]=='450':
            color=l_colors[1]
        elif df_combined_results_plot['cond'][i]=='600':
            color=l_colors[2]
        elif df_combined_results_plot['cond'][i]=='900':
            color=l_colors[3]
        plt.plot(x_plot,y_plot,marker=marker,color=color,alpha=0.5,ms=4.5,markeredgewidth=0)
        

    
    plt.title(nano_solid+'D, s_r='+str(solid_repartition)+',xvar='+x_var+', y_var='+y_var+'vers2024 03 28',fontsize='x-small')
    plt.xlabel(r'Nano-solid excluded volume fraction $\Phi_n$',font=font);plt.ylabel(r'Cytoplasmic diffusivity $D_{eff}$ [$\mu m^2 s^{-1}$]',font=font)
    
    plt.xlim([0,0.6])
    plt.ylim([0,30])
    ax.set_box_aspect(1) 
    plt.show()
    
  
"====== 2024 03 28 AFFICHAGE FONCTIONS POREUX NORMEES==="
L_model=['tortuosity','hydrodynamic','combined']
L_model_label=['Tortuosity only (TH)','Hydrodynamic only (HH)','Full model']
f_micro=0.08
plt.figure(dpi=400,figsize=[9,9])
L_linestyle=['-','--',':']
x_f_ex_nano=np.arange(0,0.65,0.001)
for i_model_type in range(len(L_model)):
    model_type=L_model[i_model_type]
    if model_type=="tortuosity":
        def function_to_fit(F_sol,diff_cytosol):
            F_act=F_sol
            F_orga=f_micro
            effect_diff=diff_cytosol*(1-np.interp(F_act/(1-F_orga),x_t_ex_nano,y_t_ex_nano,left=0))*(1-np.interp(F_orga,x_t_ex_micro,y_t_ex_micro,left=0))
            return effect_diff
    elif model_type=='hydrodynamic':
        def function_to_fit(F_sol,diff_cytosol):
            F_act=F_sol
            F_orga=f_micro
            effect_diff=diff_cytosol*(1-np.interp(F_act/(1-F_orga),x_hd_ex_nano,y_hd_ex_nano,left=0))*(1-np.interp(F_orga,x_hd_ex_micro,y_hd_ex_micro,left=0))
            return effect_diff
    elif model_type=='combined':
        def function_to_fit(F_sol,diff_cytosol):
            F_act=F_sol
            F_orga=f_micro
            effect_diff=diff_cytosol*(1-np.interp(F_act/(1-F_orga),x_t_ex_nano,y_t_ex_nano,left=0))*(1-np.interp(F_orga,x_t_ex_micro,y_t_ex_micro,left=0))*(1-np.interp(F_act/(1-F_orga),x_hd_ex_nano,y_hd_ex_nano,left=0))*(1-np.interp(F_orga,x_hd_ex_micro,y_hd_ex_micro,left=0))
            return effect_diff  
        
    plt.plot(x_f_ex_nano,function_to_fit(x_f_ex_nano,1),color='k',linestyle=L_linestyle[i_model_type],label=L_model_label[i_model_type])
plt.grid()
plt.xlabel(r'Nano-solid excluded volume fraction $\Phi_n$')
plt.ylabel(r'Relative cytoplasmic/cytosolic diffusivity $D_eff/D_\alpha$')
plt.title(r'Multiscale model function to fit, $\Phi_m$='+str(f_micro)+'nano-s='+str(nano_solid)+', micro-s='+str(micro_solid))
plt.legend(title='Porous medium model')     
plt.xlim([-0.02,0.68])
plt.ylim([0.1,1.02])
plt.show()
    

   
      
#%% 
"=========================================================================="
"2025 05 22 MODEL FIT WITH BOOTSTRAPING"
"=========================================================================="
"1 -- AFFICHAGE DES FITS SEPARES SANS CORRECTION"
#L_k_visc_cyto=[]#value of the cytosolic viscosity constant k 
L_osmotic_ref_shift=[True]
L_poro_ref_iso=[0.85]
solid_repartition=('cte micro',0.08)#micro-obstacles are approximated with a constant volume fraction Phi_m=0.08
bootstraping=1 #activate bootstraping analysis
L_restrictions=[['300','450','600'],['300'],['450'],['600']] #which osmostic conditions to fit
L_std_D_alpha_0=[]
N_bootstrap=1000#1000
for i_restriction in range(len(L_restrictions)):
    restrictions_fit=('cond',L_restrictions[i_restriction])
    if i_restriction==0:
        k_visc_cyto=1#value of the cytosolic viscosity constant k set to 1 for all data fit (fig 5 main manuscript)
    else:
        k_visc_cyto=0 #value of the cytosolic viscosity constant k set to 0 for fit of individual osmotic condition

    osmotic_ref_shift=True
    poro_ref_iso=L_poro_ref_iso[0]
                    
    x_correction=', ref='+str(poro_ref_iso)+', shift='+str(osmotic_ref_shift)
    y_correction='_cor_k='+str(k_visc_cyto)
    x_var='bin_cor_nano'+x_correction
    y_var='diff'+y_correction
    figsize=[12/2.54,12/2.54]
    (df_combined_results,df_combined_results_mean,x_var,y_var,solid_repartition,figsize,restrictions_fit,restrictions_plot,legend,bootstrap)=(df_combined_results,df_combined_results_mean,x_var,y_var,solid_repartition,figsize,restrictions_fit,restrictions_fit,1,bootstraping)

    L_linestyle=['-','--',':']
    L_model=['combined','hydrodynamic','tortuosity']
    L_model_type_short=['Full model,','HDH only,  ','TDH only,  ']
    fig,ax=plt.subplots(dpi=800,figsize=figsize)
    X_fit=np.linspace(0,0.72,1001)
    
    if restrictions_fit!=False : 
        df_combined_results_mean_fit=df_combined_results_mean[df_combined_results_mean[restrictions_fit[0]].isin(restrictions_fit[1])]
    else : 
        df_combined_results_mean_fit=df_combined_results_mean
        
    if restrictions_plot!=False : 
        
        df_combined_results_mean_plot=df_combined_results_mean[df_combined_results_mean[restrictions_plot[0]].isin(restrictions_plot[1])].reset_index()
        df_combined_results_mean_plot_std=df_combined_results_std[df_combined_results_std[restrictions_plot[0]].isin(restrictions_plot[1])].reset_index()
        df_combined_results_plot=df_combined_results[df_combined_results[restrictions_plot[0]].isin(restrictions_plot[1])].reset_index()
    
    else : 
        df_combined_results_mean_plot=df_combined_results_mean
        df_combined_results_mean_plot_std=df_combined_results_std
        df_combined_results_plot=df_combined_results
    
    X_experimental_fit=np.array(df_combined_results_mean_fit[x_var])
    Y_experimental_fit=np.array(df_combined_results_mean_fit[y_var])
    
    for i_model in range(len(L_model)):   
        model_type=L_model[i_model] ; model_type_short=L_model_type_short[i_model]
        function_to_fit,popt, error=P25_fit_exp_model(X_experimental_fit,Y_experimental_fit,model_type,solid_repartition)
        
        if restrictions_fit!=False : 
            df_combined_results_fit=df_combined_results[df_combined_results[restrictions_plot[0]].isin(restrictions_plot[1])].reset_index()
        else:
            df_combined_results_fit=df_combined_results_mean

        def P25_fit_exp_model_bootstrap_ready(i):
            global var 
            
            df_combined_results_fit_local_LP=df_combined_results_fit[df_combined_results_fit['region']=='LP'].iloc[i]
            df_combined_results_fit_local_HP=df_combined_results_fit[df_combined_results_fit['region']=='HP'].iloc[i]
            df_combined_results_fit_local=pd.concat((df_combined_results_fit_local_LP,df_combined_results_fit_local_HP))
            
            "=== DEFINITION DE df_combined_results_mean==="
            df_combined_results_fit_local_mean= pd.DataFrame().reindex(columns=df_combined_results_fit_local.columns)
            
            for condition in unique_string(df_combined_results_fit_local['cond']):
                for region in unique_string(df_combined_results_fit_local['region']):      
                    df_combined_results_fit_local_mean.loc[len(df_combined_results_fit_local_mean)] =df_combined_results_fit_local[(df_combined_results_fit_local['cond']==condition)*(df_combined_results_fit_local['region']==region)].mean(axis=0,numeric_only=True)
                    df_combined_results_fit_local_mean['cond'].loc[len(df_combined_results_fit_local_mean)-1]=condition
                    df_combined_results_fit_local_mean['region'].loc[len(df_combined_results_fit_local_mean)-1]=region   

            X_experimental_fit=np.array(df_combined_results_fit_local_mean[x_var])
            Y_experimental_fit=np.array(df_combined_results_fit_local_mean[y_var])
            function_to_fit,popt, error=P25_fit_exp_model(X_experimental_fit,Y_experimental_fit,model_type,solid_repartition,verbosity=0)
            if var=='D_alpha_0':
                var_value=popt[0]
            elif var=='RMSE':
                var_value=error
            return var_value
        
        i=np.arange(len(df_combined_results_fit)//2)
        var='D_alpha_0'
        std_D_alpha_0 =scipy.stats.bootstrap((i,), P25_fit_exp_model_bootstrap_ready,n_resamples=N_bootstrap,random_state=1).standard_error#
        L_std_D_alpha_0.append([restrictions_fit[1],model_type,std_D_alpha_0])
        var='RMSE'
        std_RMSE=scipy.stats.bootstrap((i,), P25_fit_exp_model_bootstrap_ready,n_resamples=N_bootstrap,random_state=1).standard_error#

        if len(restrictions_fit[1])==1:
            plt.plot(X_fit, function_to_fit(X_fit, *popt),color='k',linestyle=L_linestyle[i_model],label=model_type_short+r' $D_{\alpha}$='+str(round(popt[0],1))+r'$\pm$'+str(round(std_D_alpha_0,2))+', RMSE='+str(round(np.sqrt(error),2))+r'$\pm$'+str(round(std_RMSE,2)))
        elif len(restrictions_fit[1])==3:
            plt.plot(X_fit, function_to_fit(X_fit, *popt),color='k',linestyle=L_linestyle[i_model],label=model_type_short+r' $D_{\alpha}^0$='+str(round(popt[0],1))+r'$\pm$'+str(round(std_D_alpha_0,2))+', RMSE='+str(round(np.sqrt(error),2))+r'$\pm$'+str(round(std_RMSE,2)))

    l_colors=L_color
    l_markers=('^','o','^','o','^','o','^','o','^','o')
    X_experimental_plot=np.array(df_combined_results_mean_plot[x_var])
    Y_experimental_plot=np.array(df_combined_results_mean_plot[y_var])
    X_experimental_plot_std=np.array(df_combined_results_mean_plot_std[x_var])
    Y_experimental_plot_std=np.array(df_combined_results_mean_plot_std[y_var])
    for i in range(len(X_experimental_plot)):
        if df_combined_results_mean_plot['region'][i]=='HP':
            marker='o'
        else :
            marker='^'
        if df_combined_results_mean_plot['cond'][i]=='300':
            color=l_colors[0]
        elif df_combined_results_mean_plot['cond'][i]=='450':
            color=l_colors[1]
        elif df_combined_results_mean_plot['cond'][i]=='600':
            color=l_colors[2]
        elif df_combined_results_mean_plot['cond'][i]=='900':
            color=l_colors[3]
        plt.errorbar(X_experimental_plot[i], Y_experimental_plot[i], yerr=Y_experimental_plot_std[i],xerr=X_experimental_plot_std[i],marker=marker,markeredgewidth=1,markerfacecolor=color, markersize=6,capsize=5,color='k')#,color=l_colors[i])                
        
    if legend:
        plt.legend(fontsize='small')
    
    
    for i in range(df_combined_results_plot.shape[0]):
        x_plot=df_combined_results_plot[x_var][i]
        y_plot=df_combined_results_plot[y_var][i]
        if df_combined_results_plot['region'][i]=='HP':
            marker='o'
        else :
            marker='^'
        if df_combined_results_plot['cond'][i]=='300':
            color=l_colors[0]
        elif df_combined_results_plot['cond'][i]=='450':
            color=l_colors[1]
        elif df_combined_results_plot['cond'][i]=='600':
            color=l_colors[2]
        elif df_combined_results_plot['cond'][i]=='900':
            color=l_colors[3]
        plt.plot(x_plot,y_plot,marker=marker,color=color,alpha=0.5,ms=4.5,markeredgewidth=0)#, markeredgecolor=[0.3,0.3,0.3]

    plt.title(nano_solid+'D, s_r='+str(solid_repartition)+',xvar='+x_var+', y_var='+y_var+'vers2024 03 28',fontsize='x-small')
    plt.xlabel(r'Nano-solid excluded volume fraction $\Phi_n$',font=font);plt.ylabel(r'Cytoplasmic diffusivity $D_{eff}$ or cytoplasmic corrected diffusivity $D_{eff}\times(D_\alpha^0/D_\alpha)$ if k>0 [$\mu m^2 s^{-1}$]',font=font,fontsize='x-small')#plt.ylabel('Diffusivité absolue ($\mu m²s^{-1}$)');plt.legend(fontsize='x-large')

    plt.xlim([0,0.6])
    plt.ylim([0,30])

    plt.show()
    
    
figsize=[5/2.54,5/2.54]
"2 -- AFFICHAGE évolution DIFFUSIVITY cytosolique avec le volume"
L_rel_vol=np.arange(0.6,1.01,0.01)
M_rel_viscosity=np.zeros((len(L_k),len(L_rel_vol)))
for i_k in range(len(L_k)):
    k=L_k[i_k]
    for i in range (len(L_rel_vol)):
        rel_vol=L_rel_vol[i]
        M_rel_viscosity[i_k,i]=relative_cytosolic_viscosity(rel_vol,k)

plt.figure(figsize=figsize,dpi=800)
plt.plot(L_rel_vol,M_rel_viscosity[0,:],'--',color='grey',linewidth=1)
plt.plot(L_rel_vol,1/M_rel_viscosity[2,:],color='grey')
plt.fill_between(L_rel_vol, 1/M_rel_viscosity[1,:], 1/M_rel_viscosity[3,:],alpha=0.3,color='grey',linewidth=0)

"calcul des incertitudes (standard dev) sur le ratio de diffusivité cytosolique via résultats bootstrap 2025 05 22"
"propagation des incertitudes d'après https://en.wikipedia.org/wiki/Propagation_of_uncertainty voir calculs sur mon cahier"
def std_ratio_a_b(a,b,std_a,std_b): #for a and b independent
    return ((std_a/b)**2+(a*std_b)**2/b**4)**(1/2)


"données std en provenance de L_std_D_alpha_0"
"Full model 5img classical data"
yerr=[0,std_ratio_a_b(25.3,28.0,0.75,0.67),std_ratio_a_b(22.1,28.0,0.66,0.67)]
plt.errorbar([1,0.80,0.72],[1,25.3/28.0,22.1/28.0],xerr=[0,0.05,0.07],yerr=yerr,marker='',linestyle='-',color='k',capsize=3)#,linewidth=1#,xerr=[0,0.05,0.07]

plt.xlabel(r'Relative cell volume $V_r$',font=font)
plt.xticks(ticks=[0.6,0.7,0.8,0.9,1], labels=[str(x) for x in [0.6,0.7,0.8,0.9]]+['1'])
plt.ylabel(r'Cytosolic diffusivity $D_\alpha$ relative to 300mOsm',font=font)
plt.title('Cytosolic diffusivity change with cell volume')   
plt.gca().invert_xaxis()
plt.xlim([1,0.6])
plt.ylim([0.7,1.1])
#plt.grid()  
plt.xlim
plt.show()  

plt.figure(figsize=figsize,dpi=800)
plt.plot(L_rel_vol,M_rel_viscosity[0,:],'--',color='grey',label='Pure solvent (k=0)',linewidth=1)
plt.plot(L_rel_vol,1/M_rel_viscosity[2,:],color='grey',label=r'$k=1 \pm 0.5$')#,marker=L_marker[2-1],markevery=10)
"3D12P5NM Full model"
plt.plot([1,0.80,0.72],[1,25.3/28.0,22.1/28.0],linestyle='-',color='k',label=r'$D_\alpha$ from Full model')#,xerr=[0,0.05,0.07]
plt.xlim([-2,-1])
plt.legend(fontsize='small')
plt.show() 

#%%
"=========================================================================="
"""STUDY OF DIFFUSIVITY DEPENDANCE WItH PARTICLE SIZE"""
"=========================================================================="
gfp_mass=27 # en kda
r_gfp=2.3   # en nm
"=== PASSAGE R INTERP A 4NM POUR AVOIR RESULTATS LUBY PHELPS"
r_interp_y=3.50 #en nm  #arbitrary particle radius taken for data normalization

"Bubak 2021 : dextrans from 4.4 to 155Kda, in cytoplasm"
X_bubak=np.array([1.24058559153217,	4.88917325357807,	5.57181223550925,	8.584839466102])
Y_bubak=np.array([3.02526248776257,4.69583586077942	,5.24091720303265	,7.59284225386602])
"normalisation by the interpolated value at r=r_interp_y"
Y_interp=np.interp(r_interp_y,X_bubak,Y_bubak)
Y_bubak=(Y_bubak/Y_interp)**-1


"Baum 2014 : GFP oligomers (GFP1, GFP3, GFP5) in cytoplasm"
X_baum=np.array([2.8	,5.5	,7.9])
Y_baum=np.array([1.29253731343283	,1.79402985074626,	2.59402985074626])
"normalisation by the interpolated value at r=r_interp_y"
Y_interp=np.interp(r_interp_y,X_baum,Y_baum)
Y_baum=(Y_baum/Y_interp)**-1

"Luby-Phelps 1987  : DEXTRANS in 3T3 cells cytoplasm"
X_luby_dextrans=np.array([3.406451612903226	,4.87741935483871	,6.348387096774194,	7.122580645161291,	9.367741935483872	,10.529032258064516	,14.090322580645163	,36.38709677419355,	57.83225806451613])
Y_luby_dextrans=np.array([0.2090267983074753	,0.18977433004231312	,0.1796191819464034	,0.17031029619181948,	0.1495768688293371	,0.12968970380818054,	0.06960507757404795	,0.07849083215796897,	0.037870239774330045])
"range restriction"
X_luby_dextrans=X_luby_dextrans[:5]
Y_luby_dextrans=Y_luby_dextrans[:5]
"normalisation by the interpolated value at r=r_interp_y"
Y_interp=np.interp(r_interp_y,X_luby_dextrans,Y_luby_dextrans)
Y_luby_dextrans=Y_luby_dextrans/Y_interp

"Luby-Phelps 1987  : FICOLL in 3T3 cells cytoplasm"
X_luby_ficoll=np.array([3.406451612903226,	6.5032258064516135	,10.916129032258064,	14.090322580645163	,18.116129032258065,	22.761290322580646	,24.69677419354839])
Y_luby_ficoll=np.array([0.27609308885754585,	0.22320169252468267,	0.16650211565585332,	0.11445698166431593	,0.096262341325811,	0.034908321579689705,	0.03236953455571227])
"range restriction"
X_luby_ficoll=X_luby_ficoll[:2]
Y_luby_ficoll=Y_luby_ficoll[:2]
"normalisation by the interpolated value at r=r_interp_y"
Y_interp=np.interp(r_interp_y,X_luby_ficoll,Y_luby_ficoll)
Y_luby_ficoll=Y_luby_ficoll/Y_interp

#user selection of nano obstacles configuration considered
nano_solid='3d_r12p5nm'#'2d_r4nm'= 2D CD F-actine  '3d_r12p5nm'=3D CFC ribosomes
micro_solid='3d_r200nm' #3D CFC micro-obstacles
subpath_effects='results_particle_size_updates_2024_03_28/' 

x_particle_radius=np.load(subpath_effects+'x_particle_radius_2024_03_28.npy')

if nano_solid=='2d_r4nm':
    y_t_particle_radius_nano=np.load(subpath_effects+'y_t_particle_radius_2d_r4nm_fraction_0p08_2024_03_28.npy')
    y_hd_particle_radius_nano=np.load(subpath_effects+'y_hd_particle_radius_2d_r4nm_fraction_0p08_2024_03_28.npy')
    y_cb_particle_radius_nano=np.load(subpath_effects+'y_cb_particle_radius_2d_r4nm_fraction_0p08_2024_03_28.npy')
    
elif nano_solid=='3d_r12p5nm':
    y_t_particle_radius_nano=np.load(subpath_effects+'y_t_particle_radius_3d_r12p5nm_fraction_0p12_2024_03_28.npy')
    y_hd_particle_radius_nano=np.load(subpath_effects+'y_hd_particle_radius_3d_r12p5nm_fraction_0p12_2024_03_28.npy')
    y_cb_particle_radius_nano=np.load(subpath_effects+'y_cb_particle_radius_3d_r12p5nm_fraction_0p12_2024_03_28.npy')
    
if micro_solid=='3d_r200nm':
    y_t_particle_radius_micro=np.load(subpath_effects+'y_t_particle_radius_3d_r200nm_fraction_0p08_2024_03_28.npy')
    y_hd_particle_radius_micro=y_t_particle_radius_micro*0
    y_cb_particle_radius_micro=y_t_particle_radius_micro*0





# "2 - affichage réduction diffusivité pour solides représentatifs pour nano et micro solides sélectionnés "
# plt.figure(dpi=400,figsize=[6,6])
# plt.plot(x_particle_radius,1-y_cb_particle_radius_nano,'k-',label='Full model')
# plt.plot(x_particle_radius,1-y_hd_particle_radius_nano,'k--',label='HDH only')
# plt.plot(x_particle_radius,1-y_t_particle_radius_nano,'k:',label='TH only')
# plt.legend(title='Extrapolated model',loc='upper right')
# plt.title('Nano-solid selected = '+nano_solid)
# plt.grid()
# plt.xlabel('Particle radius R (nm)')
# plt.ylabel(r'Relative diffusivity   $D_{eff}/D_\alpha$')
# plt.ylim([0,1])
# plt.xlim([0,10])
# plt.show()

def Porous_effect_calculation(R_partic,model_type='combined'):#(R_partic,f_nano,f_micro,model_type='combined'):
        if model_type=="tortuosity":
            effect_diff=(1-np.interp(R_partic,x_particle_radius,y_t_particle_radius_nano,left=0))*(1-np.interp(R_partic,x_particle_radius,y_t_particle_radius_micro,left=0))
            return effect_diff
        elif model_type=='hydrodynamic':
            effect_diff=(1-np.interp(R_partic,x_particle_radius,y_hd_particle_radius_nano,left=0))*(1-np.interp(R_partic,x_particle_radius,y_hd_particle_radius_micro,left=0))
            return effect_diff
        elif model_type=='combined':
            effect_diff=(1-np.interp(R_partic,x_particle_radius,y_t_particle_radius_nano,left=0))*(1-np.interp(R_partic,x_particle_radius,y_hd_particle_radius_micro,left=0))*(1-np.interp(R_partic,x_particle_radius,y_hd_particle_radius_nano,left=0))*(1-np.interp(R_partic,x_particle_radius,y_hd_particle_radius_micro,left=0))
            return effect_diff 
        
L_model_type=['combined','hydrodynamic','tortuosity']
L_model_label=['Full model','HDH only ','TH only    ']

X_experimental=np.concatenate((X_bubak,X_baum,X_luby_dextrans,X_luby_ficoll))
Y_experimental=np.concatenate((Y_bubak,Y_baum,Y_luby_dextrans,Y_luby_ficoll))
X_luby=np.concatenate((X_luby_dextrans,X_luby_ficoll))
L_weight=np.size(X_bubak)*[1/np.size(X_bubak)]+np.size(X_baum)*[1/np.size(X_baum)]+np.size(X_luby)*[1/np.size(X_luby)]

# #fig=plt.figure(dpi=800,figsize=[5,5])
# L_line_type=['-','--',':']
# L_R=np.linspace(1,10,100)
# for i_model_type in range(len(L_model_type)):
#     model_type= L_model_type[i_model_type]
#     L_porous_effect=[] 
#     for i_R in range(len(L_R)):
#         R_partic=L_R[i_R]  
#         L_porous_effect.append(Porous_effect_calculation(R_partic,model_type))
    
#     L_porous_effect=np.array(L_porous_effect)

#     def function_without_fit(x):
#         return np.interp(x,L_R,L_porous_effect)/np.interp(r_interp_y,L_R,L_porous_effect)
    
#     error=np.sum([L_weight[i]*(Y_experimental[i]-function_without_fit(X_experimental[i]))**2 for i in range(len(X_experimental))])/np.sum(L_weight)
#     "without error"
#     #plt.plot(L_R,function_without_fit(L_R),'k',label=L_model_label[i_model_type],linestyle=L_line_type[i_model_type])

# plt.plot(X_bubak,Y_bubak,'o',color=L_color[0])
# plt.plot(X_baum,Y_baum,'s',color=L_color[1])
# plt.plot(X_luby_dextrans,Y_luby_dextrans,'^',color=L_color[2])
# plt.plot(X_luby_ficoll,Y_luby_ficoll,'*',color=L_color[2])

# plt.legend(title='Extrapolated model')
# plt.xlabel(r'Particle radius   $R$(nm)')
# plt.ylabel(r'Quotient of relative diffusivity   $Q_{exp}$ and $Q_{model}$')
# plt.title('Bubak pre-study, nano_solid='+str(nano_solid)+', NO micro-sol, v2024 10 14',fontsize='x-small')
# plt.xlim([1,10])
# plt.ylim([0,1.5])
# plt.grid()

# fig=plt.figure(dpi=800,figsize=[6,6])
# plt.plot([0,1],[0,1],'o',color=L_color[0],label='Bubak & al. 2021 - Dextrans - HeLa (human)')
# plt.plot([0,1],[0,1],'s',color=L_color[1],label='Baum & al. 2014 - GFP oligomers - U2OS (human)')
# plt.plot([0,1],[0,1],'^',color=L_color[2],label='Luby-Phelps & al. 1987 - Dextrans - 3T3 (mouse)')
# plt.plot([0,1],[0,1],'*',color=L_color[2],label='Luby-Phelps & al. 1987 - Ficolls - 3T3 (mouse)')
# plt.legend(title='Literature data',fontsize='small')
# plt.show()


"2025 05 27 version affichage ARTICLE"
fig=plt.figure(dpi=800,figsize=[9/2.54,9/2.54])
L_line_type=['-','--',':']
L_R=np.linspace(0,10,100)
for i_model_type in range(len(L_model_type)):
    model_type= L_model_type[i_model_type]
    L_porous_effect=[] 
    for i_R in range(len(L_R)):
        R_partic=L_R[i_R]  
        L_porous_effect.append(Porous_effect_calculation(R_partic,model_type))
    
    L_porous_effect=np.array(L_porous_effect)
    def function_without_fit(x):
        return np.interp(x,L_R,L_porous_effect)/np.interp(r_interp_y,L_R,L_porous_effect)
    
    error=np.sum([L_weight[i]*(Y_experimental[i]-function_without_fit(X_experimental[i]))**2 for i in range(len(X_experimental))])/np.sum(L_weight)
    "with error"
    "without error"
    plt.plot(L_R,function_without_fit(L_R),'k',label=L_model_label[i_model_type],linestyle=L_line_type[i_model_type])

plt.plot(X_baum,Y_baum,'s',color='white',markeredgecolor=L_color[1],markeredgewidth=1.5,ms=6,fillstyle='none')#'#FFC000'
plt.plot(X_bubak,Y_bubak,'o',color='white',markeredgecolor=L_color[0],markeredgewidth=1.5,ms=6,fillstyle='none')#'#C00000'

plt.plot(X_luby_dextrans,Y_luby_dextrans,'^',color='white',markeredgecolor=L_color[2],markeredgewidth=1.5,ms=6,fillstyle='none')#'#005390'
plt.plot(X_luby_ficoll,Y_luby_ficoll,'v',color='white',markeredgecolor=L_color[2],markeredgewidth=1.5,ms=6,fillstyle='none')#'#005390'

plt.plot([0,10],[1,1],linestyle='-.',color='grey',zorder=0,linewidth=1.2)

plt.xlabel(r'Particle radius   $R$(nm)')
plt.ylabel(r'Normalized diffusivities   $D_{exp}^*$ and $D_{eff}^*$')
plt.title('Particle size study, nano_solid='+str(nano_solid)+', v2024 10 14',fontsize='x-small')
plt.xlim([0,10])
plt.ylim([0,1.6])

fig=plt.figure(dpi=800,figsize=[6,6])
plt.plot([0,1],[0,1],'o',color=L_color[0],label='Bubak & al. 2021 - Dextrans - HeLa (human)')
plt.plot([0,1],[0,1],'s',color=L_color[1],label='Baum & al. 2014 - GFP oligomers - U2OS (human)')
plt.plot([0,1],[0,1],'^',color=L_color[2],label='Luby-Phelps & al. 1987 - Dextrans - 3T3 (mouse)')
plt.plot([0,1],[0,1],'v',color=L_color[2],label='Luby-Phelps & al. 1987 - Ficolls - 3T3 (mouse)')

plt.legend(title='Literature data',fontsize='small')
plt.show()










    
