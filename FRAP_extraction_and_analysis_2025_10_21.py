#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CURRENT VERSION WRITTEN BY OLIVIER DESTRIAN IN OCTOBER 2025
PAPER "Cytoplasmic crowding acts as a porous medium reducing macromolecule diffusion"
This code allows to analyse FRAP data obtained from a 980LSM Zeiss confocal microscope
To infer the FRAP diffusivity of GFP in living cells
"""
import aicspylibczi as new_czi_reader
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import scipy.optimize as scop

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

def build_num(num): #Format a number as a two-digit string, adding a leading zero if needed.
    if num<10:
        return '0'+str(int(num))
    else : 
        return str(int(num))
    
def moving_average2(x,n1,n2): #Apply 2D moving average smoothing using a convolution window of size (n1, n2)
    w=np.ones((n1,n2))#/2
    #w[1,1]=1
    somme=np.sum(w)
    return  signal.convolve2d(x, w, mode='same', boundary='symm')/ somme


def P25_extraction_data_FRAP(path_data,name_0,L_indices,T_fading,background_value,N_rafale_separees,N_rafale_fichier,i_postbleach): #T_fading : tuple with instructions for correcting fading. Also see the build_image function
    #"Extract and average FRAP image data across multiple repetitions.
    N_cellules=len(L_indices)
    T_img_array=()
    
    # 1 - Extract image bursts from one or multiple repetitions
    for i_C in range(N_cellules):
        name=name_0+str(L_indices[i_C]) 
        print("P25_extraction_data_FRAP : "+name)
        T_img_array=T_img_array+(P25_build_img_array(name,path_data,N_rafale_fichier,background_value,T_fading,i_postbleach)[0],)
        N_t,N_x=T_img_array[0].shape                                                 
    # 2 - Merge and average bursts from multiple repetitions
    img_array=np.zeros((N_t,N_x))
    for i_rafale in range(N_rafale_separees):
        img_array=img_array+T_img_array[i_rafale]   
    img_array=img_array/N_rafale_separees
    return img_array

def CLEAN_FRAP_analysis(img_array,i_postbleach,N_t_decalage_fit,N_t_fit_duration,T_acq,Size_px,dim,fit_fading=0,plots=1,info=['','']):
    k_zone=info[0]; k_cell=info[1]
    
    # Select image portion used for fitting
    img_to_fit=img_array[i_postbleach+N_t_decalage_fit:,:]
    N_x=img_to_fit.shape[1]
    img_to_fit=img_to_fit[:N_t_fit_duration,:]


    #Fitting parameters boundaries
    t_pb_bd=[1e-6,1e6]#[1e-6,1e6] #min and max for t_pb
    K_bd=[1e-6,1e6] #[1e-6,1e6] min and max for K   
    D_bd=[1e-6,1e4] #[1e-6,1e3] min and max for D    
    N_x=img_to_fit.shape[1]
    px_center_bd=[0.25*N_x,0.75*N_x]
    
    #Mathematical model for FRAP recovery in 2D.
    def function_2D_to_fit(XT,t_pb,K,D,px_center):
        x,t=XT   
        value=1-np.multiply(np.power(np.divide((K/(4*D)), t+t_pb),dim/2),np.exp(np.divide(-np.power(x-px_center,2),(4*D*(t+t_pb)))))
        return np.array(value,dtype=float)

    x=np.linspace(0,N_x-1,N_x)
    t=np.linspace(0,N_t_fit_duration-1,N_t_fit_duration)
    x,t = np.meshgrid(x,t) 
    XT=np.stack((x,t), axis=2).reshape(-1, 2)
    x, t = np.hsplit(XT, 2)
    x=x.flatten()
    t=t.flatten()
    
    
    coefs_fit,pcov=scop.curve_fit(function_2D_to_fit, (x,t), img_to_fit.flatten(),bounds=([t_pb_bd[0],K_bd[0],D_bd[0],px_center_bd[0]],[t_pb_bd[1],K_bd[1],D_bd[1],px_center_bd[1]]))
    
    # Compute fitted data
    Y_fit=function_2D_to_fit((x,t),coefs_fit[0],coefs_fit[1],coefs_fit[2],coefs_fit[3])
    Y_fit_2D=Y_fit.reshape((N_t_fit_duration,N_x))
    
    # Compute dimensional quantities
    D_fit=(Size_px**2*coefs_fit[2])/(T_acq/1000) #µm²s-1
    t_bleach_fit=-coefs_fit[0]*T_acq #ms
    K=coefs_fit[1] #non dimensionné
    print('Fit coefficients t_pb(ms)='+str(round(t_bleach_fit))+', K(SU)='+str(round(K))+', D(µm²s-1)='+str(round(D_fit,2)))
    
    #plots
    if plots:  
        "chronographs"
        x=np.linspace(0,(N_x-1)*Size_px,N_x)-((N_x-1)*Size_px)/2
        X_um=np.linspace(0,(N_x-1)*Size_px,N_x)-((N_x-1)*Size_px)/2
        t=np.linspace(0,(N_t_fit_duration-1)*T_acq,N_t_fit_duration)
        T_ms=np.linspace(0,(N_t_fit_duration-1)*T_acq,N_t_fit_duration)
        x,t = np.meshgrid(x,t) 
        
        fig,ax=plt.subplots(1,2,dpi=200)
        "expé"
        img_to_fit_modplot=moving_average2(img_to_fit, 5, 5)
        img_to_fit_modplot[img_to_fit_modplot>1]=1
        ax[0].contourf(x,t,img_to_fit_modplot ,np.linspace(0.4,1.1,15))
        ax[0].set_title('Experimental - '+k_zone+str(k_cell[0]))
        ax[0].set_box_aspect(1.5)
        ax[0].set_xlabel('Distance (µm)')
        ax[0].set_ylabel('Temps (ms)')
        "modèle"
        plotb=ax[1].contourf(x,t,moving_average2(Y_fit_2D, 5, 5) ,np.linspace(0.4,1.1,15))
        ax[1].set_title('Model - '+k_zone+str(k_cell[0]))
        ax[1].set_box_aspect(1.5)
        ax[1].set_xlabel('Distance (µm)')
        fig.suptitle('dec_fit='+str(N_t_decalage_fit*T_acq)+'ms, D='+str(round(D_fit,2))+'µm²s-1')
           
    return D_fit

def P25_build_img_array(name,path,N_rafale,VALUE_background_noise,T_fading,i_postbleach):  
    #"Build averaged FRAP image arrays from .czi files.\"
    T_imgarray_rafale=()
    
    img_full=new_czi_reader.CziFile(path+name+".czi") ##https://allencellmodeling.github.io/aicspylibczi/
    img_full,_=img_full.read_image(C=0)
    img_full=np.squeeze(img_full)    
    img_full=P20_LSM_fuse_lines(img_full)

    N_t_frap,N_px=round(img_full.shape[0]/N_rafale),img_full.shape[1]
    for i_rafale in range(N_rafale):
        img=img_full[N_t_frap*i_rafale:N_t_frap*(i_rafale+1),:]
        
        "check intensité minimale"
        initial_intensity=np.mean(img[i_postbleach-5:i_postbleach,:])
        print('P24_build_img_array (FRAP), prebleach intensity='+str(round(initial_intensity)))
        
        img.astype(float)-VALUE_background_noise
        img[img<0]=0.001
        img=P21_fading_correction(img,T_fading,i_postbleach,i_rafale)
        
        T_imgarray_rafale=T_imgarray_rafale+(img,)
        
    img_fused=np.zeros((N_t_frap,N_px))
    for i_rafale in range(N_rafale):
        img_fused=img_fused+T_imgarray_rafale[i_rafale]
    img_fused=img_fused/N_rafale
    
    return (img_fused,)+T_imgarray_rafale
        
def P21_fading_correction(img,T_fading,i_postbleach,i_rafale=0):
    #"Apply fading correction to FRAP image data.\"
    N_px=img.shape[1]
    N_t_frap=img.shape[0]
    #structure T_fading : 
    #[0] type correction
    #[1] i_postbleach
    #[2] N_pxcellROI (pour [0]=1) // Courbe_fading (pour [0]=2)
    #print(T_fading)
    if T_fading[0]=='':
        return img
    elif T_fading[0]=='pre_bleach_only': 
        img_prebleach_avg=np.mean(img[:i_postbleach,:],axis=0)
        img=np.divide(img,img_prebleach_avg)
        return img
    
    elif T_fading[0]=='pre_bleach+boundaries': 
        N_px_cellROI=T_fading[1]
      
        img_prebleach_avg=np.mean(img[:i_postbleach,:],axis=0)
        img=np.divide(img,img_prebleach_avg)
        
        courbe_ROIcell=(np.mean(img[:,:N_px_cellROI],axis=1)+np.mean(img[:,N_px-N_px_cellROI:],axis=1))/2
        courbe_ROIcell=np.outer(courbe_ROIcell,np.ones((N_px)))
        img=np.divide(img,courbe_ROIcell)
        return img    

def P20_LSM_fuse_lines(img): #Fuse scan lines for LSM microscopy data.
    print(img.shape)
    Nt,Ny,Nx=img.shape
    img2=np.mean(img,axis=1)
    return img2


#%%
"=========================================================================="
"=============== METADATA===================="
"=========================================================================="
path=''
subpath_data=''

path_data=path+subpath_data

file='METADATA_FRAP_EXAMPLE' #example given is one burst of three FRAP repetitions in water+glycerol mixture containing FD20

df_0=pd.read_excel (path+file+'.xlsx',sheet_name='data') 
array_0=pd.DataFrame(df_0).to_numpy()

df_1=pd.read_excel (path+file+'.xlsx',sheet_name='config_1_frap') 
array_1=pd.DataFrame(df_1).to_numpy()

background_value=0 #fluorescence background level to be subtracted from data


"=========================================================================="
"====================== FRAP ANALYSIS ==============================="
"=========================================================================="
"experimental data import"
 

#FRAP numerical parameters
N_px_cellROI=40# number of px on each side of imaged rectangle to be used for fluorescence normalization
parametres_fading=('pre_bleach+boundaries',N_px_cellROI)#fluorescence normalization setup
N_t_decalage_fit=0 #first postbleach image to be used (5 for in-cell measurements)
N_t_fit_duration=25 #fit duration in images (25 for in-cell measurements)
plots=1 #should results be plotted 0 or 1
    
#Import data parameters
array_zone=pd.DataFrame(df_0['zone']).to_numpy() #list of intracellular regions to be analysed
array_nom=pd.DataFrame(df_0['nom']).to_numpy() #list of names to be analysed
array_config=pd.DataFrame(df_0['config']).to_numpy() #list of configs to be analysed
array_num=pd.DataFrame(df_0['num']).to_numpy() #list of file numbers to be analysed
array_type=pd.DataFrame(df_0['type']).to_numpy()#list of data types to be analysed
array_effet=pd.DataFrame(df_0['effet']).to_numpy() #list of effects to be analysed
array_unique_effet=unique_string(array_effet) #unique list of effects to be analysed
array_cell=pd.DataFrame(df_0['cell']).to_numpy() #list of cells to be analysed
array_unique_cell=remove_nans(unique_string(array_cell),'int')
N_rafale_separees=int(pd.DataFrame(df_1['N_rafale_separees']).to_numpy()[0][0]) #number of files corresponding to a given burst repetitions
N_rafale_fichier=int(pd.DataFrame(df_1['N_rafale_fichier']).to_numpy()[0][0])#number of burst repetition in a given file
i_postbleach=int(pd.DataFrame(df_1['n_pre_bleach']).to_numpy()[0][0])#first postbleach image
T_acq=float(pd.DataFrame(df_1['t_acquisition']).to_numpy()[0][0]) #acquisition time in ms
size_px_frap=float(pd.DataFrame(df_1['size_px']).to_numpy()[0][0])#pixel size in um 

L_effet,L_cell,L_zone,L_d_gfp=[],[],[],[]

for i_effet in range (len(array_unique_effet)):
    effet=array_unique_effet[i_effet]
    for k_cell in array_unique_cell:      
        #local metadata extraction
        array_local_zone=array_zone[(array_cell==k_cell)*(array_effet==effet)]
        array_unique_zone=unique_string(array_local_zone)
        array_local_num=array_num[(array_cell==k_cell)*(array_effet==effet)]
        array_local_nom=array_nom[(array_cell==k_cell)*(array_effet==effet)]
        array_local_type=array_type[(array_cell==k_cell)*(array_effet==effet)]
        for k_zone in array_unique_zone:
            array_frap_nums=array_local_num[(array_local_zone==k_zone)*(array_local_type=='frap')] 
            
            #each repetition of a burst is saved in a different .czi file
            if len(array_frap_nums)>0: 
                list_frap_nums=[str(build_num(x)) for x in array_frap_nums]
            else :
                list_frap_nums=[]

            if len(list_frap_nums)==N_rafale_separees: 
                #FRAP computation
                name_0=array_local_nom[(array_local_zone==k_zone)*(array_local_type=='frap')] [0]
                img_array=P25_extraction_data_FRAP(path_data,name_0,list_frap_nums,parametres_fading,background_value,N_rafale_separees,N_rafale_fichier,i_postbleach)
                d_gfp=CLEAN_FRAP_analysis(img_array,i_postbleach,N_t_decalage_fit,N_t_fit_duration,T_acq,size_px_frap,dim=2,plots=plots,info=[k_zone,k_cell]) 
                
                #saving result in lists
                L_effet.append(effet[0])
                L_cell.append(int(k_cell[0]))
                L_zone.append(k_zone)
                L_d_gfp.append(d_gfp)
                
                print(effet[0]+' '+str(k_cell[0])+' '+k_zone+' '+str(round(d_gfp,2)))












    

