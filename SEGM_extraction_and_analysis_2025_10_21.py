#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CURRENT VERSION WRITTEN BY OLIVIER DESTRIAN IN OCTOBER 2025
PAPER "Cytoplasmic crowding acts as a porous medium reducing macromolecule diffusion"
This code allows to compute cell volume and HP and LP regions obstacles fractions
(excluded fractions \Phi_n and \Phi_m) from Zstack GFP fluorescence data

It is divided in two parts: PRE-PROCESSING of raw .czi zstack file

Then manual steps are required:
-manual erasement of other cells on the zstack light saved by PRE-PROCESSING in files_to_be_manually_marked
using basic image editing software as paint, and save of this file in files_manually_marked
-manual tagging of nucleoplasm region (for fluorescence reference, in red (RGB [238,205,125])))
and of the HP (green [92, 163, 99]) and LP (blue [157, 129, 136]) regions where FRAP experiment have $
been conducted. (To help find these regions, pre-processing tags (not at the right Z) with green and blue dots
(these dots do not correspond to green [92, 163, 99]) and blue [157, 129, 136], thus are not counted))
, this using basic image editing software as paint, and save of this file in files_manually_marked.
- HEAVY file has to be pasted from files_to_be_manually_marked to files_manually_marked
Then POST-PROCESSING is conducted to give cell volume, and obstacles fractions in the tagged HP and LP regions


"""

import pandas as pd
from scipy import signal
import copy
import aicspylibczi as new_czi_reader
import math 
import time
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
import numpy as np
from skimage import data, restoration, util,filters, color, morphology
from skimage.segmentation import flood, flood_fill
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_sauvola)
from skimage.filters import try_all_threshold
import aicspylibczi as new_czi_reader

"======================================================="
"============= FUNCTION DEFINITION ===================="
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
    somme=np.sum(w)
    return  signal.convolve2d(x, w, mode='same', boundary='symm')/ somme


#shows zstack as an image mosaic
def P21_plot_square(ZSTACK,name, specific_title='image mosaic',color_mode='auto'): 
    N_z,_,N_x=ZSTACK.shape
    N_square=int(np.sqrt(N_z)) #number of slices corresponding to length of square
    if color_mode=='auto':
        img_plot=np.zeros((N_square*N_x,N_square*N_x))
        for i_x in range(N_square):
            for i_y in range(N_square):
                
                i_z=i_y+N_square*i_x
                if i_z<N_z:
                    img_plot[i_x*N_x:(i_x+1)*N_x,i_y*N_x:(i_y+1)*N_x]=ZSTACK[i_z,:,:]
                else : 
                    img_plot[i_x*N_x:(i_x+1)*N_x,i_y*N_x:(i_y+1)*N_x]=0
    elif color_mode=='green':
        img_plot=np.zeros((N_square*N_x,N_square*N_x,3))
        for i_x in range(N_square):
            for i_y in range(N_square):
                
                i_z=i_y+N_square*i_x
                if i_z<N_z:
                    img_plot[i_x*N_x:(i_x+1)*N_x,i_y*N_x:(i_y+1)*N_x,1]=ZSTACK[i_z,:,:]
                else : 
                    img_plot[i_x*N_x:(i_x+1)*N_x,i_y*N_x:(i_y+1)*N_x,1]=0
        
    plt.figure(dpi=800)
    plt.imshow(img_plot)
    plt.axis('off')
    plt.text((N_square-0.1)*N_x,(N_square-0.1)*N_x,name+' : '+specific_title,size='small',color='white',ha='right')
    


def P23_ZSTACK_PRE_TRAITEMENT(path_data,path_segm_1,name,background_value,charray_color_correspondance,T_relative_coordinates,binning_z,binning_heavy,binning_light,size_px_zstack,size_z_zstack,plots=0):  #tiré du programme newprocess
    def binning(ZSTACK,n_binning):
            N_z,_,N_x=ZSTACK.shape
            ZSTACK_smooth=np.zeros((ZSTACK.shape),dtype=np.uint16)
            for i_z in range(N_z):
                ZSTACK_smooth[i_z,:,:]=np.array(moving_average2(ZSTACK[i_z,:,:], n_binning, n_binning),dtype=np.uint16)#création de l'image binnée 2x2
            ZSTACK=ZSTACK_smooth[:,0:N_x:n_binning,0:N_x:n_binning]
            return ZSTACK    
        
    "1 ===importation ZSTACK RAW==="
    ZSTACK_heavy=new_czi_reader.CziFile(path_data+name+".czi") ##https://allencellmodeling.github.io/aicspylibczi/
    ZSTACK_heavy,_=ZSTACK_heavy.read_image(C=0)
    ZSTACK_heavy=np.squeeze(ZSTACK_heavy)   

    N_z, _, N_x = ZSTACK_heavy.shape
    ZSTACK_heavy[ZSTACK_heavy < np.uint16(background_value)] = np.uint16(background_value) # removing negative values
    ZSTACK_heavy = ZSTACK_heavy-np.uint16(background_value)  # correction background
    "2 ===creation ZSTACK HEAVY==="
    ZSTACK_heavy = ZSTACK_heavy[0:N_z:binning_z, :, :]
    ZSTACK_heavy = binning(ZSTACK_heavy, binning_heavy)
    N_z, _, N_x = ZSTACK_heavy.shape
    ZSTACK_heavy = np.uint16(ZSTACK_heavy)

    np.save(path_segm_1+name+'_HEAVY.npy', ZSTACK_heavy)

    "3 ===creation ZSTACK light==="
    ZSTACK_light = np.array(binning(ZSTACK_heavy, int(binning_light/binning_heavy)),dtype=np.float32)
    plt.imshow(ZSTACK_light[20,:,:])
    N_z, _, N_x = ZSTACK_light.shape
    #"removal of too saturated pixels
    saturation = np.quantile(ZSTACK_light, 0.99)
    ZSTACK_light[ZSTACK_light > saturation] = saturation


    #===saving image globale light==="
    IMAGE_GLOBAL = np.zeros((N_z*N_x, N_x))
    for i_z in range(N_z):
        IMAGE_GLOBAL[i_z*N_x:(i_z+1)*N_x, :] = ZSTACK_light[i_z, :, :]
    mpl_img.imsave(path_segm_1+name+'_light.tiff', IMAGE_GLOBAL, cmap='Greys')    

    "3b ===creation ZSTACK marked ==="
    def compute_and_tag_position_on_img(relative_coordinates, charray_color_correspondance):
        xrel, yrel, zrel, theta1, theta21 = relative_coordinates[1]

        theta1 = -theta1 #angle conversion
        "==="

        # coordinate from image center in um
        x_on_img = xrel*np.cos(np.deg2rad(theta1)) + \
        yrel*np.sin(np.deg2rad(theta1))
        y_on_img = yrel*np.cos(np.deg2rad(theta1)) - \
            xrel*np.sin(np.deg2rad(theta1))

        x_on_img /= size_px_zstack_light  # coordinate from image center in px
        y_on_img /= size_px_zstack_light

        x_on_img += N_x//2  # coordinate from bottom left corner in px
        y_on_img += N_x//2

        # zstack slice closest to indicated frap z coordinate 
        z_on_img = round(zrel/(2*size_z_zstack))

        x_on_img, y_on_img, z_on_img = int(x_on_img), int(y_on_img), int(z_on_img)

        index_color = int(np.where(charray_color_correspondance == relative_coordinates[0])[
                          1])  # relative_coordinates[0] : region name (ex : "HP") !!

        if z_on_img >= 0 and z_on_img < N_z:
            if x_on_img >= 0 and x_on_img < N_x:
                if y_on_img >= 0 and y_on_img < N_x:
                    intensity_bw = np.mean(
                        ZSTACK_marked_color[z_on_img, y_on_img-5:y_on_img+6, x_on_img-5:x_on_img+5, :])
                    intensity_color = np.max(
                        charray_color_correspondance[2, index_color])
                    intensity_factor = 124/intensity_color
                    ZSTACK_marked_color[z_on_img, y_on_img-5:y_on_img+6, x_on_img-5:x_on_img+5, :] = np.array(intensity_factor*np.array(charray_color_correspondance[2, index_color]),dtype=np.uint32)

    "create color matrix"
    ZSTACK_marked = copy.deepcopy(ZSTACK_light)
    ZSTACK_marked /= 255*np.max(ZSTACK_marked)
    size_px_zstack_light = size_px_zstack*binning_light
    N_z, _, N_x = ZSTACK_marked.shape
    ZSTACK_marked_color = np.zeros((N_z, N_x, N_x, 3), dtype='int') 
    for i_channel in range(3):
        ZSTACK_marked_color[:, :, :, i_channel] = 255* \
            copy.deepcopy(ZSTACK_marked)/np.max(ZSTACK_marked)
    ZSTACK_marked_color = 255-ZSTACK_marked_color

    "adding FRAP position points"
    L_zone, L_positions_on_img_frap = [], []
    for i in range(len(T_relative_coordinates)):
        position_on_img = compute_and_tag_position_on_img(
            T_relative_coordinates[i], charray_color_correspondance)
        L_zone.append(T_relative_coordinates[i][0])
        L_positions_on_img_frap.append(position_on_img)

    #"removal of over-satured points"
    saturation = np.quantile(ZSTACK_marked, 0.99)
    ZSTACK_light[ZSTACK_marked > saturation] = saturation
    #===save image global light==="
    IMAGE_GLOBAL = np.zeros((N_z*N_x, N_x, 3), dtype='uint8')
    for i_z in range(N_z):
        IMAGE_GLOBAL[i_z*N_x:(i_z+1)*N_x, :, :] = ZSTACK_marked_color[i_z, :, :]   
    mpl_img.imsave(path_segm_1+name+'_marked.tiff',
                    IMAGE_GLOBAL) 
   
    "4 ===optionnal plotting==="
    if plots:   
        "A=== ZSTACK HEAVY"
        plt.figure(dpi=400)
        plt.imshow(ZSTACK_heavy[15,:,:])
        plt.title('ZSTACK-heavy : binning 2x2')       
        "B=== ZSTACK light"
        plt.figure(dpi=400)
        plt.imshow(ZSTACK_light[15,:,:])
        plt.title('ZSTACK-light : binning 8x8 + saturation')#
        "C=== image_square ZSTACK light"
        P21_plot_square(ZSTACK_light,name,specific_title='ZSTACK_light')
        
    return L_zone,L_positions_on_img_frap

def P25_ZSTACK_SEGM(path_segm,name,charray_color_correspondance,size_px_zstack,size_z_zstack_binned,parameters,plots=[0,0,0]):
    t0=time.time()
    [seuil_segmentation_obstacles,seuil_segmentation_background,binning_heavy,binning_light,FLUO_RATIO,size_critical_opening,size_critical_closing,Radius_closing_bg,window_size_µm,nucleoplasm_reference]=parameters
    [plots_light,plots_heavy,plots_porosity_zones]=plots
    """==============================================================="""
    """MANUALLY EDITED FILE IMPORT AND PROCESSING"""
    """==============================================================="""
    def P21_IMAGECOLUMN_to_ZSTACK(IMAGE_GLOBAL): 
        if len (IMAGE_GLOBAL.shape) ==2 : 
            N_z_N_x,N_x=IMAGE_GLOBAL.shape
            N_z=N_z_N_x//N_x
            ZSTACK=np.zeros((N_z,N_x,N_x),dtype=np.float16)
            for i_z in range(N_z):
                ZSTACK[i_z,:,:]=IMAGE_GLOBAL[i_z*N_x:(i_z+1)*N_x,:]  
            return ZSTACK
        elif len (IMAGE_GLOBAL.shape) ==3 : #if several colors
            N_z_N_x,N_x,N_c=IMAGE_GLOBAL.shape
            N_z=N_z_N_x//N_x
            ZSTACK=np.zeros((N_z,N_x,N_x,N_c))
            for i_z in range(N_z):
                ZSTACK[i_z,:,:,:]=IMAGE_GLOBAL[i_z*N_x:(i_z+1)*N_x,:,:]  
            return ZSTACK
    """==============================================================="""
    """PROCESSING OF MANUALLY EDITED ZSTACK LIGHT AND MARKED"""
    """==============================================================="""
    def P21_importation_ZSTACK_light(path_segm,name,plots=0):
        if os.path.isfile(path_segm+name+'_light.tif'):
            full_name=path_segm+name+'_light.tif'
        elif os.path.isfile(path_segm+name+'_light.tiff'):
            full_name=path_segm+name+'_light.tiff'
        else : 
            print('file '+path_segm+name+" doesnt exist")
        IMAGE_GLOBAL=plt.imread(full_name)
        IMAGE_GLOBAL=IMAGE_GLOBAL[:,:,0]#[:,:,:3]
        #shifting black and white tints
        IMAGE_GLOBAL=np.float32(255-IMAGE_GLOBAL)
        #normalisation
        val_min,val_max=np.min(IMAGE_GLOBAL),np.max(IMAGE_GLOBAL)
        IMAGE_GLOBAL-=val_min
        IMAGE_GLOBAL/=val_max
        #conversion into a zstack
        ZSTACK_light=P21_IMAGECOLUMN_to_ZSTACK(IMAGE_GLOBAL)    
        if plots:
            P21_plot_square(ZSTACK_light,name,specific_title='ZSTACK_light after manual segmentation')
        return ZSTACK_light
    
    
    def P21_importation_traitement_ZSTACK_marked(path_segm,name,charray_color_correspondance,plots=0):
        def P21_marked_position(ZSTACK_marked,color):
            condition_r=ZSTACK_marked[:,:,:,0]==color[0]#value R
            condition_g=ZSTACK_marked[:,:,:,1]==color[1]#value G 
            condition_b=ZSTACK_marked[:,:,:,2]==color[2]#value B  
            condition=condition_r*condition_g*condition_b#the three values have to match what is expected
            positions=np.argwhere(condition)
            return positions
        
        if os.path.isfile(path_segm+name+'_marked.tif'):
            full_name=path_segm+name+'_marked.tif'
        elif os.path.isfile(path_segm+name+'_marked.tiff'):
            full_name=path_segm+name+'_marked.tiff'
        IMAGE_GLOBAL_marked=plt.imread(full_name)
        ZSTACK_marked=P21_IMAGECOLUMN_to_ZSTACK(IMAGE_GLOBAL_marked)
        #4rth channel is useless
        ZSTACK_marked=ZSTACK_marked[:,:,:,:3]
        N_z,_,N_x,_=ZSTACK_marked.shape
        
        "AUTRES MARKER : ZONES MESURES PORO"
        #red color correspond to nucleoplasmic fluorescence rreference region
        color_paint_ref_fluo=np.array([[237,28,36],]) 
        #other colors correspond to regions of interest
        color_paint_others=[np.array(x) for x in charray_color_correspondance[2,:]]
        
        color_paint=np.concatenate((color_paint_ref_fluo,color_paint_others),axis=0)
        
        TUPLE_ZSTACK_region=()
        T_zone_correspondance=()
        for i_marked in range(color_paint.shape[0]):
            #recovery of positions corresponding to each color
            positions_marked=P21_marked_position(ZSTACK_marked,color_paint[i_marked,:])
            if i_marked==0 and positions_marked.size==0 :
                print('THE NUCLEOPLASMIC REFERENCE REGION HAS NOT BEEN TAGGED')
            elif positions_marked.size>0:   
                ZSTACK_region=np.array(np.zeros((N_z,N_x,N_x)),dtype='bool')
                for i in range (positions_marked.shape[0]):
                    #MASK for positions of each region
                    ZSTACK_region[tuple(positions_marked[i,:])]=1
                #tuple of the zstacks mmasks for each marked region (nucleoplasm, HP, LP...)
                TUPLE_ZSTACK_region+=(ZSTACK_region,)
                if i_marked==0:
                    T_zone_correspondance+=('ref_fluo',)
                else :
                    T_zone_correspondance+=(charray_color_correspondance[0,i_marked-1],)
        if plots :  
            for i_marked in range(len(TUPLE_ZSTACK_region)):
                for i_z in range(N_z):
                    if np.sum(TUPLE_ZSTACK_region[i_marked][i_z,:,:])>0:       
                        plt.figure(dpi=200)
                        plt.imshow(TUPLE_ZSTACK_region[i_marked][i_z,:,:])
                        if i_marked==0:
                            plt.title('marquage posfluo (N°0), i_z='+str(i_z))
                        else : 
                            plt.title('marquage N°'+str(i_marked)+', i_z='+str(i_z))
        return TUPLE_ZSTACK_region,T_zone_correspondance
    def P21_courbe_correction_fading(ZSTACK_light,FLUO_RATIO,plots=0):
        N_z,_,N_x=ZSTACK_light.shape
        L_sumsignal=[]
        L_sum_sumsignal=[]
        for i_z in range(N_z):
            L_sumsignal.append(np.sum(ZSTACK_light[i_z,:,:],dtype=np.float64)) #be cautious about possible overflow
            L_sum_sumsignal.append(np.sum(L_sumsignal))
        L_sumsignal/=np.max(L_sumsignal)
        L_sum_sumsignal/=np.max(L_sum_sumsignal)

        #proportion of fadding during the zstack
        FADING=1-FLUO_RATIO
        L_fading_correction=[1] #no fadding correction for the first slice
        for i_z in range(1,N_z):
            Fc=1/(1-L_sum_sumsignal[i_z-1]*FADING)
            L_fading_correction.append(Fc)
            ZSTACK_light[i_z,:,:]*=Fc
            
        if plots : 
            plt.figure(dpi=400)
            plt.plot(L_fading_correction)
            plt.title('Fonction de correction FADING avec FLUO_RATIO='+str(FLUO_RATIO))
            
            P21_plot_square(ZSTACK_light,name,specific_title='ZSTACK_light after fading correction')
        return ZSTACK_light,L_fading_correction
    
    def P21_normalisation_fluo (ZSTACK_light,TUPLE_ZSTACK_region,plots=0):
        #"normalization by the nucleoplasmic region tagged"
        reference_fluo=np.sum(ZSTACK_light*TUPLE_ZSTACK_region[0])/np.sum(TUPLE_ZSTACK_region[0])
        ZSTACK_light/=reference_fluo
        if plots : 
            ZSTACK_light_plot=copy.deepcopy(ZSTACK_light)
            ZSTACK_light_plot[ZSTACK_light_plot>1]=1   
            P21_plot_square(ZSTACK_light_plot,name,specific_title='ZSTACK_light after normalisation')
        return ZSTACK_light
    
    def P21_segmentation_background_light(ZSTACK_light,seuil_segmentation_background,Radius,plots=0):
        N_z,_,N_x=ZSTACK_light.shape
        #threshold at seuil_segmentation_background
        ZSTACK_light_BGsegm=copy.deepcopy(ZSTACK_light)
        ZSTACK_light_BGsegm[ZSTACK_light_BGsegm<seuil_segmentation_background]=0
        ZSTACK_light_BGsegm[ZSTACK_light_BGsegm>seuil_segmentation_background*0.99]=1
        ZSTACK_light_BGsegm=np.array(ZSTACK_light_BGsegm,dtype='bool')
        if plots : 
            P21_plot_square(ZSTACK_light_BGsegm,name,specific_title='ZSTACK_light background trigger')  
            
        #adding border pixel to images for better closing process
        ZSTACK_light_BGsegm_crop=np.zeros((N_z,N_x+4*Radius,N_x+4*Radius),dtype='bool')
        ZSTACK_light_BGsegm_crop[:,2*Radius:N_x+2*Radius,2*Radius:N_x+2*Radius]=ZSTACK_light_BGsegm
        #applying closing for each slice
        for i_z in range(N_z):
            ZSTACK_light_BGsegm_crop[i_z,:,:]=morphology.closing(ZSTACK_light_BGsegm_crop[i_z,:,:],morphology.disk(Radius))#[:3000,:3000], ft)
        #erasing added image borders
        ZSTACK_light_BGsegm=ZSTACK_light_BGsegm_crop[:,2*Radius:N_x+2*Radius,2*Radius:N_x+2*Radius]
        
        if plots : 
            P21_plot_square(ZSTACK_light_BGsegm,name,specific_title='ZSTACK_light background closing')
            ZSTACK_light_plot=np.array(copy.deepcopy(ZSTACK_light*ZSTACK_light_BGsegm),dtype=np.float16)
            ZSTACK_light_plot[ZSTACK_light_plot>1]=1 
            P21_plot_square(ZSTACK_light_plot,name,specific_title='ZSTACK_light after background segmentation')  

        return ZSTACK_light_BGsegm
    
    "1 ===importation ZSTACK light==="
    "sert à segmenter la cellule à moindre cout"
    ZSTACK_light=P21_importation_ZSTACK_light(path_segm,name,plots=plots_light)
    
    "2 ===importation ZSTACK marked==="
    "sert pour taguer les zones marquées manuellement sur paint"
    TUPLE_ZSTACK_region,T_zone_correspondance=P21_importation_traitement_ZSTACK_marked(path_segm,name,charray_color_correspondance,plots=plots_light)    
    "3 ===fonction de correction du fading==="
    "sert à corriger l'évolution de la quantité de signal tranche après tranche"
    
    ZSTACK_light,L_fading_correction=P21_courbe_correction_fading(ZSTACK_light,FLUO_RATIO,plots=plots_light)
    "4 ===normalisation fluo et segmentation Background==="
    ZSTACK_light=P21_normalisation_fluo (ZSTACK_light,TUPLE_ZSTACK_region,plots=plots_light)
    
    "5 ===segmentation BG via closing==="
    
    ZSTACK_light_closingBG=P21_segmentation_background_light(ZSTACK_light,seuil_segmentation_background,Radius_closing_bg,plots=plots_light)
    
    "6===calcul volume sur ZSTACK_LIGHT (maj 2023_10_16)"
    V_cell=np.sum(ZSTACK_light_closingBG)*(size_px_zstack*binning_light)**2*size_z_zstack_binned#*2 car utilisation ZSTACK heavy avec binning 2x2
    print('Volume cellule (µm3)='+str(V_cell))
    
    
    """==============================================================="""
    """TRAITEMENT DU ZSTACK heavy"""
    """==============================================================="""
    def P21_convert_light_to_heavy(ZSTACK_light,N_x_heavy,array_dtype):
        N_z,N_x,_=ZSTACK_light.shape
        light_z,light_row,light_col=np.indices((N_z,N_x_heavy,N_x_heavy))
        
        float_binning_ratio=N_x_heavy/N_x
        
        light_row,light_col=light_row/float_binning_ratio,light_col/float_binning_ratio   
        light_row,light_col=np.array(light_row,dtype=np.uint16),np.array(light_col,dtype=np.uint16)
        
        ZSTACK_heavy=np.array(ZSTACK_light[light_z,light_row,light_col],dtype=array_dtype)

        return ZSTACK_heavy
    def P21_segmentation_background_heavy(ZSTACK_heavy,ZSTACK_light_closingBG,plots=0):
        #"conversion format light vers HEAVY"   
        N_z,N_x,_=ZSTACK_heavy.shape
        #binning_ratio=int(8/2)
        ZSTACK_heavy_closingBG=P21_convert_light_to_heavy(ZSTACK_light_closingBG,N_x,array_dtype='bool')#P21_convert_light_to_heavy(ZSTACK_light_closingBG,binning_ratio=int(binning_light/binning_heavy))
        #application au ZSTACK heavy
        ZSTACK_heavy*=ZSTACK_heavy_closingBG
        
        if plots : 
            P21_plot_square(ZSTACK_heavy,name,specific_title='ZSTACK_heavy after background segmentation')       
        return ZSTACK_heavy,ZSTACK_heavy_closingBG
    def P21_normalisation_fluo_heavy(ZSTACK_heavy,ZSTACK_heavy_ref_fluo,plots=0):
        N_z,N_x,_=ZSTACK_heavy.shape
        sum_fluo,sum_npixels=0,0
        for i_z in range(N_z):
            img_heavy_ref_fluo=ZSTACK_heavy_ref_fluo[i_z,:,:]
            if np.sum(img_heavy_ref_fluo)>0:
                img_heavy=ZSTACK_heavy[i_z,:,:]                      
                img_heavy_times_ref_fluo=img_heavy*img_heavy_ref_fluo
                                     
                sum_fluo+=np.sum(img_heavy_times_ref_fluo,dtype=np.float64)
                sum_npixels+=np.sum(img_heavy_ref_fluo,dtype=np.float64)

        reference_fluo=sum_fluo/sum_npixels
        ZSTACK_heavy/=reference_fluo
        
        if plots : 
            for i_z in range(N_z):
                if np.sum(ZSTACK_heavy_ref_fluo[i_z,:,:])>0:
                    plt.figure(dpi=400)
                    plt.imshow(ZSTACK_heavy_ref_fluo[i_z,:,:])
                    plt.figure(dpi=400)
                    plt.imshow(ZSTACK_heavy[i_z,:,:])
                    plt.figure(dpi=400)
                    plt.imshow(ZSTACK_heavy[i_z,:,:]*ZSTACK_heavy_ref_fluo[i_z,:,:])

            P21_plot_square(ZSTACK_heavy,name,specific_title='ZSTACK_heavy after normalisation') 
        return ZSTACK_heavy     
    def P23_troncage_zstack(ZSTACK_heavy,TUPLE_ZSTACK_region,ZSTACK_light_closingBG,L_fading_correction):
        "A- récupération des tranches des zones du TUPLE_ZSTACK_region"
        L_z_troncage=[]
        for i in range(len(TUPLE_ZSTACK_region)):
            for j in range(len(TUPLE_ZSTACK_region[i])):
                if np.sum(TUPLE_ZSTACK_region[i][j,:,:])>0:
                    L_z_troncage.append(j)
        L_z_troncage=np.sort(unique_string(L_z_troncage))
        
        "A- récupération du ZSTACK_heavy_tronqué"
        ZSTACK_heavy=ZSTACK_heavy[L_z_troncage,:,:]
        
        ZSTACK_light_closingBG=ZSTACK_light_closingBG[L_z_troncage,:,:]

        TUPLE_ZSTACK_region_troncage=()
        for i in range(len(TUPLE_ZSTACK_region)):
            TUPLE_ZSTACK_region_troncage+=(TUPLE_ZSTACK_region[i][L_z_troncage,:,:],)
            
        L_fading_correction=np.array(L_fading_correction)[L_z_troncage]

        return ZSTACK_heavy,TUPLE_ZSTACK_region_troncage,ZSTACK_light_closingBG,L_fading_correction,L_z_troncage       
    def P24_segmentation_echelles_porosite(ZSTACK_heavy,ZSTACK_heavy_closingBG,window_size_µm=2.0,plots=plots_heavy):
        window_size_px = math.ceil(window_size_µm/(size_px_zstack*binning_heavy))
        if window_size_px%2==0:
            window_size_px+=1
            
        ZSTACK_region_hyaloplasm=np.array(np.zeros((N_z,N_x,N_x)),dtype='bool')
        ZSTACK_region_macroobstacles=np.array(np.zeros((N_z,N_x,N_x)),dtype='bool')
        
        for i_z in range(N_z):              
            image = ZSTACK_heavy[i_z,:,:]
            thresh_sauvola = threshold_sauvola(image, window_size=window_size_px)
            binary_sauvola = image > thresh_sauvola
            "C - segmentation en deux étapes : opening puis closing"
            binary_opening=morphology.opening(binary_sauvola, morphology.disk(math.ceil(size_critical_opening/(binning_heavy*size_px_zstack))))#1                        
            binary_closing=morphology.closing(binary_opening, morphology.disk(math.ceil(size_critical_closing/(binning_heavy*size_px_zstack))))#2

            if plots : 
                plt.figure(dpi=400)    
                plt.imshow(image)
                plt.title('original image')
                plt.axis('off')
                
                plt.figure(dpi=400)
                plt.imshow(binary_sauvola)
                plt.title('Sauvola Threshold')
                plt.axis('off')
                
                plt.figure(dpi=400)
                plt.imshow(binary_closing)   
                plt.title('after opening+closing')
                plt.axis('off')
         
            ZSTACK_region_hyaloplasm[i_z,:,:]=binary_closing
            image_closingBG=ZSTACK_heavy_closingBG[i_z,:,:]
            ZSTACK_region_macroobstacles[i_z,:,:]=(1-binary_closing)*image_closingBG
        
        return ZSTACK_region_hyaloplasm,ZSTACK_region_macroobstacles
    
    def P25_calcul_fractions_volumiques(ZSTACK_heavy,ZSTACK_region_macroobstacles,ZSTACK_region_hyaloplasm,ZSTACK_heavy_closingBG,method,nucleoplasm_reference):
        if method=='binary_corrected':
            #\Phi_m is computed according to binary method (micro-obstacle segmented surface/total region surface)
            F_Phi_m=np.sum(ZSTACK_region_macroobstacles)/np.sum(ZSTACK_heavy_closingBG) 
            #Accessible volume fraction is computed according to proportional method (average region fluorescence/reference fluorescence)
            F_cytosol=np.sum(ZSTACK_heavy)/np.sum(ZSTACK_heavy_closingBG)*nucleoplasm_reference
            #\Phi_n is the remaining obstacle fraction
            F_Phi_n=1-F_cytosol-F_Phi_m         
        else:
            print('error this method is not implemented')

        return F_cytosol,F_Phi_n,F_Phi_m
    
    def P23_plot_segmented_region(IMG_heavy,IMG_region_macroobstacles,IMG_region_hyaloplasm,IMG_heavy_closingBG,IMG_heavy_region,fractions,method,plots_porosity_zones=plots_porosity_zones):
        F_cyto,F_obs_micro,F_obs_macro=fractions #récupération des fractions volumiques
        "calcul du contour de la zone étudiée à l'aide du gradient -> pour affichage"
        grad_img=np.gradient(np.array(IMG_heavy_region,dtype=np.float16))
        grad_0=grad_img[0]
        grad_1=grad_img[1]
        grad_0[grad_0!=0]=1
        grad_1[grad_1!=0]=1
        contour_region=grad_img[0]+grad_img[1]
        contour_region[contour_region!=0]=1
        contour_region=np.array(contour_region,dtype='bool')
        img_blue=np.array(morphology.dilation(contour_region,morphology.disk(2)),dtype='bool')
        "importation des canaux vert et rouge "
        if method=='binary_corrected':
            img_red=copy.deepcopy(IMG_region_macroobstacles)
            img_green=IMG_region_hyaloplasm*IMG_heavy

        img_green[img_blue]=0
        img_red[img_blue]=0
        "renormalisation pour optimisation de l'affichage des canaux vert et rouge"
        img_green_nan=copy.deepcopy(img_green)  
        img_green_nan[img_green_nan==0]=np.nan
        saturation=np.nanquantile(img_green_nan,0.99)
        img_green/=saturation
        "création de l'image rgb"
        img_to_plot=np.zeros((N_x,N_x,3))
        img_to_plot[:,:,0]=img_red
        img_to_plot[:,:,1]=img_green
        img_to_plot[:,:,2]=img_blue
        "affichage"
        plt.figure(dpi=200)
        plt.imshow(img_to_plot)
        plt.text(0.97*N_x,0.75*N_x,'method='+method,color='white',size='small',ha='right')
        plt.text(0.97*N_x,0.80*N_x,r'$\epsilon_{obstacle MACRO} =$'+str(round(100*F_obs_macro,1))+'%',color='white',size='small',ha='right')
        plt.text(0.97*N_x,0.85*N_x,r'$\epsilon_{obstacle micro} =$'+str(round(100*F_obs_micro,1))+'%',color='white',size='small',ha='right')
        plt.text(0.97*N_x,0.90*N_x,r'$\epsilon_{cytosol} =$'+str(round(100*F_cyto,1))+'%',color='white',size='small',ha='right')
        plt.text(0.03*N_x,0.05*N_x,'Region='+T_zone_correspondance[i_region],color='white',size='small')
        plt.text(0.03*N_x,0.10*N_x,'Z='+str(round(i_z*size_z_zstack_binned,2))+'µm (i_z='+str(i_z)+')',color='white',size='small')
        plt.text(0.03*N_x,0.15*N_x,'file='+name,color='white',size='small')
        plt.axis('off')


        
    
    
    
    "1 ===importation ZSTACK heavy .npy==="
    ZSTACK_heavy=np.array(np.load(path_segm+name+'_HEAVY.npy'),dtype=np.float16)
    N_z,N_x,_=ZSTACK_heavy.shape
    
    "1bis===== TRONCAGE DU ZSTACK-HEAVY (maj du 2023 10 16)"            
    troncage_heavy=1
    if troncage_heavy:
        ZSTACK_heavy,TUPLE_ZSTACK_region,ZSTACK_light_closingBG,L_fading_correction,L_z_troncage=P23_troncage_zstack(ZSTACK_heavy,TUPLE_ZSTACK_region,ZSTACK_light_closingBG,L_fading_correction)
        N_z,N_x,_=ZSTACK_heavy.shape
    
    "2 ===segmentation BG_heavy via BG_light==="
    ZSTACK_heavy,ZSTACK_heavy_closingBG=P21_segmentation_background_heavy(ZSTACK_heavy,ZSTACK_light_closingBG,plots=plots_heavy)    
            
    "3 ===correction fading==="
    for i_z in range(N_z):
        ZSTACK_heavy[i_z,:,:]*=L_fading_correction[i_z]
    
    "4 ===normalisation porosité==="
    ZSTACK_heavy_ref_fluo=np.array(P21_convert_light_to_heavy(TUPLE_ZSTACK_region[0],N_x,array_dtype='bool'))
    ZSTACK_heavy=P21_normalisation_fluo_heavy(ZSTACK_heavy,ZSTACK_heavy_ref_fluo,plots=plots_heavy)
    
    "6 ===segmentation des echelles micro et macro ==="   
    ZSTACK_region_hyaloplasm,ZSTACK_region_macroobstacles=P24_segmentation_echelles_porosite(ZSTACK_heavy,ZSTACK_heavy_closingBG,window_size_µm=window_size_µm,plots=plots_heavy)   

    
    "7 ===calcul et affichage des poro micro et méso ==="           
    L_zone,L_cellvol,L_method_bin,L_fraction_cytosol,L_fraction_Phi_n,L_fraction_Phi_m=[],[],[],[],[],[]
    for i_region in range(len(TUPLE_ZSTACK_region)): #for i_region==0, it is the reference cytoplasmic region
        if T_zone_correspondance[i_region]!='ref_fluo':               
            ZSTACK_heavy_region=P21_convert_light_to_heavy(TUPLE_ZSTACK_region[i_region],N_x,array_dtype='bool') 
            
            "===2023_10_04 -- sélection des tranches utiles uniquement"
            N_z,N_x,_=ZSTACK_heavy_region.shape
            L_z=[]
            for i_z in range(N_z):
                if np.sum(ZSTACK_heavy_region[i_z,:,:])>0:
                    L_z.append(i_z)
            reduced_ZSTACK_heavy=ZSTACK_heavy[L_z,:,:]   
            reduced_ZSTACK_heavy_region=ZSTACK_heavy_region[L_z,:,:]
            reduced_ZSTACK_region_macroobstacles=ZSTACK_region_macroobstacles[L_z,:,:]
            reduced_ZSTACK_region_hyaloplasm=ZSTACK_region_hyaloplasm[L_z,:,:]
            reduced_ZSTACK_heavy_closingBG=ZSTACK_heavy_closingBG[L_z,:,:]
            
            r_heavy_times_heavy_region=reduced_ZSTACK_heavy*reduced_ZSTACK_heavy_region
            r_region_macroobstacles_times_heavy_region=reduced_ZSTACK_region_macroobstacles*reduced_ZSTACK_heavy_region
            r_region_hyaloplasm_times_heavy_region=reduced_ZSTACK_region_hyaloplasm*reduced_ZSTACK_heavy_region
            r_heavy_closingBG_times_heavy_region=reduced_ZSTACK_heavy_closingBG*reduced_ZSTACK_heavy_region
   
            L_zone.append(T_zone_correspondance[i_region])
            L_cellvol.append(V_cell)
        
            "porositiy and obstacle fractions computation in each cytoplasmic region using the so-called "binary corrected" segmentation method (see Supporting Appendix 1E)
            method='binary_corrected'
            fractions=P25_calcul_fractions_volumiques(r_heavy_times_heavy_region,r_region_macroobstacles_times_heavy_region,r_region_hyaloplasm_times_heavy_region,r_heavy_closingBG_times_heavy_region,method,nucleoplasm_reference)
            L_method_bin.append(method)
            L_fraction_cytosol.append(fractions[0])
            L_fraction_Phi_n.append(fractions[1])
            L_fraction_Phi_m.append(fractions[2])
            for i_z in range(N_z):
                if np.sum(ZSTACK_heavy_region[i_z,:,:])>0:
                    P23_plot_segmented_region(ZSTACK_heavy[i_z,:,:],ZSTACK_region_macroobstacles[i_z,:,:],ZSTACK_region_hyaloplasm[i_z,:,:],ZSTACK_heavy_closingBG[i_z,:,:],ZSTACK_heavy_region[i_z,:,:],fractions,method,plots_porosity_zones=plots_porosity_zones)
                            

    if len(TUPLE_ZSTACK_region)==1:
        L_zone.append('cell')
        L_cellvol.append(V_cell)
        L_method_bin.append(np.nan)
        L_fraction_cytosol.append(np.nan)
        L_fraction_Phi_n.append(np.nan)
        L_fraction_Phi_m.append(np.nan)

            
            
    charray_segm_subresults=np.array([L_zone,
                                    L_cellvol,
                                    L_method_bin,
                                    L_fraction_cytosol,
                                    L_fraction_Phi_n,
                                    L_fraction_Phi_m])
    t1=time.time()
    print('P24_SEGM, temps total (s)='+str(round(t1-t0,2)))
    print(charray_segm_subresults)
    return charray_segm_subresults
#%%
"=========================================================================="
"===============MANUAL INPUTS===================="
"=========================================================================="
path=''
subpath_data='data/'
subpath_segm_1='files_to_be_manually_marked/'
subpath_segm_2='files_manually_marked/'

path_data=path+subpath_data

path_segm_1=path+subpath_segm_1
path_segm_2=path+subpath_segm_2
file='METADATA_SEGM_EXAMPLE'

df_0=pd.read_excel (path+file+'.xlsx',sheet_name='data') 
array_0=pd.DataFrame(df_0).to_numpy()

df_3=pd.read_excel (path+file+'.xlsx',sheet_name='config_3_zstack')
array_3=pd.DataFrame(df_3).to_numpy()

background_value=7.5 #intensity of fluorescence background

plots=1 #should intermediate plots be printed

#numerical parameters
binning_heavy=2 #xy binning used for the "HEAVY" zstack file, usefull for micro/nano precise segmentation
binning_light=8 #xy binning used for the "LIGHT" zstack file, usefull for volume computation and manual operations
binning_z=1#no binning in the Z direction

#NUMERICAL PARAMETERS
nucleoplasm_reference=0.85 #in our study nucleoplasm reference was set to 0.85 for 300mOsm
FLUO_RATIO=0.8 #empirical value to compensate for fluorescence bleaching between beginning and end of confocal zstack
seuil_segmentation_obstacles=-1 #THRESHOLD FOR SEGMENTATION, OBSOLETE as SAVOLA THRESHOLDING is now used
window_size_µm=2.0#window size for savola thresholding, in µm
size_critical_opening=0.08#Morphological micro-obstacle opening radius parameter in µm
size_critical_closing=0.16#Morphological micro-obstacle closing radius parameter in µm
   
seuil_segmentation_background=0.5#MULTIPLICATIVE CONSTANT FOR BACKGROUND THRESHOLDING


#definition of the colors that should be recognized by the PROCESSING
#nucleoplasm_fluorescence_reference is [238,205,125]
charray_color_correspondance=np.array([['HP'     ,'LP','whatever_region'],
                                       ['green' ,'blue','purple'],
                                       [(34,177,76),(63,72,204),(163,73,164)]],dtype='object')


"=========================================================================="
"====================== PRE-PROCESSING FOR CELL VOLUME AND EXCLUDED OBSTACLE FRACTIONS \Phi_n and \Ph_m COMPUTATIONS ==============================="
"=========================================================================="

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

"importation des données expérimentales"
#indices, positions et angles
array_pos_x=pd.DataFrame(df_0['pos_x']).to_numpy()
array_pos_y=pd.DataFrame(df_0['pos_y']).to_numpy()
array_pos_z=np.array(pd.DataFrame(df_0['pos_z_low']).to_numpy(),dtype=np.float32)
array_angle=pd.DataFrame(df_0['angle']).to_numpy()
size_px_zstack=float(pd.DataFrame(df_3['size_px']).to_numpy()[0][0])
Radius_closing_bg=int(1/(binning_light*size_px_zstack))#Morphological background closing radius parameter in µm
size_z_zstack=float(pd.DataFrame(df_3['size_z']).to_numpy()[0][0])

L_effet,L_cell,L_zone,L_positions_on_img_frap=[],[],[],[]
for i_effet in range (len(array_unique_effet)):
    effet=array_unique_effet[i_effet]  
    
    array_unique_local_cell=unique_string(array_cell[array_effet==effet])
    
    for k_cell in array_unique_local_cell :
        #local metadata
        array_local_num=array_num[(array_cell==k_cell)*(array_effet==effet)]
        array_local_nom=array_nom[(array_cell==k_cell)*(array_effet==effet)] 
        
        array_local_type=array_type[(array_cell==k_cell)*(array_effet==effet)]
        
        array_local_zone=array_zone[(array_cell==k_cell)*(array_effet==effet)]
        array_unique_zone=unique_string(array_local_zone)
        array_local_effet=array_effet[(array_cell==k_cell)*(array_effet==effet)]
        
        array_local_pos_x=array_pos_x[(array_cell==k_cell)*(array_effet==effet)]
        array_local_pos_y=array_pos_y[(array_cell==k_cell)*(array_effet==effet)]
        array_local_pos_z=array_pos_z[(array_cell==k_cell)*(array_effet==effet)]
        array_local_angle=array_angle[(array_cell==k_cell)*(array_effet==effet)]
        
        
        zstack_position=np.where(array_local_type=='zstack')[0][0]
        zstack_num=array_local_num[zstack_position]
        zstack_coordinates=array_local_pos_x[zstack_position],array_local_pos_y[zstack_position],array_local_pos_z[zstack_position],array_local_angle[zstack_position]
        T_relative_coordinates=()
        for k_zone in array_unique_zone:
            #adding FRAP experiments positions
            array_frap_positions=np.where((array_local_zone==k_zone)*(array_local_nom=='frap'))[0]
            if array_frap_positions.size==1:
                frap_position=array_frap_positions[0]
                frap_coordinates=array_local_pos_x[frap_position],array_local_pos_y[frap_position],array_local_pos_z[frap_position],array_local_angle[frap_position]            
                relative_coordinates=np.array(frap_coordinates[:3])-np.array(zstack_coordinates[:3])
                angles=np.array([array_local_angle[zstack_position],array_local_angle[frap_position]-array_local_angle[zstack_position]])#0 : zstack angle; 1 : frap angle relative to zstack
                relative_coordinates=np.concatenate((relative_coordinates,angles))
                T_relative_coordinates+=((k_zone, relative_coordinates,),)

        name='zstack-p-'+build_num(zstack_num)
        
        #PRE-PROCESSING
        L_zone_plus,L_positions_on_img_frap_plus=P23_ZSTACK_PRE_TRAITEMENT(path_data,path_segm_1,name,background_value,charray_color_correspondance,T_relative_coordinates,binning_z,binning_heavy,binning_light,size_px_zstack,size_z_zstack,plots)
#         for i in range(len(L_zone_plus)):
#             L_effet.append(effet[0])
#             L_cell.append(int(k_cell))
#             L_zone.append(L_zone_plus[i])
#             L_positions_on_img_frap.append(L_positions_on_img_frap_plus[i])

# charray_positions_on_img_frap=np.array([L_effet,
#                                         L_cell,
#                                         L_zone,
#                                         L_positions_on_img_frap],dtype='object')
 


"""============MANUAL PROCESSING EXPECTED AT THIS POINT=================""" 
stop_here=np.stop_here #only run next block if pro-processing HAS BEEN already conducted
#%%
"=========================================================================="
"====================== POST-PROCESSING FOR CELL VOLUME AND EXCLUDED OBSTACLE FRACTIONS \Phi_n and \Ph_m COMPUTATIONS ==============================="
"=========================================================================="

#PROCESSING
plots=[plots]*3# should plots be printed 0 or 1
size_z_zstack_binned=size_z_zstack*binning_z
parameters=[seuil_segmentation_obstacles,seuil_segmentation_background,binning_heavy,binning_light,FLUO_RATIO,size_critical_opening,size_critical_closing,Radius_closing_bg,window_size_µm,nucleoplasm_reference] 
for i_effet in range (len(array_unique_effet)):
    effet=array_unique_effet[i_effet]  

    array_unique_local_cell=unique_string(array_cell[array_effet==effet])
    for i_cell in range(len(array_unique_local_cell)):
        k_cell=array_unique_local_cell[i_cell]
        print(effet[0]+' '+str(int(k_cell)))
        
        #local metadata
        array_local_num=array_num[(array_cell==k_cell)*(array_effet==effet)]
        array_local_nom=array_nom[(array_cell==k_cell)*(array_effet==effet)]  
        array_local_type=array_type[(array_cell==k_cell)*(array_effet==effet)]
        array_local_zone=array_zone[(array_cell==k_cell)*(array_effet==effet)]
        array_unique_zone=unique_string(array_local_zone)
        array_local_pos_x=array_pos_x[(array_cell==k_cell)*(array_effet==effet)]
        array_local_pos_y=array_pos_y[(array_cell==k_cell)*(array_effet==effet)]
        array_local_pos_z=array_pos_z[(array_cell==k_cell)*(array_effet==effet)]
        array_local_angle=array_angle[(array_cell==k_cell)*(array_effet==effet)]
        
        #zstack coordinates
        zstack_position=np.where(array_local_type=='zstack')[0][0]
        zstack_num=array_local_num[zstack_position]
        zstack_coordinates=array_local_pos_x[zstack_position],array_local_pos_y[zstack_position],array_local_pos_z[zstack_position],array_local_angle[zstack_position]

        #computation cell volume and obstacle volume fractions
        name='zstack-p-'+build_num(zstack_num)       
        charray_segm_results_newsub=P25_ZSTACK_SEGM(path_segm_2,name,charray_color_correspondance,size_px_zstack,size_z_zstack_binned,parameters,plots=plots)
        
        #charray_segm_results_newsub format
        #1rst line: region name
        #2nd line:volume in um3
        #3rd segmentation method name
        #4th: accessible volume fraction
        #5th: nano-obstacle excluded fraction \Phi_n
        #6th: micro-obstacle fraction \Phi_m
        
















    
