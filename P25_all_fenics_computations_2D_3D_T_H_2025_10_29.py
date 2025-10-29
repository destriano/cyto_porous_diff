#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CURRENT VERSION WRITTEN BY OLIVIER DESTRIAN IN OCTOBER 2025
PAPER "Cytoplasmic crowding acts as a porous medium reducing macromolecule diffusion"
This code allows to generate several 2D and 3D periodic mesh types en solve closure problems in tortuosity and permeability
Results can then be used to estimate porous tortuosity and hydrodynmaic hindrance as in manuscript Fig. 5
"""
#Done with help from:
#https://fenicsproject.org/olddocs/dolfin/1.3.0/python/demo/documented/stokes-iterative/python/documentation.html
#https://fenicsproject.discourse.group/t/converting-n-s-example-to-use-a-mixed-space/2213
#https://gmsh.info/doc/texinfo/gmsh.html

"functions adapted to 3D random computations"
#import P20old_3D_gmshfenics_functions_from2022_3D_random_media  as P20old 
import P25_all_fenics_functions_2D_3D_T_H.py as functions 

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time


#%%
'BLOCK TO RUN TORTUOSITY AND PERMEABILITY COMPUTATIONS FOR 2D CD' 
geom=2 #2 : 2D centered disk
R_GFP=2.3 #diffusive particle physical radius
R_fibre=4 #fiber physical radius

Mesh_size=0.02 #square size is 1.0

show_geometry=1 #shows the meshed geometry on a separate gmsh windows, but stops the program until the gmsh window is closed

L_R=[0.2] #numerical obstacle radii list (square size is 1.0)

for i_R in range(len(L_R)):
    Radius=L_R[i_R] #numerical obstacle radius
    
    #useless parameters
    N_obstacles=1;file_name='OLI16_mesh_2D';Radius_inerte=0;Length=1;MATRICE_OBSTACLES=np.zeros((100,8))
    refine_obstacle=1
    
    functions.P18_2D_gmsh_generator(Radius,Radius_inerte,Length,Mesh_size,geom,N_obstacles,file_name,MATRICE_OBSTACLES,refine_obstacle,show_geometry=show_geometry)
    functions.OLI16_conversion_maillage_2D(file_name)
    
    "tortuosity computation"
    (Res_diff,_,_)=functions.OLI18_solver_diff_2D(file_name)
    
    obstacle_fraction=1-Res_diff[0]
    tortuosity=Res_diff[1]/Res_diff[0]
    
    txt_to_print_tort=f"epsilon_obstacle_excluded={obstacle_fraction} ; tortuosity T⁻1 (x)={tortuosity}  " #'poro='+str(round(a[0],5)),', Perm_adim'+str(np.round(np.array([a[1]/a[0]]),5))+', ID='+str(ID)
    print(txt_to_print_tort)

    "permeability computation"
    (Poro,Perm_adim)=functions.OLI19_solver_perm_2D(file_name)

    #going from numerical non-dimensional permeability to physical dimensional peermeability in [m²]
    R_num=Radius 
    Ratio_num=R_fibre/R_num
    Perm=1e-18*Perm_adim*Ratio_num**2
    
    txt_to_print_perm=f"epsilon_obstacle_occupied={obstacle_fraction} ; non-dimensional permeability(x)={Perm_adim}; dimensional permeability [m²] (x)={Perm}      "   
    print(txt_to_print_perm)
    
#%%
'BLOCK TO RUN TORTUOSITY AND PERMEABILITY COMPUTATIONS FOR 3D CFC' 
result_type='tort+perm'#analysis to be run: 'tort+perm'
file_prename='P21_random_3D_' #intermediate files naming
file_name='P21_mesh_3D' #intermediate mesh file name

show_geometry=1 #shows the meshed geometry on a separate gmsh windows, but stops the program until the gmsh window is closed

R_GFP=2.3 #physical parameter for GFP RADIUS
R_ribosome=12.5 #physical parameter for sphere radius 

Mesh_size=0.3 #default 0.3 ;largest size of the adaptative mesh size
geom=0.2 #geom 0.2 corrresponds to 3D-CFC configuration
 
L_R=np.array([0.2])#numerical obstacle radii list
for i_R in range(len(L_R)):
    Radius=L_R[i_R]

    ID=round(time.time())
    mesh_name=f"P21_3D_for_{result_type}_ID_{ID}"

    Radius_impenetrable=1 #useless parameter
    N_obstacles=1 #useless parameter
    Length=1 # useless parameter
    functions.P20_3D_gmsh_generator(Radius,Radius_impenetrable,Length,Mesh_size,geom,N_obstacles,mesh_name,show_geometry=show_geometry)
    functions.OLI16_conversion_maillage_3D(mesh_name)
    a,_,_,_=functions.OLI16_solver_diff_3D(mesh_name)
    txt_to_print_tort=f"epsilon_obstacle_excluded={1-a[0]} ; tortuosity T⁻1 (x,y,z)={a[1]/a[0]};{a[2]/a[0]};{a[3]/a[0]}  " #'poro='+str(round(a[0],5)),', Perm_adim'+str(np.round(np.array([a[1]/a[0]]),5))+', ID='+str(ID)
    print(txt_to_print_tort)

    #the same mesh is reused for permeability: it is the same numerical geometry, BUT DOES NOT correspond to same physical configuration
    #as tortuosity is computed on accessible fluid fraction, while permeability is computed on true physical fluid fraction
    b=functions.OLI18_solver_perm_3D(mesh_name)
    R_num=Radius
    Ratio_num=R_ribosome/R_num  
    dimensional_perm=b[1]*(1e-18)*Ratio_num**2  #going from numerical non-dimensional permeability to physical dimensional peermeability in [m²]

    txt_to_print_perm=f"epsilon_obstacle_occupied={1-b[0]} ; non-dimensional permeability(x)={b[1]}; dimensional permeability [m²] (x)={dimensional_perm}      "   
    print(txt_to_print_perm)

            
#%%
'BLOCK TO RUN TORTUOSITY AND PERMEABILITY COMPUTATIONS FOR 3D RANDOM FIBROUS' 
result_type='tort+perm'#analysis to be run: 'tort+perm'
file_prename='P21_random_3D_' #intermediate files naming
file_name='P21_mesh_3D' #intermediate mesh file name

show_geometry=1 #shows the meshed geometry on a separate gmsh windows, but stops the program until the gmsh window is closed

R_GFP=2.3 #physical parameter for GFP RADIUS
R_factin=4 #physical parameter for fiber radius 

N_obstacles=5#number of fibers
Mesh_size=0.3 #default 0.3 ;largest size of the adaptative mesh size
geom=2 #geom 2 corresponds to 3D random fibrous geometry type
 
Length=0.95 # numerical length of the fibers (numerical box size is 1)
L_R=np.array([0.05])*(6.3/4) #numerical obstacle radii list (numerical box size is 1)
for i_R in range(len(L_R)):
    Radius=L_R[i_R]

    #we set up a try/except structure to avoid the program stopping at the first error
    #errors are likely because randomly placing obstacles can result in un-meshable geometries
    #errors are become more probable as obstacle fraction is increased
    success=0

    n_failure=0
    while success<5 and n_failure<20:
        ID=round(time.time())
        mesh_name=f"P21_3D_for_{result_type}_ID_{ID}"
        #if 1:
        try:
      
            Radius_impenetrable=0.635*Radius #If Radius_impenetrable>0, randomly placed fiber cannot completely intersect each other, its a very secondary parameter, better not touch
            functions.P20_3D_gmsh_generator(Radius,Radius_impenetrable,Length,Mesh_size,geom,N_obstacles,mesh_name,show_geometry=show_geometry)
            functions.OLI16_conversion_maillage_3D(mesh_name)
            a,_,_,_=functions.OLI16_solver_diff_3D(mesh_name)
            txt_to_print_tort=f"epsilon_obstacle_excluded={1-a[0]} ; tortuosity T⁻1 (x,y,z)={a[1]/a[0]};{a[2]/a[0]};{a[3]/a[0]}  " #'poro='+str(round(a[0],5)),', Perm_adim'+str(np.round(np.array([a[1]/a[0]]),5))+', ID='+str(ID)
            print(txt_to_print_tort)
            
            #the same mesh is reused for permeability: it is the same numerical geometry, BUT DOES NOT correspond to same physical configuration
            #as tortuosity is computed on accessible fluid fraction, while permeability is computed on true physical fluid fraction
            b=functions.OLI18_solver_perm_3D(mesh_name)
            R_num=Radius
            Ratio_num=R_factin/R_num  
            dimensional_perm=b[1]*(1e-18)*Ratio_num**2  #going from numerical non-dimensional permeability to physical dimensional peermeability
            
            txt_to_print_perm=f"epsilon_obstacle_occupied={1-b[0]} ; non-dimensional permeability(x)={b[1]}; dimensional permeability [m²] (x)={dimensional_perm}      "   
            print(txt_to_print_perm)
            success+=1
            
        except Exception as error:
            n_failure+=1
            print('n_failure='+str(n_failure)+' -> failure')
            print("An exception occurred:", error)
            
        except KeyboardInterrupt as error:
            break
    
#%%
'BLOCK TO RUN TORTUOSITY AND PERMEABILITY COMPUTATIONS FOR 3D RANDOM MIXED' 
result_type='tort+perm'
file_prename='P21_random_3D_' #intermediate files naming
file_name='P21_mesh_3D' #intermediate mesh file name

R_GFP_real=2.3 #physical parameters
R_act_real=4
R_rib_real=12.5

Length_num=0.95 #numerical parameters
R_act_num=0.06
R_rib_num=0.1875

Mesh_size=0.3 #default 0.3 ; adaptattive largest mesh size
geom=3 #geom 3 : mixed random

show_geometry=1 #shows the meshed geometry on a separate gmsh windows, but stops the program until the gmsh window is closed
 
L_N_act=[6]#[2...10,12...18]
for i_N_act in range(len(L_N_act)):
    N_act=L_N_act[i_N_act]
    N_rib=int(L_N_act[i_N_act]/2) #proposed mixture: 1 ribosome for 2 F-actin
    
    #we set up a try/except structure to avoid the program stopping at the first error
    #errors are likely because randomly placing obstacles can result in un-meshable geometries
    #errors are become more probable as obstacle fraction is increased

    success=0
    n_failure=0
    while success<5 and n_failure<20:
        ID=round(time.time())
        mesh_name=f"P21_3D_ID_{ID}"
        
        #if 1:
        try:
            Proportion_impenetrable=0.1 #If >0, randomly placed fiber cannot completely intersect each other, its a very secondary parameter, better not touch
            mesh_name_perm=mesh_name+'_perm'
            
            
            #The geometry is first generated and meshed for permeability computation (meshed geometry is the true physical fluid domain)
            MATRICE_OBSTACLES=functions.P20_3D_gmsh_generator([R_act_num,R_rib_num],Proportion_impenetrable,Length_num,Mesh_size,geom,[N_act,N_rib],mesh_name_perm,show_geometry=show_geometry)
            functions.OLI16_conversion_maillage_3D(mesh_name_perm)
            b=functions.OLI18_solver_perm_3D(mesh_name_perm)
            
            
            R_num=R_act_num
            Ratio_num=R_act_real/R_num  
            dimensional_perm=b[1]*(1e-18)*Ratio_num**2  #going from numerical non-dimensional permeability to physical dimensional peermeability
     
            txt_to_print_perm=f"epsilon_obstacle_occupied={1-b[0]} ; non-dimensional permeability(x)={b[1]}; dimensional permeability [m²](x)={dimensional_perm}      "   
            print('perm:'+txt_to_print_perm)
            
            #we reuse the same placement of obstacles to construct a second mesh where the meshed geometry is the accessible fluid domain for GFP mass center
            mesh_name_tort=mesh_name+'_tort'
            Radius_ratio_act=(R_GFP_real+R_act_real)/R_act_real
            Radius_ratio_rib=(R_GFP_real+R_rib_real)/R_rib_real
            functions.P20_3D_gmsh_generator([R_act_num*Radius_ratio_act,R_rib_num*Radius_ratio_rib],Proportion_impenetrable,Length_num,Mesh_size,geom,[N_act,N_rib],mesh_name_tort,MATRICE_OBSTACLES=MATRICE_OBSTACLES,show_geometry=show_geometry)            
            functions.OLI16_conversion_maillage_3D(mesh_name_tort)
            a,_,_,_=functions.OLI16_solver_diff_3D(mesh_name_tort)
                                
            txt_to_print_tort=f"epsilon_obstacle_excluded={1-a[0]} ; tortuosity T⁻1 (x,y,z)={a[1]/a[0]};{a[2]/a[0]};{a[3]/a[0]} " #'poro='+str(round(a[0],5)),', Perm_adim'+str(np.round(np.array([a[1]/a[0]]),5))+', ID='+str(ID)
            print('tort:'+txt_to_print_tort)
        
            success+=1
        except Exception as error:
            n_failure+=1
            print('n_failure='+str(n_failure)+' -> failure')
            print("An exception occurred:", error)
              


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    