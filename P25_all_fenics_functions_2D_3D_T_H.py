#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 20:00:09 2022

@author: destriano2
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fenics import *
import dolfin
from dolfin import *

import gmsh
import math
import os
import sys
import meshio

import time
import random


#%%
"========================================================"
"========================================================"
"================2D ====================================="
"========================================================"
"========================================================"

def P18_2D_gmsh_generator(Radius,Radius_inerte,Length,Mesh_size,geom,N_obstacles,file_name,MATRICE_OBSTACLES,refine_obstacle=1,show_geometry=0): #mesh generation
    #Radius : obstacle radius
    #Length : obstacle length (only for cylinder)
    #Mesh_size : mesh typical size
    #N_obstacles : number of obstacles
    #file_name : name for file saving
    #MATRICE_OBSTACLES : obstacles coordinates, only useful for random geometries when reusing a previously defined configuration
    
    L_t=[time.time()]   
    gmsh.initialize()  
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1) 
    eps = 1e-3 #tolerance
    gmsh.model.add("modele")  
    #refine_obstacle=1 #mesh refinement near obstacles
    
    "========================================================"
    "GEOMETRiCAL DEFINItiON"
    if geom==0: #disk in rectangle, not periodical
        return "non disponible"
        # gmsh.model.occ.addRectangle(0,0,0,2,1,100)       
        # gmsh.model.occ.synchronize() #geometry update    
        # gmsh.model.addPhysicalGroup(1, [1], 12) #low
        # gmsh.model.addPhysicalGroup(1, [2], 13) #right
        # gmsh.model.addPhysicalGroup(1, [3], 14) #top
        # gmsh.model.addPhysicalGroup(1, [4], 15) #left
        # L_volume=[100]
        
    elif geom==1: #5 disk in unit square, 2 groups, PERIODIC
        return "non disponible"
        # gmsh.model.occ.addRectangle(0,0,0,1,1,100)
        # gmsh.model.occ.addDisk(0.5,0.5,0,Radius_inerte,Radius_inerte,101)
        # gmsh.model.occ.addDisk(0.25,0.25,0,Radius,Radius,102)
        # gmsh.model.occ.addDisk(0.25,0.75,0,Radius,Radius,103)
        # gmsh.model.occ.addDisk(0.75,0.25,0,Radius,Radius,104)
        # gmsh.model.occ.addDisk(0.75,0.75,0,Radius,Radius,105)
        # gmsh.model.occ.cut([(2, 100)], [(2,101),(2,102),(2,103),(2,104),(2,105)]) 
        # gmsh.model.occ.synchronize()   
        # gmsh.model.addPhysicalGroup(1, [5], 3)          #central obstacle
        # gmsh.model.addPhysicalGroup(1, [6,7,8,9], 4)          #peripheral obstacles
        # L_volume=[100]
        
    elif geom==2: #2D centered disk
        gmsh.model.occ.addRectangle(0,0,0,1,1,100)
        gmsh.model.occ.addDisk(0.5,0.5,0,Radius,Radius,101)
        gmsh.model.occ.cut([(2, 100)], [(2,101)])         
        gmsh.model.occ.synchronize()  
        L_volume=[100]#volume name for physical group
        
    elif geom==21: #OBSOLETE 2D centered disk geometry with fluid contour
        gmsh.model.occ.addRectangle(0,0,0,1,1,100)
        gmsh.model.occ.addDisk(0.5,0.5,0,Radius,Radius,101)
        gmsh.model.occ.cut([(2, 100)], [(2,101)])   
        gmsh.model.occ.synchronize() 
        L_all=gmsh.model.getEntities(2) 
        
        gmsh.model.occ.addRectangle(0,0,0,1,1,202)
        gmsh.model.occ.addRectangle(0.001,0.001,0,0.998,0.998,203)
        gmsh.model.occ.cut([(2, 202)], [(2,203)]) #FLUID frame generation
        gmsh.model.occ.synchronize()  

        for i in range(len(L_all)):
            gmsh.model.occ.fuse([(2, 202)],[L_all[i]])    
        gmsh.model.occ.synchronize() 
        L_volume=[202] 
        
    elif geom==4 : #2D randomly position disks, periodic
        gmsh.model.occ.addRectangle(0,0,0,1,1,100)  
        k=0
        for i_fibre in range(N_obstacles):
            if MATRICE_OBSTACLES[-1,0]==0: #new obstacle position generation
                xa_1=random.random() 
                ya_1=random.random()
                MATRICE_OBSTACLES[i_fibre,:]=[xa_1,ya_1,Radius]   #saving obstacle position for ulterior use      
            else : #matrice obstacle is given: we reuse previously defined geometry
                [xa_1,ya_1]=MATRICE_OBSTACLES[i_fibre,:2] 
                       
            Radius_eq=np.sqrt(Radius**2+(Length/2)**2) #spheric collision radius for the cylinder
            for d_x in range(-1,2): #each obstacle must be added 27 times to ensure periodicity
                for d_y in range(-1,2):
                    for d_z in range(-1,2):     
                        #criterion: if obstacle sufficiently far from domain, useless to add it
                        Critere_cube=max([abs(xa_1+d_x-0.5),abs(ya_1+d_y-0.5)])
                        if not Critere_cube>0.5+Radius_eq+eps: #obstacle may interesect the fluid domain we must add it            
                            gmsh.model.occ.addDisk(xa_1+d_x,ya_1+d_y,0,Radius,Radius,1000+k)
                            k+=1 #number of obstacles added                           
        gmsh.model.occ.synchronize() # synchronization MODEL with OCC        
        L_volume=[100]
        All_volumes=gmsh.model.getEntitiesInBoundingBox(-2-eps,-2-eps,-2-eps,3+eps,3+eps,3+eps,2)
        Main_volume=[(2,100)] #volume du cube unité 
        Cut_volumes=list(set(All_volumes)-set(Main_volume)) #obstacles to remove
        gmsh.model.occ.cut(Main_volume,Cut_volumes) 
        gmsh.model.occ.synchronize()    
        #suppression of isolated internal fluid volumes
        Internal_volumes=gmsh.model.getEntitiesInBoundingBox(+eps,+eps,-eps,1-eps,1-eps,+eps,2) 
        gmsh.model.occ.remove(Internal_volumes) 
       
    "========================================================"
    "PHYSICAL GROUPS"
    L_bordcarre=Recherche_bords(eps) 
    L_pos_all=[x[1] for x in gmsh.model.getEntities(1)]
    L_obstacle=list(set(L_pos_all)-set(L_bordcarre)) 

    gmsh.model.addPhysicalGroup(2, L_volume, 1) #fluid volume domain
    gmsh.model.addPhysicalGroup(1, L_bordcarre,2)
    gmsh.model.addPhysicalGroup(1, L_obstacle,3)
    
    "========================================================"
    "PERIODIC BOUNDARIES"
    Set_periodic_boundaries(eps)

    "========================================================"
    "MESHING"
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), Mesh_size)
    if refine_obstacle: #mesh refinement near obstacles
        gmsh.model.mesh.field.add("Distance", 1)
        #gmsh.model.mesh.field.setNumbers(1, "PointsList", [5])
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", L_obstacle)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)        
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", Mesh_size / 5)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", Mesh_size)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(2, "DistMax", min(0.5-Radius,0.1))    
        gmsh.model.mesh.field.setAsBackgroundMesh(2)        
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)      
        gmsh.option.setNumber("Mesh.Algorithm", 5)  
    gmsh.model.mesh.generate(2)
    #shows the obtained mesh, pauses the program until popup window is closed
    if show_geometry:
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()
    
    "========================================================"
    "EXPORTING"   
    #gmsh.write(file_name+".brep") #
    gmsh.write(file_name+".msh")
    L_t.append(time.time())
    print('Temps Gmsh total : '+str(L_t[-1]-L_t[0])+'s')
    gmsh.finalize()
    return MATRICE_OBSTACLES

def OLI19_solver_perm_2D(file_name):   
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_2D(file_name)
    pbc = OLI16_PeriodicBC2()   #function for matching periodic boundaries on FEniCS
    P1,P2,V,n,V_alpha,e_vectors,u_tuple=OLI16_init_perm_fenics_2D(mesh,dx,pbc)
    noslip = Constant((0.0, 0.0)) #null speed at obstacle surface
    bc_obstacle=DirichletBC(V.sub(0), noslip, boundaries,3)  #W.sub(0) is the speed vector
    bcs = [bc_obstacle] #useful when several Dirichlet boundary conditions to implement (not the case here)  
    
    "========================================================"
    "FINITE ELEMENTS"
    
    (u, p) = TrialFunctions(V) #"""WWARNING FEniCS uses p_numerical=-p_real""" # Define variational problem
    (v, q) = TestFunctions(V)
    f = Constant((1, 0.0)) #source term
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
    L = inner(f, v)*dx
    # PRECONDITIONNING
    b = inner(grad(u), grad(v))*dx + p*q*dx    # Form for use in constructing preconditioner matrix
    A, bb = assemble_system(a, L, bcs) # Assemble system
    P, btmp = assemble_system(b, L, bcs) # Assemble preconditioner system
    U = Function(V)
    # Create Krylov solver and AMG preconditioner    
    
    solver=KrylovSolver("tfqmr", "amg") #seem to be best preconditionner, see:  
    #https://scicomp.stackexchange.com/questions/513/why-is-my-iterative-linear-solver-not-converging
    #solver = KrylovSolver("gmres")#KrylovSolver("tfqmr", "amg") solver itératif 
#    solver.parameters["relative_tolerance"] = 1.0e-8 / solver.parameters["absolute_tolerance"] = 1.0e-6 /solver.parameters["monitor_convergence"] = True/solver.parameters["maximum_iterations"] = 1000
    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)
    # L_t.append(time.time())
    # print('Temps FEniCS prepa : '+str(L_t[-1]-L_t[-2])+'s')
    
    "========================================================"
    "SOLVER"
    
    solver.solve(U.vector(), bb)    
    # L_t.append(time.time())
    # print('Temps FEniCS solve : '+str(L_t[-1]-L_t[-2])+'s')
    
    "========================================================"
    "COMPUTING GLOBAL VARIABLES"
    
    # Get sub-functions
    u, p = U.split() #separation speed vector and negative pressure      #p=-p 
    #First column computation of the permeability tensor
    V_tot=1
    #Epsi=V_alpha/V_tot #porosity   
    kperm_px=[(1/V_tot)*assemble(dot(u,e_vectors[0])*dx(mesh)),(1/V_tot)*assemble(dot(u,e_vectors[1])*dx(mesh))] 
    Kx=kperm_px[0] #we only consider kxx because kyx and kzx are expected to be very small
    #B_px=u*(Epsi/Kx)-e_vectors[0]
    #v_moy_x=1
    #v_moyint=[v_moy_x,0,0]
    #v_tilde=B_px*v_moy_x
    #v_tot=v_tilde+Constant((v_moyint[0],v_moyint[1],v_moyint[2]))     
    #C_drag=(2*Radius**2)/(9*(1-Epsi))*(1/Kx) #variable used for validation by Morgan Chabanon & al.
    #abscisse_SU=((1-Epsi)**(1/3))/Epsi  
    # L_t.append(time.time())
    # print('Temps FEniCS varglob : '+str(L_t[-1]-L_t[-2])+'s')
    L_t.append(time.time())
    print('Time FEniCS_perm total : '+str(L_t[-1]-L_t[0])+'s')  
    return (V_alpha, Kx)#,abscisse_SU,C_drag)#(abscisse_SU,C_drag,B_px,Kx,v_tot,v_tilde,v_moy_x) #Epsi,int_vtot_surf

def OLI16_init_perm_fenics_2D(mesh,dx,pbc):
    set_log_level(40)   
    # Define function spaces : we used mixed element spaces for direct resolution of the problem
    P2 = VectorElement("CG", mesh.ufl_cell(), 2) #order 2 element necessary for Stokes resolution 
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1) #order 1 element for pressure
    TH = MixedElement([P2, P1]) #Taylor Hood mixed element
    V = FunctionSpace(mesh, TH,constrained_domain=pbc) #periodc boundary condition applied here 

    n=FacetNormal(mesh) 
    V_alpha=assemble(Constant(1.0)*dx)               
    e_vectors=(Constant((1,0)),Constant((0,1))) 
    u_tuple=()
    
    return P1,P2,V,n,V_alpha,e_vectors,u_tuple


def OLI16_volume_2D(file_name):
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_2D(file_name)
    pbc = OLI16_PeriodicBC2()   #matching periodic boundaries on FEniCS
    P1,R,V,n,V_alpha,e_vectors,u_tuple=OLI16_initialisation_fenics_2D(mesh,dx,pbc)
    Resultat=V_alpha #porosity, then other results    
    L_t.append(time.time())
    #print('Temps FEniCS total : '+str(L_t[-1]-L_t[0])+'s')
    return Resultat
    
def OLI18_solver_diff_2D(file_name): #resolution using finite elements method
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_2D(file_name)
    pbc = OLI16_PeriodicBC2()   #periodic boundary matching 
    P1,R,V,n,V_alpha,e_vectors,u_tuple=OLI16_initialisation_fenics_2D(mesh,dx,pbc)
    Resultat=[V_alpha] #porosity, then other results  

    "========================================================"
    "FINITE ELEMENTS"
    
    for i_dir in range(2): #computation along the three directions to get Dxx Dyy and Dzz   
        e_x=e_vectors[i_dir]
 
        # Define variational problem
        (u, c) = TrialFunction(V) 
        (v, d) = TestFunction(V) 
        a = (dot(grad(u), grad(v))+ c*v + u*d)*dx #bilinear form
        L = -dot(e_x,n)*v*(ds(3)+ds(4))  #f*v*dx-g*dot(e_x,n)*v*ds #linear form, contains Neumann boundary condition on obstacles surfaces
    
        "========================================================"
        "SOLVER"
    
        w = Function(V) #solution definition
        solve(a == L, w,solver_parameters={'linear_solver': 'gmres'}) #USE OF GMRES
        (u, c) = w.split() #separating solution u and constant c
    
        "========================================================"
        "COMPUTING GLOBAL VARIABLES"

        dxx = assemble(dot(e_x, n)*u*(ds(3)+ds(4))) #integration of bx on this surface
        Dxx_ad=1+(1/V_alpha)*dxx #we get Dxx 
        epsi_Dad=V_alpha*Dxx_ad      
        Resultat.append(epsi_Dad) 
        u_tuple=u_tuple+(u,) 
        
    L_t.append(time.time())
    print('Temps FEniCS_diff total : '+str(L_t[-1]-L_t[0])+'s')
    return (Resultat,)+u_tuple



"========================================================"
"SECONDARY FUNCTIONS"
"========================================================"

def OLI16_conversion_maillage_2D(file_name):
    t_deb=time.time()
    mesh_transitoire=meshio.read(file_name+'.msh')
    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:,:2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
        return out_mesh
    
    line_mesh = create_mesh(mesh_transitoire, "line", prune_z=True)
    meshio.write(file_name+"_mf.xdmf", line_mesh)
    
    triangle_mesh = create_mesh(mesh_transitoire, "triangle", prune_z=True)
    meshio.write(file_name+"_mesh.xdmf", triangle_mesh) 
    os.system("rm "+file_name+".msh")
    #os.system("rm "+file_name+"_gmsh*") INUTILE
    t_fin=time.time()   
    #print('Temps Meshio total : '+str(t_fin-t_deb)+'s')    


def OLI16_importation_mesh_2D(file_name):
    mesh=Mesh() #définition du mesh
    
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()) #recovering triangle elements (surfaces)
    with XDMFFile(file_name+"_mesh.xdmf") as infile:
       infile.read(mesh)
       infile.read(mvc, "name_to_read")
    cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1) #recovery line elements 
    with XDMFFile(file_name+"_mf.xdmf") as infile:
        infile.read(mvc, "name_to_read")   
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc) 
    
    ds = Measure("ds", domain=mesh, subdomain_data=mf) #integration on lines
    dx = Measure("dx", domain=mesh, subdomain_data=cf) #integration on surfaces
    
    #mesh = Mesh("yourmeshfile.xml")
    # subdomains = MeshFunction("size_t", mesh, cf)
    # boundaries = MeshFunction("size_t", mesh, mf)
    #bcs = [DirichletBC(V, 5.0, boundaries, 1),# of course with your boundary
    #DirichletBC(V, 0.0, boundaries, 0)]
    
    # print("surf tot",assemble(Constant(1)*dx))
    # print("surf tot",assemble(Constant(1)*dx(1)))
    # print("courbe tot",assemble(Constant(1)*ds))
    # print("courbe 2+3",assemble(Constant(1)*(ds(3)+ds(4))))
    
    return mesh, ds, dx, mf, cf


def OLI16_initialisation_fenics_2D(mesh,dx,pbc):
    set_log_level(40)  
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #mixed finite elements
    R = FiniteElement("Real", mesh.ufl_cell(), 0)
    V = FunctionSpace(mesh, P1 * R,constrained_domain=pbc)
    n=FacetNormal(mesh) 
    V_alpha=assemble(Constant(1.0)*dx) #porosity computation             
    e_vectors=(Constant((1,0)),Constant((0,1))) 
    u_tuple=()
    
    return P1,R,V,n,V_alpha,e_vectors,u_tuple


class OLI16_PeriodicBC2(SubDomain): #function FEniCS for matching of periodic boundaries
    def inside(self, x, on_boundary):
    # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool((near(x[0], 0) or near(x[1],0)) and 
                    (not ((near(x[0], 1) and near(x[1], 0)) or 
                    (near(x[0], 0) and near(x[1], 1)))) and on_boundary)
    def map(self, x, y):        
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1
            y[1] = x[1] - 1
            ##### define mapping for edges in the box, such that mapping in 2 Cartesian coordinates are required
            #### right maps to left: left/right is defined as the x-direction
        elif near(x[0], 1):
            y[0] = x[0] - 1
            y[1] = x[1]
            ### back maps to front: front/back is defined as the y-direction    
        elif near(x[1], 1):
            y[0] = x[0]
            y[1] = x[1] - 1
        else: #Tres important meme si je ne saurai pas l'expliquer
            y[0] = -1000
            y[1] = -1000

def Recherche_bords(eps):
    L_bordcarre=[]
    Bounding_Box=gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps,  + eps, 1 + eps, + eps, 1)
    for i in range(len(Bounding_Box)):
        L_bordcarre.append(Bounding_Box[i][1])
    Bounding_Box=gmsh.model.getEntitiesInBoundingBox(- eps + 1, - eps,  - eps, 1 + eps + 1,1 + eps, + eps, 1)
    for i in range(len(Bounding_Box)):
        L_bordcarre.append(Bounding_Box[i][1])
    Bounding_Box=gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps, 1 + eps,  eps,  + eps, 1)
    for i in range(len(Bounding_Box)):
        L_bordcarre.append(Bounding_Box[i][1])
    Bounding_Box=gmsh.model.getEntitiesInBoundingBox( - eps ,  - eps+1, - eps, 1 + eps ,1 + eps+1,  + eps, 1)
    for i in range(len(Bounding_Box)):    
        L_bordcarre.append(Bounding_Box[i][1])
    return L_bordcarre

def Set_periodic_boundaries(eps):
    #gmsh can be used to force the mesh to be periodic, which is a requirement for FEniCS periodic boundary application
    #affine transform matrix
    # x axis
    for i_dir in range(2):
        if i_dir==0:
            d=[1,0]
        elif i_dir==1:
            d=[0,1]
        
        translation =[1, 0, 0, d[0], 0, 1, 0, d[1], 0, 0, 1, 0, 0, 0, 0, 1]
        sxmin = gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps, d[1] + eps, d[0] + eps, 1 + eps, 2)
        for i in sxmin:
            # Then we get the bounding box of each left surface
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i[0], i[1])
        # We translate the bounding box to the right and look for surfaces inside
        # it:
            #print(xmin)
            sxmax = gmsh.model.getEntitiesInBoundingBox(xmin - eps + d[0], ymin - eps+d[1],
                                                        zmin - eps, xmax + eps + d[0],
                                                        ymax + eps+d[1], zmax + eps, 2)
            # For all the matches, we compare the corresponding bounding boxes...
            for j in sxmax:
                xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                        j[0], j[1])
                xmin2 -= 1
                xmax2 -= 1
                # ...and if they match, we apply the periodicity constraint
                if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                        and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                        and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                    gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], translation)
        gmsh.model.occ.synchronize() #update




#%%
"========================================================"
"========================================================"
"================3D ====================================="
"========================================================"
"========================================================"

"========================================================"
"PRINCIPAL FUNCTIONS"
"========================================================"

def P20_3D_gmsh_generator(Radius,Radius_impenetrable,Length,Mesh_size,geom,N_obstacles,file_name,MATRICE_OBSTACLES=np.zeros((0)),show_geometry=0): #mesh generation
    #Radius : obstacle radius
    #Radius_impenetrable : radius obstacle to define its subdomain which cannot be intersected by other obstacles
    #Length : obstacle length (only for cylinder)
    #Mesh_size : mesh typical size
    #N_obstacles : number of obstacles
    #file_name : name for file saving
    #MATRICE_OBSTACLES : obstacles coordinates, only useful for random geometries when reusing a previously defined configuration

    L_t=[time.time()]   
    gmsh.initialize()  
    gmsh.option.setNumber('General.Verbosity', 3)
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1) 
    gmsh.option.setNumber("Geometry.Tolerance", 2e-8)
    
    
    eps = 1e-3 #tolérance
    gmsh.model.add("model")  
    refine_obstacle=1
    
    "========================================================"
    "GEOMETRY DEFINITION
    
    if geom==0: #Centered spheric obstacle periodic
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, 10) #domain cube "10"
        gmsh.model.occ.addSphere(0.5, 0.5, 0.5, Radius, 11, angle3=math.pi) #sphere "11" (obstacle)
        c = gmsh.model.occ.copy([(3, 11)])
        gmsh.model.occ.rotate(c, 0.5, 0.5, 0.5, 0, 0, 1, math.pi)
        gmsh.model.occ.cut([(3, 10)], [(3,11), c[0]]) #difference cube "10" - 2 half spheres               
        L_volume=[10] 
        if Radius<0.5:
            L_bordcarre=[1,2,3,4,5,6]
            L_obstacle=[7,8]      
        elif Radius>0.5:
            L_bordcarre=[1,2,3,4,5,8]
            L_obstacle=[6,7]
        if Radius==0.5:
            return "erreur R=0.5"      
        gmsh.model.occ.synchronize() #update
        
        
    elif geom==0.2: #3D cubic face centered periodic
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, 10) #domain cube "10"
        
        k_obstacle=0
        for i_x in range(2):
            for i_y in range(2):
                for i_z in range(2): 
                    gmsh.model.occ.addSphere(i_x, i_y, i_z, Radius, 11+k_obstacle) #sphere "11" (obstacle)
                    k_obstacle+=1
            
        for i_x in range(2):
            gmsh.model.occ.addSphere(i_x, 0.5, 0.5, Radius, 11+k_obstacle) #sphere "11" (obstacle)
            k_obstacle+=1
        for i_y in range(2):
            gmsh.model.occ.addSphere(0.5, i_y, 0.5, Radius, 11+k_obstacle) #sphere "11" (obstacle)
            k_obstacle+=1
        for i_z in range(2): 
            gmsh.model.occ.addSphere(0.5, 0.5, i_z, Radius, 11+k_obstacle) #sphere "11" (obstacle)
            k_obstacle+=1
            
        for i_k_obstacle in range(k_obstacle):
            gmsh.model.occ.cut([(3, 10)], [(3,11+i_k_obstacle)]) #difference cube "10" - 2 half spheres
        gmsh.model.occ.synchronize() #update     
        if Radius<np.sqrt(2)/4:
            L_volume=[10] 
            L_bordcarre=[1,2,5,6,9,13]
            L_obstacle=[3,4,7,8,10,11,12]+list(np.arange(14,22,1))
        else :
            L_volume=[10] 
            L_bordcarre=[1,2,7,9,10,11,14,15,16,17,19,22,24,27,29,31,32,33,34,35,36,37,38,39]
            L_obstacle=[3, 4, 5, 6, 8, 12, 13, 18, 20, 21, 23, 25, 26, 28, 30]
            #print('Sphere interseciton not implemented')
            
    #=============================SPHERES PARTIAL INTERSECTIONS====================================
    elif geom==1: #spheres randomly placed with only partial intersection allowed
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, 10) #domain cube "10"
        
        if MATRICE_OBSTACLES.size==0: #= no matrice obstacle provided
            print('CREATION NOUVELLE GEOMETRIE')
            iteration_max=1000 
            k=0 #number of placed obstacles (*27) 
            L_x,L_y,L_z=[],[],[] #placed obstacles coordinates    
            for i_fibre in range(N_obstacles):
                flag_validationencours=1 #new obstacle placement not validated yet
                iteration=0             
                while flag_validationencours and iteration<iteration_max : #looking for valid positions
                    [xa_1,ya_1,za_1]=[random.random(),random.random(),random.random()]
                    flag_colision=0 #pas encore de colision entre obstacles
                    i=0 #verification of the criterion               
                    while i<len(L_x) and flag_colision==0: 
                        d=np.sqrt((xa_1-L_x[i])**2+(ya_1-L_y[i])**2+(za_1-L_z[i])**2) 
                        if d<2*Radius_impenetrable:
                            flag_colision=1 #colision!
                        i+=1                 
                    if i==len(L_x) and flag_colision==0:
                        flag_validationencours=0 #placement is validated
                    iteration+=1
                    if iteration==iteration_max: #impossible to place a new obstacle
                        print("FAILURE OBSTACLE POSITIONING N°"+str(i_fibre))
                        return 
                 
                #COPYING 27 TIMES THE OBSTACLE FOR PERIODICITY 
                for d_x in range(-1,2):
                    for d_y in range(-1,2):
                        for d_z in range(-1,2):     
                            L_cube=1 
                            [x_cube,y_cube,z_cube]=[0.5,0.5,0.5] 
                            Critere_cube=abs(xa_1+d_x-x_cube)<(L_cube/2+Radius) and abs(ya_1+d_y-y_cube)<(L_cube/2+Radius) and abs(za_1+d_z-z_cube)<(L_cube/2+Radius)
                            if Critere_cube:
                                gmsh.model.occ.addSphere(xa_1+d_x, ya_1+d_y, za_1+d_z, Radius, 1000+k)
                                L_x.append(xa_1+d_x) #coordinates saving
                                L_y.append(ya_1+d_y)
                                L_z.append(za_1+d_z)
                                k+=1
            MATRICE_OBSTACLES=np.zeros((len(L_x),3)) 
            MATRICE_OBSTACLES[:,:]=np.transpose([L_x,L_y,L_z]) 
        
        else : #IF A MATRICE OBSTACLES IS PROVIDED, WE REUSE IT
            N_obstacles_repetes=MATRICE_OBSTACLES.shape[0] 
            for k in range(N_obstacles_repetes):
                [x,y,z]=MATRICE_OBSTACLES[k,:] 
                gmsh.model.occ.addSphere(x, y, z, Radius, 1000+k)                 
        gmsh.model.occ.synchronize() #update      
    
    #=============================CYLINDERS WITH PARTIAL INTERSECTION====================================
    elif geom==2: #
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, 10) #domain cube "10"
        
        if MATRICE_OBSTACLES.size==0: #no MATRICE_OBSTACLES provided
            print('NEW GEOMETRY CREATION')
            iteration_max=10000
            k=0
            L_xa,L_ya,L_za,L_xb,L_yb,L_zb=[],[],[],[],[],[]            
            Radius_eq=np.sqrt(Radius**2+(Length/2)**2) 
            for i_fibre in range(N_obstacles):  
                flag_validationencours=1
                iteration=0              
                while flag_validationencours and iteration<iteration_max :
                    #POINT : CENTRE BASE DU CYLINDRE
                    [xa,ya,za]=[random.random(),random.random(),random.random()]  #first point definition: center of base of the cylinder
                    #vector: cylinder central axis
                    [xb,yb,zb]=[random.uniform(-1,2),random.uniform(-1,2),random.uniform(-1,2)]
                    #print('xb,yb,zb='+str([xb,yb,zb]))
                    n_l2=np.sqrt(xb**2+yb**2+zb**2) #norm of the random vector
                    [xb,yb,zb]=(Length/n_l2)*np.array([xb,yb,zb]) #normalization of the size of the random vector
                                
                    flag_colision=0
                    i=0
                    while i<len(L_xa) and flag_colision==0:
                        p1=np.array([xa,ya,za]) 
                        p2=p1+np.array([xb,yb,zb]) 
                        p3=np.array([L_xa[i],L_ya[i],L_za[i]]) 
                        p4=p3+np.array([L_xb[i],L_yb[i],L_zb[i]])   
                        
                        d=closest_line_seg_line_seg(p1, p2, p3, p4) #computation of the distance between the two cylinders to see if intersection criterion is respected

                        if d<2*Radius_impenetrable:
                            flag_colision=1
                            if iteration==1000:
                                print('echec*1000')
                        i+=1                       
                    if i==len(L_xa) and flag_colision==0:
                        flag_validationencours=0
                        #print("success obstacle N°"+str(i_fibre))
                    iteration+=1
                    if iteration==iteration_max:
                        print("FAILURE POSITIONNING OBSTACLE N°")  
                        return
                
                #adding the validated obstacle
                [xc,yc,zc]=np.array([xa,ya,za])+np.array([xb,yb,zb])/2           
                for d_x in range(-1,2): 
                    for d_y in range(-1,2):
                        for d_z in range(-1,2):     
                            L_cube=1
                            [x_cube,y_cube,z_cube]=[0.5,0.5,0.5]
                            "Radius_eq*=2 is used to increase program robustness "
                            Radius_eq*=2
                            Critere_cube=abs(xc+d_x-x_cube)<(L_cube/2+Radius_eq) and abs(yc+d_y-y_cube)<(L_cube/2+Radius_eq) and abs(zc+d_z-z_cube)<(L_cube/2+Radius_eq)
                            if Critere_cube :            
                                gmsh.model.occ.addCylinder(xa+d_x, ya+d_y, za+d_z, xb, yb, zb, Radius, 1000+k)
                                L_xa.append(xa+d_x)
                                L_ya.append(ya+d_y)
                                L_za.append(za+d_z)
                                L_xb.append(xb)
                                L_yb.append(yb)
                                L_zb.append(zb)                              
                                k+=1 #number of added obstacles
            MATRICE_OBSTACLES=np.zeros((k,6))
            MATRICE_OBSTACLES[:,:]=np.transpose([L_xa,L_ya,L_za,L_xb,L_yb,L_zb])
               
        else : #MATRICE_OBSTACLES was provided
            print('reusing existing geometry')
            N_obstacles_repetes=MATRICE_OBSTACLES.shape[0]
            for k in range (N_obstacles_repetes):
                [xa,ya,za,xb,yb,zb]=MATRICE_OBSTACLES[k,:]
                gmsh.model.occ.addCylinder(xa, ya, za, xb, yb, zb, Radius, 1000+k)                                   
        gmsh.model.occ.synchronize() #update
        
    #============================= CYLINDERS + SPHERES : Factin + Ribosomes ====================================
    elif geom==3: #
        N_act=N_obstacles[0] ; N_rib=N_obstacles[1]
        Radius_act=Radius[0] ; Radius_rib=Radius[1]

        Radius_impenetrable_rib=Radius_impenetrable*Radius_rib
        Radius_impenetrable_act=Radius_impenetrable*Radius_act
    
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, 10) #domaine cube "10"
        
        if len(MATRICE_OBSTACLES)==0: 
            
            '====Placing the spheres===='
            print('new geom creation')
            iteration_max=1000 
            k=0 #nombre d'obstacles placés (*27) 
            L_x,L_y,L_z=[],[],[]   
            for i_fibre in range(N_rib):
                flag_validationencours=1 
                iteration=0             
                while flag_validationencours and iteration<iteration_max : 
                    [xa_1,ya_1,za_1]=[random.random(),random.random(),random.random()] 
                    flag_colision=0 
                    i=0                
                    while i<len(L_x) and flag_colision==0: 
                        d=np.sqrt((xa_1-L_x[i])**2+(ya_1-L_y[i])**2+(za_1-L_z[i])**2) 
                        if d<2*Radius_impenetrable_rib:
                            flag_colision=1 #on a colision
                        i+=1                 
                    if i==len(L_x) and flag_colision==0:
                        flag_validationencours=0 
                    iteration+=1
                    if iteration==iteration_max: 
                        print("FAILURE POSITIONNING OBSTACLE N°"+str(i_fibre))
                        return
                 
                #POSITIONNING OF THE 27* OBSTACLES
                for d_x in range(-1,2):
                    for d_y in range(-1,2):
                        for d_z in range(-1,2):     
                            L_cube=1 
                            [x_cube,y_cube,z_cube]=[0.5,0.5,0.5] 
                            Radius_eq=Radius_rib
                            Radius_eq*=2
                            "==============="
                            Critere_cube=abs(xa_1+d_x-x_cube)<(L_cube/2+Radius_eq) and abs(ya_1+d_y-y_cube)<(L_cube/2+Radius_eq) and abs(za_1+d_z-z_cube)<(L_cube/2+Radius_eq)
                            if Critere_cube:
                                gmsh.model.occ.addSphere(xa_1+d_x, ya_1+d_y, za_1+d_z, Radius_rib, 1000+k)
                                L_x.append(xa_1+d_x) 
                                L_y.append(ya_1+d_y)
                                L_z.append(za_1+d_z)
                                k+=1   
                MATRICE_OBSTACLES_spheres=np.zeros((len(L_x),3)) 
                MATRICE_OBSTACLES_spheres[:,:]=np.transpose([L_x,L_y,L_z]) 
        
            '====CYLINDERS PLACEMENT===='
            iteration_max=10000
            #k=0
            L_xa,L_ya,L_za,L_xb,L_yb,L_zb=[],[],[],[],[],[]       
            Radius_eq=np.sqrt(Radius_act**2+(Length/2)**2) #cylinder potential collision sphere
            for i_fibre in range(N_act):  
                flag_validationencours=1
                iteration=0              
                while flag_validationencours and iteration<iteration_max :
                    [xa,ya,za]=[random.random(),random.random(),random.random()]  
                    [xb,yb,zb]=[random.uniform(-1,2),random.uniform(-1,2),random.uniform(-1,2)]
                    #print('xb,yb,zb='+str([xb,yb,zb]))
                    n_l2=np.sqrt(xb**2+yb**2+zb**2) 
                    [xb,yb,zb]=(Length/n_l2)*np.array([xb,yb,zb]) 
                                
                    flag_colision_act_act=0 ; flag_colision_act_rib=0
                    i=0
                    while i<len(L_xa) and flag_colision_act_act==0:
                        p1=np.array([xa,ya,za]) 
                        p2=p1+np.array([xb,yb,zb]) 
                        p3=np.array([L_xa[i],L_ya[i],L_za[i]]) 
                        p4=p3+np.array([L_xb[i],L_yb[i],L_zb[i]])   
                        
                        d=closest_line_seg_line_seg(p1, p2, p3, p4) #minimal distance between the two cylinders

                        if d<2*Radius_impenetrable_act:
                            flag_colision=1
                            if iteration==1000:
                                print('echec*1000')
                        i+=1       
                    j=0  
                    while j<len(L_x) and flag_colision_act_rib==0: #computing collisions between cylinder and spheres
                        p1=np.array([xa,ya,za]) 
                        p2=p1+np.array([xb,yb,zb]) 
                        p3=np.array([L_x[j]-eps,L_y[j]-eps,L_z[j]-eps]) 
                        p4=p3+np.array([L_x[j]+eps,L_y[j]+eps,L_z[j]+eps])   
                            
                        d=closest_line_seg_line_seg(p1, p2, p3, p4) 

                        if d<(Radius_impenetrable_act+Radius_impenetrable_rib):
                            flag_colision=1
                            if iteration==1000:
                                print('echec*1000')
                        j+=1
                        
                    if i==len(L_xa) and j==len(L_x) and flag_colision_act_act==0 and flag_colision_act_rib==0 :
                        flag_validationencours=0
                        #print("success obstacle N°"+str(i_fibre))
                    else:
                        iteration+=1
                        
                    if iteration==iteration_max:
                        print("FAILURE PLACEMENT OBSTACLE N°")  
                        return
                
                #adding the validation obstacle
                [xc,yc,zc]=np.array([xa,ya,za])+np.array([xb,yb,zb])/2            
                for d_x in range(-1,2): 
                    for d_y in range(-1,2):
                        for d_z in range(-1,2):     
                            L_cube=1
                            [x_cube,y_cube,z_cube]=[0.5,0.5,0.5]
                            Radius_eq*=2
                            "==============="
                            Critere_cube=abs(xc+d_x-x_cube)<(L_cube/2+Radius_eq) and abs(yc+d_y-y_cube)<(L_cube/2+Radius_eq) and abs(zc+d_z-z_cube)<(L_cube/2+Radius_eq)
                            if Critere_cube : #la sphère de colision est susceptible d'intersecter le cube unité            
                                gmsh.model.occ.addCylinder(xa+d_x, ya+d_y, za+d_z, xb, yb, zb, Radius_act, 1000+k) #we add obstacle
                                L_xa.append(xa+d_x)
                                L_ya.append(ya+d_y)
                                L_za.append(za+d_z)
                                L_xb.append(xb)
                                L_yb.append(yb)
                                L_zb.append(zb)                              
                                k+=1 #compte le nombre d'obstacle effectivement ajoutés 
            MATRICE_OBSTACLES_cylindres=np.zeros((len(L_xa),6))
            MATRICE_OBSTACLES_cylindres[:,:]=np.transpose([L_xa,L_ya,L_za,L_xb,L_yb,L_zb])
            MATRICE_OBSTACLES=(MATRICE_OBSTACLES_spheres,MATRICE_OBSTACLES_cylindres)
               
        else : #MATRICE_OBSTACLES provided
            MATRICE_OBSTACLES_spheres,MATRICE_OBSTACLES_cylindres=MATRICE_OBSTACLES
            "Spheres"
            N_obstacles_repetes=MATRICE_OBSTACLES_spheres.shape[0]
            for k_1 in range (N_obstacles_repetes):
                [x,y,z]=MATRICE_OBSTACLES_spheres[k_1,:]
                gmsh.model.occ.addSphere(x,y,z, Radius_rib, 1000+k_1)#we add the sphere 
            "Cylindres"
            N_obstacles_repetes=MATRICE_OBSTACLES_cylindres.shape[0]
            for k_2 in range (N_obstacles_repetes):
                [xa,ya,za,xb,yb,zb]=MATRICE_OBSTACLES_cylindres[k_2,:]
                gmsh.model.occ.addCylinder(xa, ya, za, xb, yb, zb, Radius_act, 2000+k_2) #we add the cylinder                              
        
        
    gmsh.model.occ.synchronize() #maj geometrie  

        
        
        
    if geom in [1,2,3]:
        #cutting obstacles 
        All_volumes=gmsh.model.getEntitiesInBoundingBox(-2-eps,-2-eps,-2-eps,3+eps,3+eps,3+eps,3)
        Main_volume=[(3,10)] #fluid volume

        Cut_volumes=list(set(All_volumes)-set(Main_volume)) #obstacles to remove
        gmsh.model.occ.cut(Main_volume,Cut_volumes) 
        print('cut done')
        gmsh.model.occ.synchronize() #update

        #suppressing internal isolated fluid volumes
        Internal_volumes=gmsh.model.getEntitiesInBoundingBox(+eps,+eps,+eps,1-eps,1-eps,1-eps,3) 
        gmsh.model.occ.remove(Internal_volumes) 
        gmsh.model.occ.synchronize() #update
        
        print('nombre de volumes'+str(gmsh.model.getEntities(3)))
        
        
        L_volume=gmsh.model.getEntities(3)#gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps,  1+ eps, 1 + eps, 1 + eps, 3)
        L_volume=[x[1] for x in L_volume]
        print (L_volume)

        
        L_allsurfaces=gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps,  1+ eps, 1 + eps, 1 + eps, 2)
        L_allsurfaces=[x[1] for x in L_allsurfaces]

        L_pos_cube=[]
        selection=[x[1] for x in gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps,  + eps, 1 + eps, 1 + eps, 2)] 
        L_pos_cube+=selection
        selection=[x[1] for x in gmsh.model.getEntitiesInBoundingBox(- eps + 1, - eps,  - eps, 1 + eps + 1,1 + eps, 1 + eps, 2)]
        L_pos_cube+=selection        
        selection=[x[1] for x in gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps, 1 + eps,  eps, 1 + eps, 2)]
        L_pos_cube+=selection
        selection=[x[1] for x in gmsh.model.getEntitiesInBoundingBox( - eps ,  - eps+1, - eps, 1 + eps ,1 + eps+1, 1 + eps, 2)]
        L_pos_cube+=selection
        selection=[x[1] for x in gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps, 1 + eps, 1+ eps, eps, 2)]
        L_pos_cube+=selection
        selection=[x[1] for x in gmsh.model.getEntitiesInBoundingBox(0 - eps , 0 - eps,0 - eps+1, 1 + eps ,1 + eps, 1 + eps+1, 2)]
        L_pos_cube+=selection
        
        L_bordcarre=L_pos_cube
        L_obstacle=list(set(L_allsurfaces)-set(L_bordcarre))
        
    
    gmsh.model.occ.synchronize() #maj geometrie  
    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()
    
    "========================================================"
    "PHYSICAL GROUPS DEFINIITON"
    gmsh.model.addPhysicalGroup(2, L_bordcarre,2)
    gmsh.model.addPhysicalGroup(2, L_obstacle,3)
    gmsh.model.addPhysicalGroup(3, L_volume,1)
    
    "========================================================"
    "PERIODIC BOUNDARIES"
    Set_periodic_boundaries(eps)

    "========================================================"
    "MESHING"
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), Mesh_size)
    if refine_obstacle:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "SurfacesList", L_obstacle)
        #gmsh.model.mesh.field.setNumbers(1, "PointsList", [5])
        #SI R>0.5 : #gmsh.model.mesh.field.setNumbers(1, "CurvesList", [5,6,10,13,14,17,18,24,26])         
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)        
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", Mesh_size / 20)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", Mesh_size)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)#min(0.5-Radius,0.1))          
        gmsh.model.mesh.field.setAsBackgroundMesh(2)          
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)         
        gmsh.option.setNumber("Mesh.Algorithm", 5)  
    gmsh.model.mesh.generate(3)
    
    
    if show_geometry:
        #shows the mesh but stops the program
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()
    
    "========================================================"
    "EXPORTING"   
    #gmsh.write(file_name+".brep") #
    gmsh.write(file_name+".msh")
    L_t.append(time.time())
    print('Total GMsh time: '+str(L_t[-1]-L_t[0])+'s')
    gmsh.finalize()
    return MATRICE_OBSTACLES
    
    
def closest_line_seg_line_seg(p1, p2, p3, p4):
    "distance between two 3D segments : https://math.stackexchange.com/questions/846054/closest-points-on-two-line-segments" 

    P1 = p1 
    P2 = p3 #
    V1 = p2 - p1 
    V2 = p4 - p3 
    V21 = P2 - P1 

    v22 = np.dot(V2, V2) 
    v11 = np.dot(V1, V1) 
    v21 = np.dot(V2, V1) 
    v21_1 = np.dot(V21, V1) 
    v21_2 = np.dot(V21, V2)
    denom = v21 * v21 - v22 * v11 

    if np.isclose(denom, 0.): #if vectors are colinear
        s = 0.
        t = (v11 * s - v21_1) / v21
    else: #if vectors are not colinear
        s = (v21_2 * v21 - v22 * v21_1) / denom
        t = (-v21_1 * v21 + v11 * v21_2) / denom

    s = max(min(s, 1.), 0.) 
    t = max(min(t, 1.), 0.)

    p_a = P1 + s * V1 
    p_b = P2 + t * V2
    
    d=np.sqrt(np.dot(p_b-p_a,p_b-p_a)) #minimal distance between segments
    return d 
    

class OLI16_PeriodicBC3(SubDomain): #FEniCS function for matching of periodic boundaries
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool((near(x[0], 0) or near(x[1],0)  or near(x[2], 0)) and 
            (not ((near(x[0], 1) and near(x[2], 0)) or 
                  (near(x[0], 0) and near(x[2], 1)) or
                  (near(x[1], 1) and near(x[2], 0)) or 
                  (near(x[1], 0) and near(x[2], 1)) or
                  (near(x[0], 1) and near(x[1], 0)) or 
                  (near(x[0], 0) and near(x[1], 1)))) and on_boundary)
    def map(self, x, y):        
        if near(x[0], 1) and near(x[1], 1) and near(x[2],1):
            y[0] = x[0] - 1
            y[1] = x[1] - 1
            y[2] = x[2] - 1
        ##### define mapping for edges in the box, such that mapping in 2 Cartesian coordinates are required
        elif near(x[0], 1) and near(x[2], 1):
            y[0] = x[0] - 1
            y[1] = x[1] 
            y[2] = x[2] - 1      
        elif near(x[1], 1) and near(x[2], 1):
            y[0] = x[0] 
            y[1] = x[1] - 1
            y[2] = x[2] - 1
        elif near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1
            y[1] = x[1] - 1
            y[2] = x[2]         
        #### right maps to left: left/right is defined as the x-direction
        elif near(x[0], 1):
            y[0] = x[0] - 1
            y[1] = x[1]
            y[2] = x[2]
        ### back maps to front: front/back is defined as the y-direction    
        elif near(x[1], 1):
            y[0] = x[0]
            y[1] = x[1] - 1
            y[2] = x[2] 
        #### top maps to bottom: top/bottom is defined as the z-direction        
        elif near(x[2], 1):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - 1   
        else: #Tres important meme si je ne saurai pas l'expliquer
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000 
            
            
def Set_periodic_boundaries(eps):
    #gmsh function to force periodic meshing of boundaries 
    #affine transformations matrices
    #selon x
    translation =[1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]   
    #selon y    
    translation2=[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
    #selon z 
    translation3=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]
    sxmin = gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps,  + eps, 1 + eps, 1 + eps, 2)
    for i in sxmin:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i[0], i[1])
        # We translate the bounding box to the right and look for surfaces inside
        # it:
        sxmax = gmsh.model.getEntitiesInBoundingBox(xmin - eps + 1, ymin - eps,
                                                    zmin - eps, xmax + eps + 1,
                                                    ymax + eps, zmax + eps, 2)
        # For all the matches, we compare the corresponding bounding boxes...
        for j in sxmax:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                    j[0], j[1])
            xmin2 -= 1
            xmax2 -= 1
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                #print('x match found')
                gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], translation)
    #on y axis
    symin = gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps, 1 + eps,  eps, 1 + eps, 2)
    for i in symin:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i[0], i[1])
        # We translate the bounding box to the right and look for surfaces inside
        # it:
        symax = gmsh.model.getEntitiesInBoundingBox(xmin - eps , ymin - eps+1,
                                                    zmin - eps, xmax + eps ,
                                                    ymax + eps+1, zmax + eps, 2)
        # For all the matches, we compare the corresponding bounding boxes...
        for j in symax:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                    j[0], j[1])
            ymin2 -= 1
            ymax2 -= 1
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                #print('y match found')
                gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], translation2)             
    #on z axis
    szmin = gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps, 1 + eps, 1+ eps, eps, 2)
    for i in szmin:
        # Then we get the bounding box of each left surface
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(i[0], i[1])
        # We translate the bounding box to the right and look for surfaces inside
        # it:
        szmax = gmsh.model.getEntitiesInBoundingBox(xmin - eps , ymin - eps,
                                                    zmin - eps+1, xmax + eps ,
                                                    ymax + eps, zmax + eps+1, 2)
        # For all the matches, we compare the corresponding bounding boxes...
        for j in szmax:
            xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = gmsh.model.getBoundingBox(
                    j[0], j[1])
            zmin2 -= 1
            zmax2 -= 1
            # ...and if they match, we apply the periodicity constraint
            if (abs(xmin2 - xmin) < eps and abs(xmax2 - xmax) < eps
                    and abs(ymin2 - ymin) < eps and abs(ymax2 - ymax) < eps
                    and abs(zmin2 - zmin) < eps and abs(zmax2 - zmax) < eps):
                #print('z match found')
                gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], translation3)
    gmsh.model.occ.synchronize() #update





"========================================================"
"FUNCTIONS"
"========================================================"

def OLI16_solver_volume_3D(file_name):
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_3D(file_name)
    pbc = OLI16_PeriodicBC3()   #periodic boundaries matching on FEniCS
    P1,R,V,n,V_alpha,e_vectors,u_tuple=OLI16_init_diff_fenics_3D(mesh,dx,pbc)
    Resultat=V_alpha #porosity then other results    
    L_t.append(time.time())
    #print('Temps FEniCS total : '+str(L_t[-1]-L_t[0])+'s')
    return Resultat
    
def OLI16_solver_diff_3D(file_name): #finite elemnts resolution for diffusion
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_3D(file_name)
    pbc = OLI16_PeriodicBC3()   #periodic boundaries matching on FEniCS
    P1,R,V,n,V_alpha,e_vectors,u_tuple=OLI16_init_diff_fenics_3D(mesh,dx,pbc)
    Resultat=[V_alpha] #porosity



    "========================================================"
    "FINITE ELEMENTS"
    
    for i_dir in range(3): #computations on the 3 directions to get Dxx Dyy et Dzz   
        e_x=e_vectors[i_dir]
 
        # Define variational problem
        (u, c) = TrialFunction(V) #u is the solution searched, c is constant
        (v, d) = TestFunction(V) #fonctions test
        a = (dot(grad(u), grad(v))+ c*v + u*d)*dx #bilinear form
        L = -dot(e_x,n)*v*ds(3)  #f*v*dx-g*dot(e_x,n)*v*ds #linear form containing the neumann boundary condition on surface of obstacles 
    
        "========================================================"
        "SOLVER"
    
        w = Function(V) #solution definition 
        
        # prm = parameters.krylov_solver  # short form
        # prm.absolute_tolerance = 1E-10
        # prm.relative_tolerance = 1E-6
        # prm.maximum_iterations = 
        
        
        solve(a == L, w,solver_parameters={'linear_solver': 'gmres'}) # GMRES used

        
        (u, c) = w.split() #separating u and c
    
        "========================================================"
        "COMPUTING GLOBAL VARIABLES"

        dxx = assemble(dot(e_x, n)*u*ds(3)) 
        Dxx_ad=1+(1/V_alpha)*dxx 
        epsi_Dad=V_alpha*Dxx_ad       
        Resultat.append(epsi_Dad) 
        u_tuple=u_tuple+(u,) 
        
    L_t.append(time.time())
    print('Temps FEniCS_diff total : '+str(L_t[-1]-L_t[0])+'s')
    return (Resultat,)+u_tuple

def OLI18_solver_perm_3D(file_name):   
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_3D(file_name)
    pbc = OLI16_PeriodicBC3()   #periodic boundaries matching on FEniCS
    P1,P2,V,n,V_alpha,e_vectors,u_tuple=OLI16_init_perm_fenics_3D(mesh,dx,pbc)
    noslip = Constant((0.0, 0.0,0.0)) #null speed vector on obstacle surfaces 
    bc_obstacle=DirichletBC(V.sub(0), noslip, boundaries,3) 
    bcs = [bc_obstacle] #if several dirichlet boundaries conditions to be applied  
    
    "========================================================"
    "FINITE ELEMENTS "
    
    (u, p) = TrialFunctions(V) #"""WARNING FEniCS uses p_numerical=-p_real""" # Define variational problem
    (v, q) = TestFunctions(V)
    f = Constant((1, 0.0,0.0)) #source term
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
    L = inner(f, v)*dx
    # PRECONDITIONNING
    b = inner(grad(u), grad(v))*dx + p*q*dx    # Form for use in constructing preconditioner matrix
    A, bb = assemble_system(a, L, bcs) # Assemble system
    P, btmp = assemble_system(b, L, bcs) # Assemble preconditioner system
    U = Function(V)
    # Create Krylov solver and AMG preconditioner    
    #solver=KrylovSolver("tfqmr", "amg")
    solver=KrylovSolver("minres", "ilu")# #voir lien :  
    solver.parameters["monitor_convergence"] = True
    #https://scicomp.stackexchange.com/questions/513/why-is-my-iterative-linear-solver-not-converging
    #solver = KrylovSolver("gmres")#KrylovSolver("tfqmr", "amg") solver itératif 
    solver.parameters["relative_tolerance"] = 1.0e-8 #/ 
    solver.parameters["absolute_tolerance"] = 1.0e-6 #/solver.parameters["monitor_convergence"] = True/solver.parameters["maximum_iterations"] = 1000
    solver.parameters["maximum_iterations"] = 1000
    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

    
    "========================================================"
    "SOLVER"
    
    solver.solve(U.vector(), bb)    

    
    "========================================================"
    "GLOBAL VARIABLES COMPUTATION"
    
    # Get sub-functions
    u, p = U.split() #séparating speed vector and numerical pressure      #p=-p 
    #computation of the first column of the permeability tensor
    V_tot=1
    #Epsi=V_alpha/V_tot #porosity
    kperm_px=[(1/V_tot)*assemble(dot(u,e_vectors[0])*dx(mesh)),(1/V_tot)*assemble(dot(u,e_vectors[1])*dx(mesh)),(1/V_tot)*assemble(dot(u,e_vectors[2])*dx(mesh))] 
    Kx=kperm_px[0] #we only consider kxx as kyx and kzx are expected to be very small
    #B_px=u*(Epsi/Kx)-e_vectors[0]
    #v_moy_x=1
    #v_moyint=[v_moy_x,0,0]
    #v_tilde=B_px*v_moy_x
    #v_tot=v_tilde+Constant((v_moyint[0],v_moyint[1],v_moyint[2]))     
    #C_drag=(2*Radius**2)/(9*(1-Epsi))*(1/Kx) #variables used by Morgan Chabanon & al for validation
    #abscisse_SU=((1-Epsi)**(1/3))/Epsi  

    L_t.append(time.time())
    print('Time FEniCS_perm total : '+str(L_t[-1]-L_t[0])+'s')  
    return (V_alpha, Kx)#,abscisse_SU,C_drag)#(abscisse_SU,C_drag,B_px,Kx,v_tot,v_tilde,v_moy_x) #Epsi,int_vtot_surf



"========================================================"
"SECONDARY FUNCTIONS"
"========================================================"

def OLI16_conversion_maillage_3D(file_name):
    t_deb=time.time()
    mesh_transitoire=meshio.read(file_name+'.msh')
    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:,:2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
        return out_mesh
    
    line_mesh = create_mesh(mesh_transitoire, "triangle", prune_z=False)
    meshio.write(file_name+"_mf.xdmf", line_mesh)
    
    triangle_mesh = create_mesh(mesh_transitoire, "tetra", prune_z=False)
    meshio.write(file_name+"_mesh.xdmf", triangle_mesh) 
    #os.system("rm "+file_name+".msh")
    #os.system("rm "+file_name+"_gmsh*")
    t_fin=time.time()   
    #print('Temps Meshio total : '+str(t_fin-t_deb)+'s')    


def OLI16_importation_mesh_3D(file_name):
    mesh=Mesh() #définition du mesh
    
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()) #recovery of triangle elements (surfaces)
    with XDMFFile(file_name+"_mesh.xdmf") as infile:
       infile.read(mesh)
       infile.read(mvc, "name_to_read")
    cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1) #recovery of line elements
    with XDMFFile(file_name+"_mf.xdmf") as infile:
        infile.read(mvc, "name_to_read")   
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc) 
    
    ds = Measure("ds", domain=mesh, subdomain_data=mf) #integration on lines
    dx = Measure("dx", domain=mesh, subdomain_data=cf) #integration on surfaces  
    
    #mesh = Mesh("yourmeshfile.xml")
    # subdomains = MeshFunction("size_t", mesh, cf)
    # boundaries = MeshFunction("size_t", mesh, mf)
    #bcs = [DirichletBC(V, 5.0, boundaries, 1),# of course with your boundary
    #DirichletBC(V, 0.0, boundaries, 0)]
    
    # print("surf tot",assemble(Constant(1)*dx))
    # print("surf tot",assemble(Constant(1)*dx(1)))
    # print("courbe tot",assemble(Constant(1)*ds))
    # print("courbe 2+3",assemble(Constant(1)*(ds(3)+ds(4))))
    
    return mesh, ds, dx, mf, cf


def OLI16_init_diff_fenics_3D(mesh,dx,pbc):
    set_log_level(40)    
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #("Lagrange", mesh.ufl_cell(), 1) #use of mixed elements for resolution with pure Neumann boundary condition
    R = FiniteElement("Real", mesh.ufl_cell(), 0)
    V = FunctionSpace(mesh, P1 * R,constrained_domain=pbc)
    n=FacetNormal(mesh) 
    V_alpha=assemble(Constant(1.0)*dx) #porosity              
    e_vectors=(Constant((1,0,0)),Constant((0,1,0)),Constant((0,0,1))) 
    u_tuple=()
    
    return P1,R,V,n,V_alpha,e_vectors,u_tuple

def OLI16_init_perm_fenics_3D(mesh,dx,pbc):
    set_log_level(40)   
    # Define function spaces : use of mixed elements for direct resolution of the problem
    P2 = VectorElement("CG", mesh.ufl_cell(), 2) #order 2 elements required for Stokes resolution 
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1) #order 1 element for pressure
    TH = MixedElement([P2, P1]) #Taylor Hood mixed element
    V = FunctionSpace(mesh, TH,constrained_domain=pbc) #periodic boundary condition   

    n=FacetNormal(mesh) 
    V_alpha=assemble(Constant(1.0)*dx)              
    e_vectors=(Constant((1,0,0)),Constant((0,1,0)),Constant((0,0,1))) 
    u_tuple=()
    
    return P1,P2,V,n,V_alpha,e_vectors,u_tuple
