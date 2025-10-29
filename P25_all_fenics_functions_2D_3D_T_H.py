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

def P18_2D_gmsh_generator(Radius,Radius_inerte,Length,Mesh_size,geom,N_obstacles,file_name,MATRICE_OBSTACLES,refine_obstacle=1,show_geometry=0): #routine d'appel à gmsh pour génération du maillage
    #Radius : rayon de l'obstacle (sphère ou cylindre)
    #Length : longueur (prise en compte seulement pour cylindre)
    #Mesh_size : taille caract du maillage
    #N_obstacles : nombre d'obstacles
    #file_name : nom pour l'enregistrement des fichiers 
    #MATRICE_OBSTACLES : coordonnées des obstacles, utile uniquement en géométrie aléatoire pour reprendre une config existante
    
    L_t=[time.time()]   
    gmsh.initialize()  
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1) #pourquoi?
    eps = 1e-3 #tolérance
    gmsh.model.add("modele")  
    #refine_obstacle=1 #raffinement du maillage près des surfaces des obstacles
    
    "========================================================"
    "DEFINITION GEOMETRIE"
    if geom==0: #disque dans rectangle, NON PERIODIQUE
        return "non disponible"
        # gmsh.model.occ.addRectangle(0,0,0,2,1,100)       
        # gmsh.model.occ.synchronize() #maj geometrie     
        # gmsh.model.addPhysicalGroup(1, [1], 12) #bas
        # gmsh.model.addPhysicalGroup(1, [2], 13) #droite
        # gmsh.model.addPhysicalGroup(1, [3], 14) #haut
        # gmsh.model.addPhysicalGroup(1, [4], 15) #gauche
        # L_volume=[100]
        
    elif geom==1: #5 disques dans carré unité, 2 groupes, PERIODIQUE
        return "non disponible"
        # gmsh.model.occ.addRectangle(0,0,0,1,1,100)
        # gmsh.model.occ.addDisk(0.5,0.5,0,Radius_inerte,Radius_inerte,101)
        # gmsh.model.occ.addDisk(0.25,0.25,0,Radius,Radius,102)
        # gmsh.model.occ.addDisk(0.25,0.75,0,Radius,Radius,103)
        # gmsh.model.occ.addDisk(0.75,0.25,0,Radius,Radius,104)
        # gmsh.model.occ.addDisk(0.75,0.75,0,Radius,Radius,105)
        # gmsh.model.occ.cut([(2, 100)], [(2,101),(2,102),(2,103),(2,104),(2,105)]) 
        # gmsh.model.occ.synchronize() #maj geometrie    
        # gmsh.model.addPhysicalGroup(1, [5], 3)          #obstacle central
        # gmsh.model.addPhysicalGroup(1, [6,7,8,9], 4)          #obstacles périphériques
        # L_volume=[100]
        
    elif geom==2: #disque 2D centré
        gmsh.model.occ.addRectangle(0,0,0,1,1,100)
        gmsh.model.occ.addDisk(0.5,0.5,0,Radius,Radius,101)
        gmsh.model.occ.cut([(2, 100)], [(2,101)])         
        gmsh.model.occ.synchronize() #maj geometrie     
        L_volume=[100]#nom volume pour groupe physique
        
    elif geom==21: #geométrie disque2D centré avec cadre pour continuité du domaine
        gmsh.model.occ.addRectangle(0,0,0,1,1,100)
        gmsh.model.occ.addDisk(0.5,0.5,0,Radius,Radius,101)
        gmsh.model.occ.cut([(2, 100)], [(2,101)])   
        gmsh.model.occ.synchronize() #maj geometrie 
        L_all=gmsh.model.getEntities(2) #je récupère les morceaux du cut, qu'il y en ai un ou plusieurs
        
        gmsh.model.occ.addRectangle(0,0,0,1,1,202)
        gmsh.model.occ.addRectangle(0.001,0.001,0,0.998,0.998,203)
        gmsh.model.occ.cut([(2, 202)], [(2,203)]) #créaton du cadre fin autour du domaine
        gmsh.model.occ.synchronize() #maj geometrie 

        for i in range(len(L_all)):
            gmsh.model.occ.fuse([(2, 202)],[L_all[i]])   #fusion des morceaux de domaine et du cadre   
        gmsh.model.occ.synchronize() #maj geometrie 
        L_volume=[202] #nom volume pour groupe physique
        
    elif geom==4 : #disque 2D aléatoires avec superposition
        gmsh.model.occ.addRectangle(0,0,0,1,1,100)  
        k=0
        for i_fibre in range(N_obstacles):
            if MATRICE_OBSTACLES[-1,0]==0: #pas de matrice obstacle : on crée une nouvelle géométrie
                xa_1=random.random() #définition du premier point : centre de la base du cylindre
                ya_1=random.random()
                MATRICE_OBSTACLES[i_fibre,:]=[xa_1,ya_1,Radius]   #sauvegarde de la géométrie        
            else : #matrice obstacle en entrée : on récupère la géométrie définie
                [xa_1,ya_1]=MATRICE_OBSTACLES[i_fibre,:2] 
                       
            Radius_eq=np.sqrt(Radius**2+(Length/2)**2) #rayon de la sphère de colision associée au cylindre
            for d_x in range(-1,2): #selon chaque axe : on ajoute 27 fois l'obstacle
                for d_y in range(-1,2):
                    for d_z in range(-1,2):     
                        #definition critere : si la sphere de colision est trop loin du pt (0.5,0.5,0.5) alors ca ne sert à rien d'ajouter l'obstacle
                        Critere_cube=max([abs(xa_1+d_x-0.5),abs(ya_1+d_y-0.5)])
                        if not Critere_cube>0.5+Radius_eq+eps: #la sphère de colision est susceptible d'intersecter le cube unité            
                            gmsh.model.occ.addDisk(xa_1+d_x,ya_1+d_y,0,Radius,Radius,1000+k)
                            k+=1 #compte le nombre d'obstacle effectivement ajoutés                           
        gmsh.model.occ.synchronize() #maj geometrie pour synchro MODEL avec OCC        
        L_volume=[100]
        #cut des obstacles (dont la majorité devrait couper le cube, mais pas forcément tous)
        All_volumes=gmsh.model.getEntitiesInBoundingBox(-2-eps,-2-eps,-2-eps,3+eps,3+eps,3+eps,2)
        Main_volume=[(2,100)] #volume du cube unité 
        Cut_volumes=list(set(All_volumes)-set(Main_volume)) #liste des obstacles à retirer
        gmsh.model.occ.cut(Main_volume,Cut_volumes) #beaucoup plus efficace de retirer toute la liste d'un coup
        gmsh.model.occ.synchronize() #maj geometrie     
        #suppression des volumes interieurs isolés
        Internal_volumes=gmsh.model.getEntitiesInBoundingBox(+eps,+eps,-eps,1-eps,1-eps,+eps,2) #morceaux intérieurs au cube unité   
        gmsh.model.occ.remove(Internal_volumes) #suppression des volumes intérieurs du cube
       
    "========================================================"
    "GROUPES PHYSIQUES"
    L_bordcarre=Recherche_bords(eps) 
    L_pos_all=[x[1] for x in gmsh.model.getEntities(1)]
    L_obstacle=list(set(L_pos_all)-set(L_bordcarre)) 

    gmsh.model.addPhysicalGroup(2, L_volume, 1) #volume de résolution du domaine
    gmsh.model.addPhysicalGroup(1, L_bordcarre,2)
    gmsh.model.addPhysicalGroup(1, L_obstacle,3)
    
    "========================================================"
    "BORDS PERIODIQUES"
    Set_periodic_boundaries(eps)

    "========================================================"
    "MAILLAGE"
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), Mesh_size)
    if refine_obstacle: #rafinage du maillage près des obstacles
        gmsh.model.mesh.field.add("Distance", 1)
        #gmsh.model.mesh.field.setNumbers(1, "PointsList", [5])
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", L_obstacle)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)        
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", Mesh_size / 5)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", Mesh_size)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(2, "DistMax", min(0.5-Radius,0.1))   #pour éviter le bug qu'on on met distmax>distance au bord     
        gmsh.model.mesh.field.setAsBackgroundMesh(2)        
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)      
        gmsh.option.setNumber("Mesh.Algorithm", 5)  
    gmsh.model.mesh.generate(2)
    #Visualisation du maillage(/géométrie) obtenu (STOPPE LE PROGRAMME)
    if show_geometry:
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()
    
    "========================================================"
    "EXPORTATION"   
    #gmsh.write(file_name+".brep") #
    gmsh.write(file_name+".msh")
    L_t.append(time.time())
    print('Temps Gmsh total : '+str(L_t[-1]-L_t[0])+'s')
    gmsh.finalize()
    return MATRICE_OBSTACLES

def OLI19_solver_perm_2D(file_name):   
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_2D(file_name)
    pbc = OLI16_PeriodicBC2()   #fonction d'appairage des bords périodiques sur FEniCS
    P1,P2,V,n,V_alpha,e_vectors,u_tuple=OLI16_init_perm_fenics_2D(mesh,dx,pbc)
    noslip = Constant((0.0, 0.0)) #vitesse nulle aux parois
    bc_obstacle=DirichletBC(V.sub(0), noslip, boundaries,3) #"""BOUNDARY NUMBER SELON GEOMETRIE"""" #W.sub(0) c'est le vecteur vitesse 
    bcs = [bc_obstacle] #ligne utilise s'il y'a plusieurs conditions de Dirichlet à implémenter (ici non)   
    
    "========================================================"
    "ELEMENTS FINIS"
    
    (u, p) = TrialFunctions(V) #"""ATTENTION FEniCS utilise p_prog=-p_reelle""" # Define variational problem
    (v, q) = TestFunctions(V)
    f = Constant((1, 0.0)) #terme source pour le probleme colonne _x (pour _y ou _z il faut changer la composante non nulle)
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
    L = inner(f, v)*dx
    # PRECONDITIONNEMENT (pour améliorer conditionnement)
    b = inner(grad(u), grad(v))*dx + p*q*dx    # Form for use in constructing preconditioner matrix
    A, bb = assemble_system(a, L, bcs) # Assemble system
    P, btmp = assemble_system(b, L, bcs) # Assemble preconditioner system
    U = Function(V)
    # Create Krylov solver and AMG preconditioner    
    
    solver=KrylovSolver("tfqmr", "amg") #solver préféré car plus stable, probablement un meilleur préconditionner, voir lien :  
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
    "CALCUL VARIABLES GLOBALES"
    
    # Get sub-functions
    u, p = U.split() #séparation vecteur vitesse et pression NEGATIVE      #p=-p 
    #Je calcule toute la première colonne du tenseur de perméabilité
    #meme si je ne me sers que du coef kxx car 
    #le problème de stokes résolu permet d'avoir aussi kyx et kzx gratos
    V_tot=1
    #Epsi=V_alpha/V_tot #porosité    
    kperm_px=[(1/V_tot)*assemble(dot(u,e_vectors[0])*dx(mesh)),(1/V_tot)*assemble(dot(u,e_vectors[1])*dx(mesh))] 
    Kx=kperm_px[0] #jon considère seulement kxx car kyx et kzx sont à priori très petits
    #B_px=u*(Epsi/Kx)-e_vectors[0]
    #v_moy_x=1
    #v_moyint=[v_moy_x,0,0]
    #v_tilde=B_px*v_moy_x
    #v_tot=v_tilde+Constant((v_moyint[0],v_moyint[1],v_moyint[2]))     
    #C_drag=(2*Radius**2)/(9*(1-Epsi))*(1/Kx) #coefficient de trainée utilisé par Morgan pour cas test
    #abscisse_SU=((1-Epsi)**(1/3))/Epsi #abscisse utilisée par Morgan pour cas test  
    # L_t.append(time.time())
    # print('Temps FEniCS varglob : '+str(L_t[-1]-L_t[-2])+'s')
    L_t.append(time.time())
    print('Temps FEniCS_perm total : '+str(L_t[-1]-L_t[0])+'s')  
    return (V_alpha, Kx)#,abscisse_SU,C_drag)#(abscisse_SU,C_drag,B_px,Kx,v_tot,v_tilde,v_moy_x) #Epsi,int_vtot_surf

def OLI16_init_perm_fenics_2D(mesh,dx,pbc):
    set_log_level(40) #FENICS N'affiche que les erreurs en principe, en pratique il affiche trop    
    # Define function spaces : on utilise des éléments mixtes pour la résolution directe du problème
    P2 = VectorElement("CG", mesh.ufl_cell(), 2) #ordre 2 nécessaire pour résolution Stokes sinon instable 
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1) #élément pour la pression
    TH = MixedElement([P2, P1]) #élément mixte de Taylor Hood
    V = FunctionSpace(mesh, TH,constrained_domain=pbc) #condition périodique implémentée ici    

    n=FacetNormal(mesh) #structure un peu bizzare qui contient la normale ext aux parois du domaine
    V_alpha=assemble(Constant(1.0)*dx) #on intègre l'espace pour obtenir la porosité                
    e_vectors=(Constant((1,0)),Constant((0,1))) 
    u_tuple=()
    
    return P1,P2,V,n,V_alpha,e_vectors,u_tuple


def OLI20_solver_perm_2D_darcy_brinkmann(file_name,nano_poro,nano_perm_adimnano,l_b_s_a):   
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_2D(file_name)
    pbc = OLI16_PeriodicBC2()   #fonction d'appairage des bords périodiques sur FEniCS
    P1,P2,V,n,V_alpha,e_vectors,u_tuple=OLI16_init_perm_fenics_2D(mesh,dx,pbc)
    noslip = Constant((0.0, 0.0)) #vitesse nulle aux parois
    bc_obstacle=DirichletBC(V.sub(0), noslip, boundaries,3) #"""BOUNDARY NUMBER SELON GEOMETRIE"""" #W.sub(0) c'est le vecteur vitesse 
    bcs = [bc_obstacle] #ligne utilise s'il y'a plusieurs conditions de Dirichlet à implémenter (ici non)   
    
    "========================================================"
    "ELEMENTS FINIS"
    
    (u, p) = TrialFunctions(V) #"""ATTENTION FEniCS utilise p_prog=-p_reelle""" # Define variational problem
    (v, q) = TestFunctions(V) 
    
    f = Constant((1, 0.0)) #terme source pour le probleme colonne _x (pour _y ou _z il faut changer la composante non nulle)
    "f2 : terme de darcy dans la formulation faible (=a' dans mon cahier)"
    nano_perm_adimmicro=nano_perm_adimnano/l_b_s_a**2
    f2 = nano_poro/nano_perm_adimmicro
    
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx   +f2*inner(u,v)*dx #AJOUT DU TERME DE DARCY
    L = inner(f, v)*dx
    b = inner(grad(u), grad(v))*dx + p*q*dx+f2*inner(u,v)*dx #+f2*inner(u,v)*dx  #inner(grad(u), grad(v))*dx + p*q*dx  # Form for use in constructing preconditioner matrix
    # ratio=1/0.1
    # a = ratio*(inner(grad(u), grad(v)) + div(v)*p + q*div(u)   +f2*inner(u,v))*dx #AJOUT DU TERME DE DARCY
    # L = ratio*(inner(f, v))*dx
    
    # PRECONDITIONNEMENT (pour améliorer conditionnement)
    #b = ratio*(inner(grad(u), grad(v)) + p*q)*dx    # Form for use in constructing preconditioner matrix
    A, bb = assemble_system(a, L, bcs) # Assemble system
    P, btmp = assemble_system(b, L, bcs) # Assemble preconditioner system
    U = Function(V)
    # Create Krylov solver and AMG preconditioner    
    
    solver=KrylovSolver("tfqmr", "amg") #KrylovSolver("gmres")#KrylovSolver("tfqmr", "amg") 
    solver.parameters["maximum_iterations"] = 1000
    #solver.parameters["relative_tolerance"] = 1.0e-3
    #solver.parameters["monitor_convergence"] = True
    #solver préféré car plus stable, probablement un meilleur préconditionner, voir lien :  
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
    "CALCUL VARIABLES GLOBALES"
    
    # Get sub-functions
    u, p = U.split() #séparation vecteur vitesse et pression NEGATIVE      #p=-p 
    #Je calcule toute la première colonne du tenseur de perméabilité
    #meme si je ne me sers que du coef kxx car 
    #le problème de stokes résolu permet d'avoir aussi kyx et kzx gratos
    V_tot=1
    #Epsi=V_alpha/V_tot #porosité    
    print('nano poro='+str(nano_poro))
    print('f2='+str(f2))
    
    kperm_px=[assemble(dot(u,e_vectors[0])*dx(mesh))*(1/V_tot)*nano_poro]#[(1/V_tot)*assemble(dot(u,e_vectors[0])*dx(mesh)),(1/V_tot)*assemble(dot(u,e_vectors[1])*dx(mesh))] 
    Kx=kperm_px[0] #jon considère seulement kxx car kyx et kzx sont à priori très petits
    
    pt1=(V_alpha*nano_poro/Kx)
    pt2=(nano_poro/nano_perm_adimmicro)
    K_star=V_alpha*(pt1-pt2)**(-1)
    
    #B_px=u*(Epsi/Kx)-e_vectors[0]
    #v_moy_x=1
    #v_moyint=[v_moy_x,0,0]
    #v_tilde=B_px*v_moy_x
    #v_tot=v_tilde+Constant((v_moyint[0],v_moyint[1],v_moyint[2]))     
    #C_drag=(2*Radius**2)/(9*(1-Epsi))*(1/Kx) #coefficient de trainée utilisé par Morgan pour cas test
    #abscisse_SU=((1-Epsi)**(1/3))/Epsi #abscisse utilisée par Morgan pour cas test  
    # L_t.append(time.time())
    # print('Temps FEniCS varglob : '+str(L_t[-1]-L_t[-2])+'s')
    L_t.append(time.time())
    print('Temps FEniCS_perm total : '+str(L_t[-1]-L_t[0])+'s')  
    return (V_alpha, Kx,K_star,nano_perm_adimmicro) #ICI Kx représente K effectif, K_star représente la contribution de l'échelle considérée



def OLI16_volume_2D(file_name):
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_2D(file_name)
    pbc = OLI16_PeriodicBC2()   #fonction d'appairage des bords périodiques sur FEniCS
    P1,R,V,n,V_alpha,e_vectors,u_tuple=OLI16_initialisation_fenics_2D(mesh,dx,pbc)
    Resultat=V_alpha #porosité, puis autres résultats      
    L_t.append(time.time())
    #print('Temps FEniCS total : '+str(L_t[-1]-L_t[0])+'s')
    return Resultat
    
def OLI18_solver_diff_2D(file_name): #routine d'appel à FEniCS pour résolution du problème FEM
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_2D(file_name)
    pbc = OLI16_PeriodicBC2()   #fonction d'appairage des bords périodiques sur FEniCS
    P1,R,V,n,V_alpha,e_vectors,u_tuple=OLI16_initialisation_fenics_2D(mesh,dx,pbc)
    Resultat=[V_alpha] #porosité, puis autres résultats

    "========================================================"
    "ELEMENTS FINIS"
    
    for i_dir in range(2): #calcul selon les 3 directions pour avoir Dxx Dyy et Dzz   
        e_x=e_vectors[i_dir]
 
        # Define variational problem
        (u, c) = TrialFunction(V) #solution cherchées : u est la sol, c la constante
        (v, d) = TestFunction(V) #fonctions test
        a = (dot(grad(u), grad(v))+ c*v + u*d)*dx #forme bilinéaire
        L = -dot(e_x,n)*v*(ds(3)+ds(4))  #f*v*dx-g*dot(e_x,n)*v*ds #forme linéaire. Contient la condition de neumann aux abords des obstacles
    
        "========================================================"
        "SOLVER"
    
        w = Function(V) #on définit une nouvelle fonction qui sera la solution
        solve(a == L, w,solver_parameters={'linear_solver': 'gmres'}) #UTILISATION GMRES OPTIMALE, precondtionner non réglé
        (u, c) = w.split() #séparation u et constante c
    
        "========================================================"
        "CALCUL VARIABLES GLOBALES"

        dxx = assemble(dot(e_x, n)*u*(ds(3)+ds(4))) #on intègre bx sur cette surface
        Dxx_ad=1+(1/V_alpha)*dxx #on obtient la compo Dxx du tenseur
        epsi_Dad=V_alpha*Dxx_ad #on obtient la grandeur qu'on plot habituellement        
        Resultat.append(epsi_Dad) #ajout du résultat 
        u_tuple=u_tuple+(u,) #sauvegarde du champ solution
        
    L_t.append(time.time())
    print('Temps FEniCS_diff total : '+str(L_t[-1]-L_t[0])+'s')
    return (Resultat,)+u_tuple

def OLI16_solver_probA2_2D(A,k2,Da,file_name): #routine d'appel à FEniCS pour résolution du problème FEM
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_2D(file_name)
    pbc = OLI16_PeriodicBC2()   #fonction d'appairage des bords périodiques sur FEniCS
    P1,R,V,n,V_alpha,e_vectors,u_tuple=OLI16_initialisation_fenics_2D(mesh,dx,pbc)
    Resultat=[V_alpha] #porosité, puis autres résultats

    "========================================================"
    "ELEMENTS FINIS"
    
    for i_dir in range(1): #calcul selon les 3 directions pour avoir Dxx Dyy et Dzz    
        # Define variational problem
        (u, c) = TrialFunction(V) #solution cherchées : u est la sol, c la constante
        (v, d) = TestFunction(V) #fonctions test
        a = (dot(grad(u), grad(v))+ c*v + u*d)*dx #forme bilinéaire
        L = -(k2/Da)*A*v*dx-(k2/Da)*v*ds(4) #forme linéaire. Contient la condition de neumann aux abords des obstacles
      
        "========================================================"
        "SOLVER"
    
        w = Function(V) #on définit une nouvelle fonction qui sera la solution
        solve(a == L, w,solver_parameters={'linear_solver': 'gmres'}) #UTILISATION GMRES OPTIMALE, precondtionner non réglé
        (u, c) = w.split() #séparation u et constante c
    
        "========================================================"
        "CALCUL VARIABLES GLOBALES"
        
    for i_dir in range(2): #calcul selon les 3 directions pour avoir Dxx Dyy et Dzz   
        e_x=e_vectors[i_dir]
        u_ab_x=assemble(dot(e_x,n)*u*(ds(3)+ds(4)))
        Resultat.append(u_ab_x)     
    u_tuple=u_tuple+(u,) #sauvegarde du champ solution
        
    L_t.append(time.time())
    #print('Temps FEniCS total : '+str(L_t[-1]-L_t[0])+'s')
    return (Resultat,)+u_tuple


def OLI16_solver_probB2_2D(k3,Db,u_A1_x,u_A1_y,file_name): #routine d'appel à FEniCS pour résolution du problème FEM
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_2D(file_name)
    pbc = OLI16_PeriodicBC2()   #fonction d'appairage des bords périodiques sur FEniCS
    P1,R,V,n,V_alpha,e_vectors,u_tuple=OLI16_initialisation_fenics_2D(mesh,dx,pbc)
    Resultat=[V_alpha] #porosité, puis autres résultats
    P1_sp=FunctionSpace(mesh, P1,constrained_domain=pbc)
    u_A1=(u_A1_x,u_A1_y)

    "========================================================"
    "ELEMENTS FINIS"
    
    for i_dir in range(2): #calcul selon les 3 directions pour avoir Dxx Dyy et Dzz   
        e_x=e_vectors[i_dir]
        w=u_A1[i_dir]
        W=Function(P1_sp) #W est une projection de w sur l'espace des fonctions V 
        W=project(w,P1_sp)
 
        # Define variational problem
        (u, c) = TrialFunction(V) #solution cherchées : u est la sol, c la constante
        (v, d) = TestFunction(V) #fonctions test
        a = (dot(grad(u), grad(v))+ c*v + u*d)*dx #forme bilinéaire
        L = (k3/Db)*W*v*dx #forme linéaire. Contient la condition de neumann aux abords des obstacles    
        "========================================================"
        "SOLVER"
    
        w = Function(V) #on définit une nouvelle fonction qui sera la solution
        solve(a == L, w,solver_parameters={'linear_solver': 'gmres'}) #UTILISATION GMRES OPTIMALE, precondtionner non réglé
        (u, c) = w.split() #séparation u et constante c
    
        "========================================================"
        "CALCUL VARIABLES GLOBALES"
   
        Dad = (1/V_alpha)*assemble(dot(e_x, n)*u*(ds(3)+ds(4))) #on intègre bx sur cette surface
        epsi_Dad=V_alpha*Dad #on obtient la grandeur qu'on plot habituellement
        
        Resultat.append(epsi_Dad) #ajout du résultat
        u_tuple=u_tuple+(u,) #sauvegarde du champ solution
        
    L_t.append(time.time())
    #print('Temps FEniCS total : '+str(L_t[-1]-L_t[0])+'s')
    return (Resultat,)+u_tuple


def OLI16_solver_probB3_2D(k3,Db,u_A2,file_name): #routine d'appel à FEniCS pour résolution du problème FEM   
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_2D(file_name)
    pbc = OLI16_PeriodicBC2()   #fonction d'appairage des bords périodiques sur FEniCS
    P1,R,V,n,V_alpha,e_vectors,u_tuple=OLI16_initialisation_fenics_2D(mesh,dx,pbc)
    P1_sp=FunctionSpace(mesh, P1,constrained_domain=pbc)
    Resultat=[V_alpha] #porosité, puis autres résultats

    "========================================================"
    "ELEMENTS FINIS"
    
    for i_dir in range(1): #calcul selon les 3 directions pour avoir Dxx Dyy et Dzz   
        w=u_A2
        W=Function(P1_sp) #W est une projection de w sur l'espace des fonctions V 
        W=project(w,P1_sp)
 
        # Define variational problem
        (u, c) = TrialFunction(V) #solution cherchées : u est la sol, c la constante
        (v, d) = TestFunction(V) #fonctions test
        a = (dot(grad(u), grad(v))+ c*v + u*d)*dx #forme bilinéaire
        L = (k3/Db)*W*v*dx #forme linéaire. Contient la condition de neumann aux abords des obstacles
    
        "========================================================"
        "SOLVER"
    
        w = Function(V) #on définit une nouvelle fonction qui sera la solution
        solve(a == L, w,solver_parameters={'linear_solver': 'gmres'}) #UTILISATION GMRES OPTIMALE, precondtionner non réglé
        (u, c) = w.split() #séparation u et constante c
    
        "========================================================"
        "CALCUL VARIABLES GLOBALES"

    for i_dir in range(2): #calcul selon les 3 directions pour avoir Dxx Dyy et Dzz   
        e_x=e_vectors[i_dir]
        u_ba_x=assemble(dot(e_x,n)*u*(ds(3)+ds(4)))        
        Resultat.append(u_ba_x) #ajout du résultat 
    u_tuple=u_tuple+(u,) #sauvegarde du champ solution
        
    L_t.append(time.time())
    #print('Temps FEniCS total : '+str(L_t[-1]-L_t[0])+'s')
    return (Resultat,)+u_tuple


"========================================================"
"FONCTIONS SECONDAIRES (NE PAS MODIFIER EN PRINCIPE)"
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
    
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()) #récupération éléments triangles (surfaces)
    with XDMFFile(file_name+"_mesh.xdmf") as infile:
       infile.read(mesh)
       infile.read(mvc, "name_to_read")
    cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1) #récupération éléments segments (bordures)
    with XDMFFile(file_name+"_mf.xdmf") as infile:
        infile.read(mvc, "name_to_read")   
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc) 
    
    ds = Measure("ds", domain=mesh, subdomain_data=mf) #définition intégrande bordure
    dx = Measure("dx", domain=mesh, subdomain_data=cf) #définition intégrande surface   
    
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
    set_log_level(40) #FENICS N'affiche que les erreurs en principe, en pratique il affiche trop    
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #utilisation d'éléments mixtes car résolution avec conditions de Neumann pures
    #(la solution est donc défninie à une constante près)
    R = FiniteElement("Real", mesh.ufl_cell(), 0)
    V = FunctionSpace(mesh, P1 * R,constrained_domain=pbc)
    n=FacetNormal(mesh) #structure un peu bizzare qui contient la normale ext aux parois du domaine
    V_alpha=assemble(Constant(1.0)*dx) #on intègre l'espace pour obtenir la porosité                
    e_vectors=(Constant((1,0)),Constant((0,1))) 
    u_tuple=()
    
    return P1,R,V,n,V_alpha,e_vectors,u_tuple


class OLI16_PeriodicBC2(SubDomain): #Fonction interne FEniCS d'appairage des bords périodiques
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
    #Avec Gmsh on peut forcer le maillage à être périodique (cad les éléments se correspondent 1 à 1 sur les bords)  
    #matrice des transformations affines pour lier les bords périodiques entre eux
    #selon x
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
        gmsh.model.occ.synchronize() #maj geometrie




#%%
"========================================================"
"========================================================"
"================3D ====================================="
"========================================================"
"========================================================"

"========================================================"
"FONCTIONS PRINCIPALES (A MODIFIER SELON CONFIGURATION)"
"========================================================"

def P20_3D_gmsh_generator(Radius,Radius_impenetrable,Length,Mesh_size,geom,N_obstacles,file_name,MATRICE_OBSTACLES=np.zeros((0)),show_geometry=0): #routine d'appel à gmsh pour génération du maillage
    #Radius : rayon de l'obstacle (sphère ou cylindre)
    #Radius_impenetrable : rayon de l'obstacle (sans intersection), utilisé dans les géoms aléatoires
    #Length : longueur (prise en compte seulement pour cylindre)
    #Mesh_size : taille caract du maillage
    #geom : type de géométrie utilisée
    #N_obstacles : nombre d'obstacles
    #file_name : nom pour l'enregistrement des fichiers 
    #MATRICE_OBSTACLES : coordonnées des obstacles, utile uniquement en géométrie aléatoire pour reprendre une config existante

    L_t=[time.time()]   
    gmsh.initialize()  
    gmsh.option.setNumber('General.Verbosity', 3)
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1) #pourquoi?
    gmsh.option.setNumber("Geometry.Tolerance", 2e-8)
    
    
    eps = 1e-3 #tolérance
    gmsh.model.add("model")  
    refine_obstacle=1
    
    "========================================================"
    "DEFINITION GEOMETRIE"
    
    if geom==0: #Obstacle Sphère 3D
        #On peut faire 1 sphère entière (si ça ne dépasse pas du cube)
        #Sinon il faut faire deux demi sphères (marche dans tous les cas)
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, 10) #domaine cube "10"
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
        gmsh.model.occ.synchronize() #maj geometrie pour synchro MODEL avec OCC
        
        
    elif geom==0.2: #cubique à faces centrées
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, 10) #domaine cube "10"
        
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
        gmsh.model.occ.synchronize() #maj geometrie pour synchro MODEL avec OCC       
        if Radius<np.sqrt(2)/4:
            L_volume=[10] 
            L_bordcarre=[1,2,5,6,9,13]
            L_obstacle=[3,4,7,8,10,11,12]+list(np.arange(14,22,1))
        else :
            L_volume=[10] 
            L_bordcarre=[1,2,7,9,10,11,14,15,16,17,19,22,24,27,29,31,32,33,34,35,36,37,38,39]
            L_obstacle=[3, 4, 5, 6, 8, 12, 13, 18, 20, 21, 23, 25, 26, 28, 30]
            #print('Intersection des sphères non implémenté')
            
    #=============================SPHERES INTERSECTIONS PARTIELLES====================================
    elif geom==1: #spheres aléatoires, intersection partielle
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, 10) #domaine cube "10"
        
        if MATRICE_OBSTACLES.size==0: #= aucune matrice obstacle n'a été fournie
            print('CREATION NOUVELLE GEOMETRIE')
            iteration_max=1000 #itération max pour le placement du k-ième obstacle respectant les intersections partielles
            k=0 #nombre d'obstacles placés (*27) 
            L_x,L_y,L_z=[],[],[] #liste de récupération des coordonnées des obstacles placés     
            for i_fibre in range(N_obstacles):
                flag_validationencours=1 #la tentative de placement du nouvel obstacle n'est pas encore validée
                iteration=0             
                while flag_validationencours and iteration<iteration_max : #BLOC DE RECHERCHE DE POSITION RESPECTANT LE CRITERE D'INTERSECTION PARTIELLE
                    [xa_1,ya_1,za_1]=[random.random(),random.random(),random.random()] #position aléatoire
                    flag_colision=0 #pas encore de colision entre obstacles
                    i=0 #vérification de la non-colision entre le nouvel obstacle et le i-ième obstacle                
                    while i<len(L_x) and flag_colision==0: #on étudie chaque obstacle tant qu'on les a pas tous fait et qu'on a pas de colision
                        d=np.sqrt((xa_1-L_x[i])**2+(ya_1-L_y[i])**2+(za_1-L_z[i])**2) #distance entre les centres spheres
                        if d<2*Radius_impenetrable:
                            flag_colision=1 #on a colision
                        i+=1                 
                    if i==len(L_x) and flag_colision==0:
                        flag_validationencours=0 #on a vérifié tous les obstacles donc c'est bon!
                    iteration+=1
                    if iteration==iteration_max: #si impossible de placer les obstacles : le programme abandonne
                        print("ECHEC POSITIONNEMENT OBSTACLE N°"+str(i_fibre))
                        return #sort de la fonction
                 
                #POSITIONNEMENT DES *27 obstacles (après validation)
                for d_x in range(-1,2):
                    for d_y in range(-1,2):
                        for d_z in range(-1,2):     
                            #definition critere : si la sphere est trop loin du pt (0.5,0.5,0.5) alors ca ne sert à rien de l'ajouter
                            L_cube=1 #taille du cube
                            [x_cube,y_cube,z_cube]=[0.5,0.5,0.5] #position du centre du cube
                            #critère nécessaire pour qu'une sphère intersecte le cube
                            Critere_cube=abs(xa_1+d_x-x_cube)<(L_cube/2+Radius) and abs(ya_1+d_y-y_cube)<(L_cube/2+Radius) and abs(za_1+d_z-z_cube)<(L_cube/2+Radius)
                            if Critere_cube:
                                gmsh.model.occ.addSphere(xa_1+d_x, ya_1+d_y, za_1+d_z, Radius, 1000+k)#on ajoute la sphère
                                L_x.append(xa_1+d_x) #récupération des coordonnées
                                L_y.append(ya_1+d_y)
                                L_z.append(za_1+d_z)
                                k+=1
            MATRICE_OBSTACLES=np.zeros((len(L_x),3)) #récupération des coordonnées
            MATRICE_OBSTACLES[:,:]=np.transpose([L_x,L_y,L_z]) #plus simple pour réutiliser les coordonnées
        
        else : #SI UNE MATRICE OBSTACLE A ETE FOURNIE
            print('UTILISATION GEOMETRIE EXISTANTE')
            N_obstacles_repetes=MATRICE_OBSTACLES.shape[0] #nombre d'obstacles (*27) à placer
            for k in range(N_obstacles_repetes):
                [x,y,z]=MATRICE_OBSTACLES[k,:] #récupération des coordonnées
                gmsh.model.occ.addSphere(x, y, z, Radius, 1000+k)                 
        gmsh.model.occ.synchronize() #maj geometrie        
    
    #=============================CYLINDRES INTERSECTIONS PARTIELLES====================================
    elif geom==2: #
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, 10) #domaine cube "10"
        
        if MATRICE_OBSTACLES.size==0: #PAS DE MATRICE OBSTACLE FOURNIE
            print('CREATION NOUVELLE GEOMETRIE')
            iteration_max=10000
            k=0
            L_xa,L_ya,L_za,L_xb,L_yb,L_zb=[],[],[],[],[],[]            
            Radius_eq=np.sqrt(Radius**2+(Length/2)**2) #rayon de la sphère de colision associée au cylindre
            for i_fibre in range(N_obstacles):  
                flag_validationencours=1
                iteration=0              
                while flag_validationencours and iteration<iteration_max :
                    #POINT : CENTRE BASE DU CYLINDRE
                    [xa,ya,za]=[random.random(),random.random(),random.random()]  #définition du premier point : centre de la base du cylindre
                    #VECTEUR : axe de revolution cylindre
                    [xb,yb,zb]=[random.uniform(-1,2),random.uniform(-1,2),random.uniform(-1,2)]
                    #print('xb,yb,zb='+str([xb,yb,zb]))
                    n_l2=np.sqrt(xb**2+yb**2+zb**2) #calcul norme du vecteur ALEATOIRE
                    [xb,yb,zb]=(Length/n_l2)*np.array([xb,yb,zb]) #CORRECTION NORME VECTEUR=Length
                                
                    flag_colision=0
                    i=0
                    while i<len(L_xa) and flag_colision==0:
                        p1=np.array([xa,ya,za]) #début 1er segment
                        p2=p1+np.array([xb,yb,zb]) #fin 1er segment (fin de l'axe du cylindre)
                        p3=np.array([L_xa[i],L_ya[i],L_za[i]]) #début 2ème segment
                        p4=p3+np.array([L_xb[i],L_yb[i],L_zb[i]])   #fin 2ème segment (fin de l'axe du cylindre)
                        
                        d=closest_line_seg_line_seg(p1, p2, p3, p4) #calcul distance minimale entre les deux cylindres

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
                        print("ECHEC POSITIONNEMENT OBSTACLE N°")  
                        return
                
                #AJOUT DE l'obstacle (après validation)
                [xc,yc,zc]=np.array([xa,ya,za])+np.array([xb,yb,zb])/2#centre du cylindre normé             
                for d_x in range(-1,2): #selon chaque axe : on ajoute 27 fois l'obstacle
                    for d_y in range(-1,2):
                        for d_z in range(-1,2):     
                            L_cube=1
                            [x_cube,y_cube,z_cube]=[0.5,0.5,0.5]
                            #si critere cube valide, l'obstacles a de fortes chances de toucher le cube
                            "Radius_eq*=2 pour fiabiliser "
                            Radius_eq*=2
                            Critere_cube=abs(xc+d_x-x_cube)<(L_cube/2+Radius_eq) and abs(yc+d_y-y_cube)<(L_cube/2+Radius_eq) and abs(zc+d_z-z_cube)<(L_cube/2+Radius_eq)
                            if Critere_cube : #la sphère de colision est susceptible d'intersecter le cube unité            
                                gmsh.model.occ.addCylinder(xa+d_x, ya+d_y, za+d_z, xb, yb, zb, Radius, 1000+k) #donc on ajoute l'obstacle
                                L_xa.append(xa+d_x)
                                L_ya.append(ya+d_y)
                                L_za.append(za+d_z)
                                L_xb.append(xb)
                                L_yb.append(yb)
                                L_zb.append(zb)                              
                                k+=1 #compte le nombre d'obstacle effectivement ajoutés 
            MATRICE_OBSTACLES=np.zeros((k,6))
            MATRICE_OBSTACLES[:,:]=np.transpose([L_xa,L_ya,L_za,L_xb,L_yb,L_zb])
               
        else : #UNE MATRICE OBSTACLE A ETE FOURNIE
            print('UTILISATION GEOMETRIE EXISTANTE')
            N_obstacles_repetes=MATRICE_OBSTACLES.shape[0]
            for k in range (N_obstacles_repetes):
                [xa,ya,za,xb,yb,zb]=MATRICE_OBSTACLES[k,:]
                gmsh.model.occ.addCylinder(xa, ya, za, xb, yb, zb, Radius, 1000+k) #donc on ajoute l'obstacle                                   
        gmsh.model.occ.synchronize() #maj geometrie pour synchro MODEL avec OCC  
        
    #============================= CYLINDRES + SPHERES : Factin + Ribosomes ====================================
    elif geom==3: #
        N_act=N_obstacles[0] ; N_rib=N_obstacles[1]
        Radius_act=Radius[0] ; Radius_rib=Radius[1]

        Radius_impenetrable_rib=Radius_impenetrable*Radius_rib
        Radius_impenetrable_act=Radius_impenetrable*Radius_act
    
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, 10) #domaine cube "10"
        
        if len(MATRICE_OBSTACLES)==0: #PAS DE MATRICE OBSTACLE FOURNIE
            
            '====Positionnement des SPHERES===='
            print('CREATION NOUVELLE GEOMETRIE')
            iteration_max=1000 #itération max pour le placement du k-ième obstacle respectant les intersections partielles
            k=0 #nombre d'obstacles placés (*27) 
            L_x,L_y,L_z=[],[],[] #liste de récupération des coordonnées des obstacles placés     
            for i_fibre in range(N_rib):
                flag_validationencours=1 #la tentative de placement du nouvel obstacle n'est pas encore validée
                iteration=0             
                while flag_validationencours and iteration<iteration_max : #BLOC DE RECHERCHE DE POSITION RESPECTANT LE CRITERE D'INTERSECTION PARTIELLE
                    [xa_1,ya_1,za_1]=[random.random(),random.random(),random.random()] #position aléatoire
                    flag_colision=0 #pas encore de colision entre obstacles
                    i=0 #vérification de la non-colision entre le nouvel obstacle et le i-ième obstacle                
                    while i<len(L_x) and flag_colision==0: #on étudie chaque obstacle tant qu'on les a pas tous fait et qu'on a pas de colision
                        d=np.sqrt((xa_1-L_x[i])**2+(ya_1-L_y[i])**2+(za_1-L_z[i])**2) #distance entre les centres spheres
                        if d<2*Radius_impenetrable_rib:
                            flag_colision=1 #on a colision
                        i+=1                 
                    if i==len(L_x) and flag_colision==0:
                        flag_validationencours=0 #on a vérifié tous les obstacles donc c'est bon!
                    iteration+=1
                    if iteration==iteration_max: #si impossible de placer les obstacles : le programme abandonne
                        print("ECHEC POSITIONNEMENT OBSTACLE N°"+str(i_fibre))
                        return #sort de la fonction
                 
                #POSITIONNEMENT DES *27 obstacles (après validation)
                for d_x in range(-1,2):
                    for d_y in range(-1,2):
                        for d_z in range(-1,2):     
                            #definition critere : si la sphere est trop loin du pt (0.5,0.5,0.5) alors ca ne sert à rien de l'ajouter
                            L_cube=1 #taille du cube
                            [x_cube,y_cube,z_cube]=[0.5,0.5,0.5] #position du centre du cube
                            #critère nécessaire pour qu'une sphère intersecte le cube
                            "mod 2x radius pour générer des trucs qui marchent même avec un rayon plus gros"
                            Radius_eq=Radius_rib
                            Radius_eq*=2
                            "==============="
                            Critere_cube=abs(xa_1+d_x-x_cube)<(L_cube/2+Radius_eq) and abs(ya_1+d_y-y_cube)<(L_cube/2+Radius_eq) and abs(za_1+d_z-z_cube)<(L_cube/2+Radius_eq)
                            if Critere_cube:
                                gmsh.model.occ.addSphere(xa_1+d_x, ya_1+d_y, za_1+d_z, Radius_rib, 1000+k)#on ajoute la sphère
                                L_x.append(xa_1+d_x) #récupération des coordonnées
                                L_y.append(ya_1+d_y)
                                L_z.append(za_1+d_z)
                                k+=1   
                MATRICE_OBSTACLES_spheres=np.zeros((len(L_x),3)) #récupération des coordonnées
                MATRICE_OBSTACLES_spheres[:,:]=np.transpose([L_x,L_y,L_z]) #plus simple pour réutiliser les coordonnées
        
            '====Positionnement des CYLINDRES===='
            iteration_max=10000
            #k=0
            L_xa,L_ya,L_za,L_xb,L_yb,L_zb=[],[],[],[],[],[]       
            Radius_eq=np.sqrt(Radius_act**2+(Length/2)**2) #rayon de la sphère de colision associée au cylindre
            for i_fibre in range(N_act):  
                flag_validationencours=1
                iteration=0              
                while flag_validationencours and iteration<iteration_max :
                    #POINT : CENTRE BASE DU CYLINDRE
                    [xa,ya,za]=[random.random(),random.random(),random.random()]  #définition du premier point : centre de la base du cylindre
                    #VECTEUR : axe de revolution cylindre
                    [xb,yb,zb]=[random.uniform(-1,2),random.uniform(-1,2),random.uniform(-1,2)]
                    #print('xb,yb,zb='+str([xb,yb,zb]))
                    n_l2=np.sqrt(xb**2+yb**2+zb**2) #calcul norme du vecteur ALEATOIRE
                    [xb,yb,zb]=(Length/n_l2)*np.array([xb,yb,zb]) #CORRECTION NORME VECTEUR=Length
                                
                    flag_colision_act_act=0 ; flag_colision_act_rib=0
                    i=0
                    while i<len(L_xa) and flag_colision_act_act==0:
                        p1=np.array([xa,ya,za]) #début 1er segment
                        p2=p1+np.array([xb,yb,zb]) #fin 1er segment (fin de l'axe du cylindre)
                        p3=np.array([L_xa[i],L_ya[i],L_za[i]]) #début 2ème segment
                        p4=p3+np.array([L_xb[i],L_yb[i],L_zb[i]])   #fin 2ème segment (fin de l'axe du cylindre)
                        
                        d=closest_line_seg_line_seg(p1, p2, p3, p4) #calcul distance minimale entre les deux cylindres

                        if d<2*Radius_impenetrable_act:
                            flag_colision=1
                            if iteration==1000:
                                print('echec*1000')
                        i+=1       
                    j=0  
                    while j<len(L_x) and flag_colision_act_rib==0: #CALCUL COLLISIONS CYLINDRE SPHERE
                        p1=np.array([xa,ya,za]) #début 1er segment
                        p2=p1+np.array([xb,yb,zb]) #fin 1er segment (fin de l'axe du cylindre)
                        p3=np.array([L_x[j]-eps,L_y[j]-eps,L_z[j]-eps]) #début 2ème segment
                        p4=p3+np.array([L_x[j]+eps,L_y[j]+eps,L_z[j]+eps])   #fin 2ème segment (fin de l'axe du cylindre)
                            
                        d=closest_line_seg_line_seg(p1, p2, p3, p4) #calcul distance minimale entre les deux cylindres

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
                        print("ECHEC POSITIONNEMENT OBSTACLE N°")  
                        return
                
                #AJOUT DE l'obstacle (après validation)
                [xc,yc,zc]=np.array([xa,ya,za])+np.array([xb,yb,zb])/2#centre du cylindre normé             
                for d_x in range(-1,2): #selon chaque axe : on ajoute 27 fois l'obstacle
                    for d_y in range(-1,2):
                        for d_z in range(-1,2):     
                            L_cube=1
                            [x_cube,y_cube,z_cube]=[0.5,0.5,0.5]
                            #si critere cube valide, l'obstacles a de fortes chances de toucher le cube
                            "mod 2x radius pour générer des trucs qui marchent même avec un rayon plus gros"
                            Radius_eq*=2
                            "==============="
                            Critere_cube=abs(xc+d_x-x_cube)<(L_cube/2+Radius_eq) and abs(yc+d_y-y_cube)<(L_cube/2+Radius_eq) and abs(zc+d_z-z_cube)<(L_cube/2+Radius_eq)
                            if Critere_cube : #la sphère de colision est susceptible d'intersecter le cube unité            
                                gmsh.model.occ.addCylinder(xa+d_x, ya+d_y, za+d_z, xb, yb, zb, Radius_act, 1000+k) #donc on ajoute l'obstacle
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
               
        else : #UNE MATRICE OBSTACLE A ETE FOURNIE
            MATRICE_OBSTACLES_spheres,MATRICE_OBSTACLES_cylindres=MATRICE_OBSTACLES
            "Spheres"
            N_obstacles_repetes=MATRICE_OBSTACLES_spheres.shape[0]
            for k_1 in range (N_obstacles_repetes):
                [x,y,z]=MATRICE_OBSTACLES_spheres[k_1,:]
                gmsh.model.occ.addSphere(x,y,z, Radius_rib, 1000+k_1)#on ajoute la sphère  
            "Cylindres"
            N_obstacles_repetes=MATRICE_OBSTACLES_cylindres.shape[0]
            for k_2 in range (N_obstacles_repetes):
                [xa,ya,za,xb,yb,zb]=MATRICE_OBSTACLES_cylindres[k_2,:]
                gmsh.model.occ.addCylinder(xa, ya, za, xb, yb, zb, Radius_act, 2000+k_2) #donc on ajoute l'obstacle                                   
        
        
    gmsh.model.occ.synchronize() #maj geometrie  

        
        
        
    if geom in [1,2,3]:
        #cut des obstacles (dont la majorité devrait couper le cube, mais pas forcément tous)
        All_volumes=gmsh.model.getEntitiesInBoundingBox(-2-eps,-2-eps,-2-eps,3+eps,3+eps,3+eps,3)
        Main_volume=[(3,10)] #volume du cube unité 

        Cut_volumes=list(set(All_volumes)-set(Main_volume)) #liste des obstacles à retirer
        gmsh.model.occ.cut(Main_volume,Cut_volumes) #beaucoup plus efficace de retirer toute la liste d'un coup
        print('cut done')
        gmsh.model.occ.synchronize() #maj geometrie  

        #suppression des volumes interieurs isolés
        Internal_volumes=gmsh.model.getEntitiesInBoundingBox(+eps,+eps,+eps,1-eps,1-eps,1-eps,3) #morceaux intérieurs au cube unité   
        gmsh.model.occ.remove(Internal_volumes) #suppression des volumes intérieurs du cube
        gmsh.model.occ.synchronize() #maj geometrie  
        
        print('nombre de volumes'+str(gmsh.model.getEntities(3)))
        
        
        # #suppression des culs de sacs
        # Border_volumes=[]
        # Border_volumes+=gmsh.model.getEntitiesInBoundingBox(-eps,-eps,-eps,1+eps,1+eps,1-eps,3)
        # Border_volumes+=gmsh.model.getEntitiesInBoundingBox(-eps,-eps,-eps,1+eps,1-eps,1+eps,3)
        # Border_volumes+=gmsh.model.getEntitiesInBoundingBox(-eps,-eps,-eps,1-eps,1+eps,1+eps,3)
        
        # Border_volumes+=gmsh.model.getEntitiesInBoundingBox(+eps,-eps,-eps,1+eps,1+eps,1+eps,3)
        # Border_volumes+=gmsh.model.getEntitiesInBoundingBox(-eps,+eps,-eps,1+eps,1+eps,1+eps,3)
        # Border_volumes+=gmsh.model.getEntitiesInBoundingBox(-eps,-eps,+eps,1+eps,1+eps,1+eps,3)
        
        # gmsh.model.occ.remove(Border_volumes) #suppression des volumes intérieurs du cube
        # gmsh.model.occ.synchronize() #maj geometrie  
        
        # print('nombre de volumes'+str(gmsh.model.getEntities(3)))
        
        
        L_volume=gmsh.model.getEntities(3)#gmsh.model.getEntitiesInBoundingBox( - eps, -eps, -eps,  1+ eps, 1 + eps, 1 + eps, 3)
        L_volume=[x[1] for x in L_volume]
        print (L_volume)
        
        
        # if len(gmsh.model.getEntities(3))>1:
        #     L_volume=[]
        #     for i in range (len(gmsh.model.getEntities(3))):
        #         L_volume.append(gmsh.model.getEntities(3)[i][1])
        #     print('erreur trop de volumes')

            #eturn 'aaa'
            
        # else : 
        #     L_volume=[gmsh.model.getEntities(3)[0][1]]
        
        
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
    "GROUPES PHYSIQUES"
    gmsh.model.addPhysicalGroup(2, L_bordcarre,2)
    gmsh.model.addPhysicalGroup(2, L_obstacle,3)
    gmsh.model.addPhysicalGroup(3, L_volume,1)
    
    "========================================================"
    "BORDS PERIODIQUES"
    Set_periodic_boundaries(eps)

    "========================================================"
    "MAILLAGE"
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
    "EXPORTATION"   
    #gmsh.write(file_name+".brep") #
    gmsh.write(file_name+".msh")
    L_t.append(time.time())
    print('Temps Gmsh total : '+str(L_t[-1]-L_t[0])+'s')
    gmsh.finalize()
    return MATRICE_OBSTACLES
    
    
#def OLI18_sMAJenCOURS_olver_perm_3D:   
def closest_line_seg_line_seg(p1, p2, p3, p4):
    "distance deux segments 3D : https://math.stackexchange.com/questions/846054/closest-points-on-two-line-segments" 

    P1 = p1 #1er pt du 1er segment
    P2 = p3 #1er pt du 2eme segment
    V1 = p2 - p1 #vecteur 1er segment 
    V2 = p4 - p3 #vecteur 2ème segment
    V21 = P2 - P1 #vecteur_diff : 1er_pt_2eme segment-1er_pt_1er segment

    v22 = np.dot(V2, V2) #norme² 2eme vecteur
    v11 = np.dot(V1, V1) #norme² 1er vecteur
    v21 = np.dot(V2, V1) #PS des deux vecteurs
    v21_1 = np.dot(V21, V1) #PS vecteur_diff et 1er vecteur
    v21_2 = np.dot(V21, V2)#PS vecteur_diff et 2ème vecteur
    denom = v21 * v21 - v22 * v11 #=0 ssi vecteurs colinéaires (ou nul)

    if np.isclose(denom, 0.): #cas ou les vecteurs sont parallèles
        s = 0.
        t = (v11 * s - v21_1) / v21
    else: #les vecteurs ne sont pas parallèles : calcul des deux pts qui tq le vecteur qui les relie est ortho aux deux vecteurs
        s = (v21_2 * v21 - v22 * v21_1) / denom
        t = (-v21_1 * v21 + v11 * v21_2) / denom

    s = max(min(s, 1.), 0.) #si jamais les pts sont hors du segment, alors le meilleur pt sur le segment est sa borne
    t = max(min(t, 1.), 0.)

    p_a = P1 + s * V1 #point absolu
    p_b = P2 + t * V2
    
    d=np.sqrt(np.dot(p_b-p_a,p_b-p_a)) #distance minimale entre les segments
    return d 
    

class OLI16_PeriodicBC3(SubDomain): #Fonction interne FEniCS d'appairage des bords périodiques
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
    #Avec Gmsh on peut forcer le maillage à être périodique (cad les éléments se correspondent 1 à 1 sur les bords)  
    #matrice des transformations affines pour lier les bords périodiques entre eux
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
                #print('Correspondance x trouvée')
                gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], translation)
    #"LES BLOCS D'APPAIRAGE PERIODIQUE SELON LES DIRECTION Y ET Z FONCTIONNENT
    #Fonction d'appairage des bords périodiques (selon y)
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
                #print('Correspondance y trouvée')
                gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], translation2)             
    #Fonction d'appairage des bords périodiques (selon z)
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
                #print('Correspondance z trouvée')
                gmsh.model.mesh.setPeriodic(2, [j[1]], [i[1]], translation3)
    gmsh.model.occ.synchronize() #maj geometrie





"========================================================"
"FONCTIONS PRINCIPALES (A MODIFIER SELON CONFIGURATION)"
"========================================================"

def OLI16_solver_volume_3D(file_name):
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_3D(file_name)
    pbc = OLI16_PeriodicBC3()   #fonction d'appairage des bords périodiques sur FEniCS
    P1,R,V,n,V_alpha,e_vectors,u_tuple=OLI16_init_diff_fenics_3D(mesh,dx,pbc)
    Resultat=V_alpha #porosité, puis autres résultats      
    L_t.append(time.time())
    #print('Temps FEniCS total : '+str(L_t[-1]-L_t[0])+'s')
    return Resultat
    
def OLI16_solver_diff_3D(file_name): #routine d'appel à FEniCS pour résolution du problème FEM
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_3D(file_name)
    pbc = OLI16_PeriodicBC3()   #fonction d'appairage des bords périodiques sur FEniCS
    P1,R,V,n,V_alpha,e_vectors,u_tuple=OLI16_init_diff_fenics_3D(mesh,dx,pbc)
    Resultat=[V_alpha] #porosité, puis autres résultats



    "========================================================"
    "ELEMENTS FINIS"
    
    for i_dir in range(3): #calcul selon les 3 directions pour avoir Dxx Dyy et Dzz   
        e_x=e_vectors[i_dir]
 
        # Define variational problem
        (u, c) = TrialFunction(V) #solution cherchées : u est la sol, c la constante
        (v, d) = TestFunction(V) #fonctions test
        a = (dot(grad(u), grad(v))+ c*v + u*d)*dx #forme bilinéaire
        L = -dot(e_x,n)*v*ds(3)  #f*v*dx-g*dot(e_x,n)*v*ds #forme linéaire. Contient la condition de neumann aux abords des obstacles
    
        "========================================================"
        "SOLVER"
    
        w = Function(V) #on définit une nouvelle fonction qui sera la solution
        
        # prm = parameters.krylov_solver  # short form
        # prm.absolute_tolerance = 1E-10
        # prm.relative_tolerance = 1E-6
        # prm.maximum_iterations = 
        
        
        solve(a == L, w,solver_parameters={'linear_solver': 'gmres'}) #UTILISATION GMRES OPTIMALE, precondtionner non réglé

        
        (u, c) = w.split() #séparation u et constante c
    
        "========================================================"
        "CALCUL VARIABLES GLOBALES"

        dxx = assemble(dot(e_x, n)*u*ds(3)) #on intègre bx sur cette surface
        Dxx_ad=1+(1/V_alpha)*dxx #on obtient la compo Dxx du tenseur
        epsi_Dad=V_alpha*Dxx_ad #on obtient la grandeur qu'on plot habituellement        
        Resultat.append(epsi_Dad) #ajout du résultat 
        u_tuple=u_tuple+(u,) #sauvegarde du champ solution
        
    L_t.append(time.time())
    print('Temps FEniCS_diff total : '+str(L_t[-1]-L_t[0])+'s')
    return (Resultat,)+u_tuple

def OLI18_solver_perm_3D(file_name):   
    L_t=[time.time()]
    mesh,ds,dx, boundaries, subdomains=OLI16_importation_mesh_3D(file_name)
    pbc = OLI16_PeriodicBC3()   #fonction d'appairage des bords périodiques sur FEniCS
    P1,P2,V,n,V_alpha,e_vectors,u_tuple=OLI16_init_perm_fenics_3D(mesh,dx,pbc)
    noslip = Constant((0.0, 0.0,0.0)) #vitesse nulle aux parois
    bc_obstacle=DirichletBC(V.sub(0), noslip, boundaries,3) #"""BOUNDARY NUMBER SELON GEOMETRIE"""" #W.sub(0) c'est le vecteur vitesse 
    bcs = [bc_obstacle] #ligne utilise s'il y'a plusieurs conditions de Dirichlet à implémenter (ici non)   
    
    "========================================================"
    "ELEMENTS FINIS"
    
    (u, p) = TrialFunctions(V) #"""ATTENTION FEniCS utilise p_prog=-p_reelle""" # Define variational problem
    (v, q) = TestFunctions(V)
    f = Constant((1, 0.0,0.0)) #terme source pour le probleme colonne _x (pour _y ou _z il faut changer la composante non nulle)
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
    L = inner(f, v)*dx
    # PRECONDITIONNEMENT (pour améliorer conditionnement)
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
    # L_t.append(time.time())
    # print('Temps FEniCS prepa : '+str(L_t[-1]-L_t[-2])+'s')
    
    "========================================================"
    "SOLVER"
    
    solver.solve(U.vector(), bb)    
    # L_t.append(time.time())
    # print('Temps FEniCS solve : '+str(L_t[-1]-L_t[-2])+'s')
    
    "========================================================"
    "CALCUL VARIABLES GLOBALES"
    
    # Get sub-functions
    u, p = U.split() #séparation vecteur vitesse et pression NEGATIVE      #p=-p 
    #Je calcule toute la première colonne du tenseur de perméabilité
    #meme si je ne me sers que du coef kxx car 
    #le problème de stokes résolu permet d'avoir aussi kyx et kzx gratos
    V_tot=1
    #Epsi=V_alpha/V_tot #porosité    
    kperm_px=[(1/V_tot)*assemble(dot(u,e_vectors[0])*dx(mesh)),(1/V_tot)*assemble(dot(u,e_vectors[1])*dx(mesh)),(1/V_tot)*assemble(dot(u,e_vectors[2])*dx(mesh))] 
    Kx=kperm_px[0] #jon considère seulement kxx car kyx et kzx sont à priori très petits
    #B_px=u*(Epsi/Kx)-e_vectors[0]
    #v_moy_x=1
    #v_moyint=[v_moy_x,0,0]
    #v_tilde=B_px*v_moy_x
    #v_tot=v_tilde+Constant((v_moyint[0],v_moyint[1],v_moyint[2]))     
    #C_drag=(2*Radius**2)/(9*(1-Epsi))*(1/Kx) #coefficient de trainée utilisé par Morgan pour cas test
    #abscisse_SU=((1-Epsi)**(1/3))/Epsi #abscisse utilisée par Morgan pour cas test  
    # L_t.append(time.time())
    # print('Temps FEniCS varglob : '+str(L_t[-1]-L_t[-2])+'s')
    L_t.append(time.time())
    print('Temps FEniCS_perm total : '+str(L_t[-1]-L_t[0])+'s')  
    return (V_alpha, Kx)#,abscisse_SU,C_drag)#(abscisse_SU,C_drag,B_px,Kx,v_tot,v_tilde,v_moy_x) #Epsi,int_vtot_surf



"========================================================"
"FONCTIONS SECONDAIRES (NE PAS MODIFIER EN PRINCIPE)"
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
    
    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()) #récupération éléments triangles (surfaces)
    with XDMFFile(file_name+"_mesh.xdmf") as infile:
       infile.read(mesh)
       infile.read(mvc, "name_to_read")
    cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1) #récupération éléments segments (bordures)
    with XDMFFile(file_name+"_mf.xdmf") as infile:
        infile.read(mvc, "name_to_read")   
    mf = cpp.mesh.MeshFunctionSizet(mesh, mvc) 
    
    ds = Measure("ds", domain=mesh, subdomain_data=mf) #définition intégrande bordure
    dx = Measure("dx", domain=mesh, subdomain_data=cf) #définition intégrande surface   
    
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
    set_log_level(40) #FENICS N'affiche que les erreurs en principe, en pratique il affiche trop    
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #("Lagrange", mesh.ufl_cell(), 1) #utilisation d'éléments mixtes car résolution avec conditions de Neumann pures
    #(la solution est donc défninie à une constante près)
    R = FiniteElement("Real", mesh.ufl_cell(), 0)
    V = FunctionSpace(mesh, P1 * R,constrained_domain=pbc)
    n=FacetNormal(mesh) #structure un peu bizzare qui contient la normale ext aux parois du domaine
    V_alpha=assemble(Constant(1.0)*dx) #on intègre l'espace pour obtenir la porosité                
    e_vectors=(Constant((1,0,0)),Constant((0,1,0)),Constant((0,0,1))) 
    u_tuple=()
    
    return P1,R,V,n,V_alpha,e_vectors,u_tuple

def OLI16_init_perm_fenics_3D(mesh,dx,pbc):
    set_log_level(40) #FENICS N'affiche que les erreurs en principe, en pratique il affiche trop    
    # Define function spaces : on utilise des éléments mixtes pour la résolution directe du problème
    P2 = VectorElement("CG", mesh.ufl_cell(), 2) #ordre 2 nécessaire pour résolution Stokes sinon instable 
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1) #élément pour la pression
    TH = MixedElement([P2, P1]) #élément mixte de Taylor Hood
    V = FunctionSpace(mesh, TH,constrained_domain=pbc) #condition périodique implémentée ici    

    n=FacetNormal(mesh) #structure un peu bizzare qui contient la normale ext aux parois du domaine
    V_alpha=assemble(Constant(1.0)*dx) #on intègre l'espace pour obtenir la porosité                
    e_vectors=(Constant((1,0,0)),Constant((0,1,0)),Constant((0,0,1))) 
    u_tuple=()
    
    return P1,P2,V,n,V_alpha,e_vectors,u_tuple
