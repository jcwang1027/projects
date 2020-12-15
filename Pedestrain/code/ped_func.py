# -*- coding: utf-8 -*-
"""

@author: Jiachun Wang
"""

import numpy as np
import skfmm
#pip install scikit-fmm
import scipy
from scipy.integrate import odeint
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage.morphology import binary_dilation
from IPython import display
from scipy.spatial.distance import cdist
import time
import pandas as pd

def h_function(r,p,R):
  "H exponential function"
  if abs(r/R)<1:
    return p*np.exp(1/((r/R)**2-1))
  else:
    return 0

def moll(x,R,p):
  "Moll function"
  if np.linalg.norm(x)<R:
    return np.exp(1)*np.exp(1/((np.linalg.norm(x)/R)**(2*p)-1))
  else:
    return 0

def r_moll(x):
  "R_moll function"
  return moll(x,1,3)*x+(1-moll(x,1,3))

def g_scale(x):
  "Rescaling function to [0,1]"
  if np.linalg.norm(x)==0:
    return np.array([0,0])
  else:
    return np.array(x/np.linalg.norm(x)*r_moll(np.linalg.norm(x)))


def fun(x,P_B=11.3,R_B=0.25):
  "Gauss-ledengre"
  if abs(np.linalg.norm(x)/P_B)<1:
    return(R_B*np.exp((x/P_B)**2-1)**(-1))
  else:
    return 0



def find_angle(vector_1,vector_2): 
  "Find the angle between two vector"
  unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
  dot_product = np.dot(unit_vector_1, unit_vector_2)
  angle = np.arccos(dot_product)
  return angle

def g_tilda(vec,x0=0.3,R=0.03):
  "One dimension rescale function"  
  return 1/(1+np.exp(-(vec-x0)/R))

def find_vec(coord_x,coord_y):
  "Get array for the two points"
  return np.array([coord_x,coord_y])

def find_length(x1,y1):
  "Get the length of two point"  
  return np.linalg.norm(find_vec(x1,y1))

def unit_vec(x1,y1):
  "Find unit vector"  
  if find_length(x1,y1) == 0:  
     return 0
  else:
     return find_vec(x1,y1)/find_length(x1,y1)

def fdiff(v1,v2,diff):
  return (v2-v1)/diff
def bdiff(v1,v2,diff):
  return (v1-v2)/diff

def cdiff(v1,v2,v3,diff):
  return (v1-2*v2+v3)/(diff**2)

"Forward and backward scheme if on boundary and central difference if on interior"  
def check_x(coord_i,coord_j,dx,datas,indexs):
  if indexs[coord_i+1,coord_j] == False:
    return fdiff(datas[coord_i,coord_j],datas[coord_i+1,coord_j],dx)
  elif indexs[coord_i-1,coord_j] == False:
    return bdiff(datas[coord_i,coord_j],datas[coord_i-1,coord_j],dx)

def check_y(coord_i,coord_j,dy,datas,indexs):
   if indexs[coord_i,coord_j+1] == False:
    return fdiff(datas[coord_i,coord_j],datas[coord_i,coord_j+1],dy)
   elif indexs[coord_i,coord_j-1] == False:
    return bdiff(datas[coord_i,coord_j],datas[coord_i,coord_j-1],dy)


def check_x_2nd(coord_i,coord_j,dx,datas,indexs):
  if indexs[coord_i+1,coord_j] == True or indexs[coord_i-1,coord_j] == True:
    return cdiff(datas[coord_i,coord_j],datas[coord_i+1,coord_j],datas[coord_i-1,coord_j],dy)
  if indexs[coord_i+1,coord_j] == False:
    return fdiff(datas[coord_i,coord_j],datas[coord_i+1,coord_j],dx)
  elif indexs[coord_i-1,coord_j] == False:
    return bdiff(datas[coord_i,coord_j],datas[coord_i-1,coord_j],dx)


def check_y_2nd(coord_i,coord_j,dy,datas,indexs):
   if indexs[coord_i,coord_j+1] == True or indexs[coord_i,coord_j-1] == True:
     return cdiff(datas[coord_i,coord_j],datas[coord_i,coord_j+1],datas[coord_i,coord_j-1],dy)
   if indexs[coord_i,coord_j+1] == False:
    return fdiff(datas[coord_i,coord_j],datas[coord_i,coord_j+1],dy)
   elif indexs[coord_i,coord_j-1] == False:
    return bdiff(datas[coord_i,coord_j],datas[coord_i,coord_j-1],dy)


"Find the graidient of sigma"
def find_grad(t,dx,dy,grid_point):
  Grad_sigma = np.zeros((2,grid_point,grid_point))
  ind = np.ma.getmaskarray(t)
  data = t.data
  for i in range(grid_point):
    for j in range(grid_point):
      #If not obstacle point continoue
      if ind[i,j] == False:
        #Corner cases
        if i==0 and j==0:
          Grad_sigma[:,i,j] = -1*np.array([fdiff(data[i,j],data[i+1,j],dx),fdiff(data[i,j],data[i,j+1],dy)])
        elif i==0 and j==grid_point-1:
          Grad_sigma[:,i,j] = -1*np.array([fdiff(data[i,j],data[i+1,j],dx),bdiff(data[i,j],data[i,j-1],dy)])
        elif i==grid_point-1 and j==0:
          Grad_sigma[:,i,j] = -1*np.array([bdiff(data[i,j],data[i-1,j],dx),fdiff(data[i,j],data[i,j+1],dy)])
        elif i==grid_point-1 and j== grid_point-1:
          Grad_sigma[:,i,j] = -1*np.array([bdiff(data[i,j],data[i-1,j],dx),bdiff(data[i,j],data[i,j-1],dy)])
        #Edge cases
        elif i==0:   
          Grad_sigma[:,i,j] = -1*np.array([fdiff(data[i,j],data[i+1,j],dx),check_y(i,j,dy,data,ind)])
        elif i==grid_point-1:
          Grad_sigma[:,i,j] = -1*np.array([bdiff(data[i,j],data[i-1,j],dx),check_y(i,j,dy,data,ind)])
        elif j==0:
          Grad_sigma[:,i,j] = -1*np.array([check_x(i,j,dx,data,ind),fdiff(data[i,j],data[i,j+1],dy)])
        elif j==grid_point-1:
          Grad_sigma[:,i,j] = -1*np.array([check_x(i,j,dx,data,ind),bdiff(data[i,j],data[i,j-1],dy)])
        else:
          #Interior case
          Grad_sigma[:,i,j] = -1*np.array([check_x(i,j,dx,data,ind),check_y(i,j,dy,data,ind)])
  return (Grad_sigma)

"Find the boundary points of the walls"
def find_obstacle_boundary(t,mesh_X,mesh_Y):
  ind = np.ma.getmaskarray(t)
  test = np.zeros(ind.shape,dtype=int)
  test[ind==True] = 1
  k = np.zeros((3,3),dtype=int)
  k[1]=1
  k[:,1] = 1
  out = binary_dilation(test==0, k) & test
  return mesh_X[np.where(out==1)],mesh_Y[np.where(out==1)]

"Find the index of boundary points of the walls"
def find_obstacle_boundary_index(t,mesh_X,mesh_Y):
  ind = np.ma.getmaskarray(t)
  test = np.zeros(ind.shape,dtype=int)
  test[ind==True] = 1
  k = np.zeros((3,3),dtype=int)
  k[1]=1
  k[:,1] = 1
  out = binary_dilation(test==0, k) & test
  return np.where(out==1)

"Sum p_ij"
def sum_pij(s):
  mat = np.zeros((s.shape[0],s.shape[2]))
  for i in range(s.shape[0]):
    for j in range(s.shape[1]):
      if i!=j:
        mat[i,:] += s[i,j,:]

  return mat
"Remove pedestrian reached exit"
def remove_0vec(N_vector,Y_vector,velo,num,exit_ind,ped_ind,end_point_x,end_point_y):
  N_grad_T_new = []
  velo_new = []
  x_posi = []
  y_posi = []
  speeds = []
  N_grad_T_list = N_vector.tolist()
  velo_new_list = velo.tolist()
  for indd in range(len(N_vector)):
    if ((Y_vector[indd]>=-end_point_x) and (Y_vector[indd]<=end_point_x) and (Y_vector[indd+num]>=-end_point_y) and (Y_vector[indd+num]<=end_point_y)) and np.linalg.norm(N_vector[indd]) != 0:
      x_posi.append(Y_vector[indd])
      y_posi.append(Y_vector[indd+num])
      speeds.append(Y_vector[indd+2*num])
      N_grad_T_new.append(N_grad_T_list[indd])
      velo_new.append(velo_new_list[indd])
    else:
      exit_ind.append(ped_ind[indd])
      ped_ind = np.delete(ped_ind,indd)

  return np.array(N_grad_T_new),np.array(x_posi+y_posi+speeds),np.array(velo_new),exit_ind,ped_ind
"Periodic boundary where pedestrian start again at initial"
def remove_0vec_pe(N_vector,Y_vector,current_speed,velo,num,init,init_speed,exit_ind,ped_ind,end_point_x,end_point_y,xx,yy,N_iT):
  #periodic boundary
  init = np.array(init)
  N_grad_T_new = []
  velo_new = []
  x_posi = []
  y_posi = []
  speeds = []
  N_grad_T_list = N_vector.tolist()
  velo_new_list = velo.tolist()
  for indd in range(len(N_vector)):
    if np.linalg.norm(N_vector[indd]) != 0 and (Y_vector[indd]>=-end_point_x) and (Y_vector[indd]<=end_point_x) and (Y_vector[indd+num]>=-end_point_y) and (Y_vector[indd+num]<=end_point_y):
      x_posi.append(Y_vector[indd])
      y_posi.append(Y_vector[indd+num])
      speeds.append(Y_vector[indd+2*num])
      N_grad_T_new.append(N_grad_T_list[indd])
      velo_new.append(velo_new_list[indd])
    else:
      x_posi.append(init[indd,0])
      y_posi.append(init[indd,1])
      speeds.append(current_speed[indd])
      N_grad_T_new.append((griddata((xx,yy),N_iT.reshape(-1,2),(init[indd]),method='nearest'))[0])
      velo_new.append(velo_new_list[indd])
      exit_ind += 1
  return np.array(N_grad_T_new),np.array(x_posi+y_posi+speeds),np.array(velo_new),exit_ind,ped_ind

"Gauss-legendre quadrature rule"
def Gauss_legendre(N_iT,a,b,grid_point):
  #Perform gauss-legendre quadrature on interval [a,b]
  eta_func = np.vectorize(fun)
  N_iT_new = np.zeros(N_iT.shape)
  # Gauss-Legendre (default interval is [-1, 1])
  deg = 21
  x_point, weight = np.polynomial.legendre.leggauss(deg)


  # Translate x_point values from the interval [-1, 1] to [a, b]
  y_point = 0.5*(x_point + 1)*(b - a) + a
  points = np.array(np.meshgrid(y_point,y_point)).T.reshape(-1,2)

  for i in range(grid_point):
    for j in range(grid_point):
      origin_point = np.array((X[i,j],Y[i,j]))

      com_points = origin_point - points
      f = eta_func(np.linalg.norm(np.array(np.meshgrid(y_point,y_point)).T.reshape(-1,2),axis=1))[:,np.newaxis]*(np.array(griddata((xx,yy),N_iT.reshape(-1,2),com_points,method='nearest')))
      N_iT_new[i,j] = sum((weight*weight[:,np.newaxis]).ravel()[:,np.newaxis]*f)* (0.5*(b - a))**2

  return N_iT_new




"Smooth the boundary of grad sigma"
def smooth_boundary_grad(N_iT,obstacle_index,grid_point):
  for i,j in zip(obstacle_index[0],obstacle_index[1]):
    if i==0 and j==0:
      N_iT[i+1,j][np.argmin(abs(N_iT[i+1,j]))] = 0
      N_iT[i,j+1][np.argmin(abs(N_iT[i,j+1]))] = 0
    elif i==0 and j==grid_point-1:
      N_iT[i+1,j][np.argmin(abs(N_iT[i+1,j]))] = 0
      N_iT[i,j-1][np.argmin(abs(N_iT[i,j-1]))] = 0
    elif i==grid_point-1 and j==0:
      N_iT[i-1,j][np.argmin(abs(N_iT[i-1,j]))] = 0
      N_iT[i,j+1][np.argmin(abs(N_iT[i,j+1]))] = 0
    elif i==grid_point-1 and j== grid_point-1:
      N_iT[i-1,j][np.argmin(abs(N_iT[i-1,j]))] = 0
      N_iT[i,j-1][np.argmin(abs(N_iT[i,j-1]))] = 0
    #Edge cases
    elif i==0:   
      N_iT[i+1,j][np.argmin(abs(N_iT[i+1,j]))] = 0
      N_iT[i,j+1][np.argmin(abs(N_iT[i,j+1]))] = 0
      N_iT[i,j-1][np.argmin(abs(N_iT[i,j-1]))] = 0
    elif i==grid_point-1:
      N_iT[i-1,j][np.argmin(abs(N_iT[i-1,j]))] = 0
      N_iT[i,j+1][np.argmin(abs(N_iT[i,j+1]))] = 0
      N_iT[i,j-1][np.argmin(abs(N_iT[i,j-1]))] = 0
    elif j==0:
      N_iT[i-1,j][np.argmin(abs(N_iT[i-1,j]))] = 0
      N_iT[i-1,j][np.argmin(abs(N_iT[i-1,j]))] = 0
      N_iT[i,j+1][np.argmin(abs(N_iT[i,j+1]))] = 0
    elif j==grid_point-1:
      N_iT[i-1,j][np.argmin(abs(N_iT[i-1,j]))] = 0
      N_iT[i-1,j][np.argmin(abs(N_iT[i-1,j]))] = 0
      N_iT[i,j-1][np.argmin(abs(N_iT[i,j-1]))] = 0
    else:
      #Interior case
      N_iT[i+1,j][np.argmin(abs(N_iT[i+1,j]))] = 0
      N_iT[i-1,j][np.argmin(abs(N_iT[i-1,j]))] = 0
      N_iT[i-1,j][np.argmin(abs(N_iT[i-1,j]))] = 0
      N_iT[i,j-1][np.argmin(abs(N_iT[i,j-1]))] = 0
  return N_iT


def find_speed(v1,v2):
  s = []
  for i in range(len(v1)):
    s.append(find_length(v1[i],v2[i]))
  return s

"Vectorize h function"
def iterate_func(r,p,R,rad,epsilon):
  return h_function(r,p,R)-h_function(r,p,epsilon)

iterate_f = np.vectorize(iterate_func)

def area_occupied(r):
    """Computes total area the pedestrians occupy - not accounting for overlaps"""
    # Sum over i of [ pi * r_i * r_i ]
    return np.pi * np.sum( np.multiply( r, r))

def occupancy(r,bounding_area):
    """Computes occupancy given bounding area - not accounting for overlaps"""

    return area_occupied(r) / bounding_area

def density(n,bounding_area):

    return n / bounding_area