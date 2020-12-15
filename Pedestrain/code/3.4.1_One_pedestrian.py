# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:08:39 2020

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
import Main
import ped_func
import math
import seaborn as sns

#To parallelize the code
from joblib import Parallel, delayed
import multiprocessing
import scipy.spatial.distance as scidist
num_cores = multiprocessing.cpu_count()
import ped_utils as putils
from pf1 import ped_funcs as pf #Assumes that pf1.so has been generated using f2py

# %%
#GNM model simulation
#Set up grid space with obstacle
grid_point = 151
end_point_x = 4
end_point_y = 1.5
X,Y = np.meshgrid(np.linspace(-end_point_x,end_point_x,grid_point), np.linspace(-end_point_y,end_point_y,grid_point),indexing = 'ij')
mesh = np.array(np.meshgrid(np.linspace(-end_point_x,end_point_x,grid_point), np.linspace(-end_point_y,end_point_y,grid_point)))
phi = -1*np.ones_like(X)
phi2 = -1*np.ones_like(X)
phi[np.logical_and(X==end_point_x,abs(Y)<=1.5)] = 1
phi2[np.logical_and(X==-end_point_x,abs(Y)<=1.5)] = 1
dx = X[1][1]-X[0][1]
dy = Y[1][1]-Y[1][0]

def set_init_ped(Total_ped,x1,x2,y1,y2):
  x1 = np.random.uniform(x1,x2,Total_ped)
  y1 = np.random.uniform(y1,y2,Total_ped)
  speed = np.random.uniform(0.9,1.4,Total_ped)
  init_pos,init_speed = [],[]
  for i in range(Total_ped):
    init_pos.append([x1[i],y1[i]])
    init_speed.append(speed[i])
  return init_pos,init_speed

speed = np.ones_like(X)

#speed[Y>0] = 1.5
z = np.sqrt(X**2+Y**2)

mask = np.zeros(phi.shape,type(bool))

phi  = np.ma.MaskedArray(phi, mask)
phi2 = np.ma.MaskedArray(phi2, mask)
t   = skfmm.travel_time(phi, speed,dx=(X[1][1]-X[0][1]).item())
t2 = skfmm.travel_time(phi2, speed,dx=(X[1][1]-X[0][1]).item())

np.random.seed(2)
#Define initial grid space 
#kappa,tau,P_p,P_B,R_p,R_B = 0.6,0.5,3.59,9.96,0.7,0.25
kappa,tau,P_p,P_B,R_p,R_B = 0.6,0.5,1.79,11.3,1,0.25
epsilon = 0.001

occupancy_vals_GNM = [None] * len(ped_num)
avgspeed_vals_GNM = [None] * len(ped_num)
avgvelo_GNM = []
all_pos_GNM = []

initial_pos1 = [[-3,0]]
initial_speed1 = [0]
  
initial_pos2 = []
initial_speed2 = [0]

combinations = mesh.T.reshape(-1,2)
#Accoring to x-axis 
xx,yy = combinations[:,0],combinations[:,1]


#Find gradient of sigma
Grad_sigma = ped_func.find_grad(t,dx,dy,grid_point)
N_iT1 = [[np.array([Grad_sigma[0][i][j],Grad_sigma[1][i][j]]) for j in range(len(Grad_sigma[1]))] for i in range(len(Grad_sigma[0]))]
N_iT1 = np.array(N_iT1)

Grad_sigma_2 = ped_func.find_grad(t2,dx,dy,grid_point)
N_iT2 = [[np.array([Grad_sigma_2[0][i][j],Grad_sigma_2[1][i][j]]) for j in range(len(Grad_sigma_2[1]))] for i in range(len(Grad_sigma_2[0]))]
N_iT2 = np.array(N_iT2)

#Direction vector become horizontal or vertical around obstacle
obstacle_index = ped_func.find_obstacle_boundary_index(t,X,Y)
N_iT1 = ped_func.smooth_boundary_grad(N_iT1,obstacle_index,grid_point)
N_iT2 = ped_func.smooth_boundary_grad(N_iT2,obstacle_index,grid_point)

#Find obstacle boundary points
obstacle_boundary_x,obstacle_boundary_y = ped_func.find_obstacle_boundary(t,X,Y)
obstacle_point = np.stack((obstacle_boundary_x,obstacle_boundary_y),axis=-1)

x1 ,x2 = [],[]
for index,val in enumerate(initial_pos1):
  x1.append(val[0])
  x2.append(val[1])
y1 = np.concatenate((x1,x2,initial_speed1))

x1 ,x2 = [],[]
for index,val in enumerate(initial_pos2):
  x1.append(val[0])
  x2.append(val[1])
y2 = np.concatenate((x1,x2,initial_speed2))

h_func = np.vectorize(ped_func.h_function)
iterate_f = np.vectorize(ped_func.iterate_func)
#Intial condition

n1 = y1.size//3
n2 = y2.size//3
index_1 = np.arange(n1)
index_2 = np.arange(n1,n1+n2)

v1 = np.random.normal(loc=1.34,scale=0.26,size=n1)
v2 = np.random.normal(loc=1.34,scale=0.26,size=n2)
dt = 0.1
radius= 0.195
start_time = time.time()
pos,velo,iteration,exit_index = Main.GNM_model(y1,y2,N_iT1,N_iT2,xx,yy,grid_point,end_point_x,end_point_y,obstacle_point,epsilon,v1,v2,dt,radius,index_1,index_2,initial_pos1,initial_speed1,initial_pos2,initial_speed2,time_limit = 90,obstacle=False)
print("--- %s seconds ---" % (time.time() - start_time))
#%%
initial_speed = np.concatenate((initial_speed1,initial_speed2))
A = v1 - initial_speed1

ww = np.zeros(len(velo)+1)
velocity = np.zeros(len(velo))
x_space = np.linspace(0,len(velo)*dt,len(velo))
N_grad_T = np.array(griddata((xx,yy),N_iT1.reshape(-1,2),(initial_pos1),method='nearest'))
N_vec = np.array(ped_func.g_scale(N_grad_T))
ww[0] = initial_speed[0]
for i in range(len(velo)):
  ww[i+1] = v1*np.linalg.norm(N_vec)-A*np.exp(-(i+1)*dt/tau)
  velocity[i] = ped_func.find_length(ww[i]*N_vec[:,0],ww[i]*N_vec[:,1])
plt.plot(x_space,np.array(velo).ravel(),label='Numerical solution')
velocity = np.ones(len(x_space))*v1
plt.plot(x_space,velocity,label='Desired velocity',marker='o')
plt.grid()
plt.plot(x_space,velocity,label='Analytical solution')
plt.title('Pedestrian velocity until destination')
plt.xlabel('Time:S')
plt.ylabel('Meter / S')
plt.legend(loc='center right')
plt.show()