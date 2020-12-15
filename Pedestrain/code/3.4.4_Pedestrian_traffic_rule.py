
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



#Set up grid space with obstacle
grid_point = 151
end_point_x = 7.88/2
end_point_y = 1.75/2
X,Y = np.meshgrid(np.linspace(-end_point_x,end_point_x,grid_point), np.linspace(-end_point_y,end_point_y,grid_point),indexing = 'ij')
mesh = np.array(np.meshgrid(np.linspace(-end_point_x,end_point_x,grid_point), np.linspace(-end_point_y,end_point_y,grid_point)))
phi = -1*np.ones_like(X)
phi2 = -1*np.ones_like(X)
phi[np.logical_and(X==end_point_x,abs(Y)<=1.75/2)] = 1
phi2[np.logical_and(X==-end_point_x,abs(Y)<=1.75/2)] = 1
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

#Set up walls 
mask = np.zeros(phi.shape,type(bool))
mask3 = np.logical_or(Y<0.2,Y>0.5)
mask4 = np.logical_and(mask3,X==0)
mask5 = np.logical_or(Y>-0.2,Y<-0.5)
mask6 = np.logical_and(mask5,X==0)
mask7 = np.logical_and(mask4,mask6)


phi  = np.ma.MaskedArray(phi, mask4)
phi2 = np.ma.MaskedArray(phi2, mask6)
t   = skfmm.travel_time(phi, speed,dx=(X[1][1]-X[0][1]).item())
t2 = skfmm.travel_time(phi2, speed,dx=(X[1][1]-X[0][1]).item())



np.random.seed(3)
#Define initial grid space 
kappa,tau,P_p,P_B,R_p,R_B = 0.6,0.5,3.59,9.96,0.7,0.25
epsilon = 0.001



def heat_plot(t,t2):
    #all_ped = [3,5,9,10,15,20,30]
    all_ped = [30]
    all_pos = []
    exit_ped = []
    
    for Total_ped in all_ped:
    
        initial_pos1,initial_speed1 = set_init_ped(Total_ped,-7.88/2,-3,-1.75/2,1.75/2)
        initial_pos2,initial_speed2 = set_init_ped(Total_ped, 3, 7.88/2,-1.75/2,1.75/2)
    
    
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
        y1 = np.array(x1+x2+initial_speed1)
        
        x1 ,x2 = [],[]
        for index,val in enumerate(initial_pos2):
          x1.append(val[0])
          x2.append(val[1])
        y2 = np.array(x1+x2+initial_speed2)
        
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
        all_pos.append(pos)
        exit_ped.append(exit_index)

    return all_pos,exit_ped

all_pos1,exit_ped1 = heat_plot(t,t2)

phi = -1*np.ones_like(X)
phi2 = -1*np.ones_like(X)
phi[np.logical_and(X==end_point_x,abs(Y)<=0.5)] = 1
phi2[np.logical_and(X==-end_point_x,abs(Y)<=0.5)] = 1
phi  = np.ma.MaskedArray(phi, mask7)
phi2 = np.ma.MaskedArray(phi2, mask7)
t   = skfmm.travel_time(phi, speed,dx=(X[1][1]-X[0][1]).item())
t2 = skfmm.travel_time(phi2, speed,dx=(X[1][1]-X[0][1]).item())

all_pos2,exit_ped2 = heat_plot(t,t2)


#%%
plt.figure()
phi  = np.ma.MaskedArray(phi, mask7)
df_points = pd.DataFrame(columns=['x','y','v'])
for ind,val in enumerate(all_pos2[-1]):  
    n = np.array(val).size//2
    f = val[:n]
    g = val[n:2*n]
    values = np.ones(n)
    new = pd.DataFrame({"x":f, "y":g, "v":values})
    df_points = pd.concat([df_points, new],sort=False)
plt.hist2d(df_points.x, df_points.y, weights=df_points.v, bins=20, cmap="viridis")
plt.colorbar()
plt.contour(X, Y, phi.mask, [0], linewidths=(3), colors='red')
plt.title('Density map without one way rule')
plt.xlim([-7.88/2,7.88/2])
plt.ylim([-1.75/2,1.75/2])


#%%
plt.figure()
df_points = pd.DataFrame(columns=['x','y','v'])
for ind,val in enumerate(all_pos1[-1]):  
    n = np.array(val).size//2
    f = val[:n]
    g = val[n:2*n]
    values = np.ones(n)
    new = pd.DataFrame({"x":f, "y":g, "v":values})
    df_points = pd.concat([df_points, new],sort=False)
plt.hist2d(df_points.x, df_points.y, weights=df_points.v, bins=20, cmap="viridis")
plt.colorbar()
plt.contour(X, Y, phi.mask, [0], linewidths=(3), colors='red')
plt.title('Density map with one way rule')
plt.xlim([-7.88/2,7.88/2])
plt.ylim([-1.75/2,1.75/2])

