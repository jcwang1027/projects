
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
ped_num = [6,12,18,24,30,36,45,60,75]
occupancy_vals_GNM = [None] * len(ped_num)
avgspeed_vals_GNM = [None] * len(ped_num)
avgvelo_GNM = []
all_pos_GNM = []

for indice,n in enumerate(ped_num):
    all_rad = np.ones(n)*62.4/320
    bounding_area = 3. * 8.
    occupancy_vals_GNM[indice] = ped_func.occupancy(all_rad,bounding_area)
    x = np.zeros( ( 2, n))
    x[0,:] = np.linspace(-4,4,n+2)[1:-1]
    x[1][::] = np.random.uniform( -1.25, 1.25, n)
    initial_pos1 = np.stack((x[0],x[1]),axis=-1)
    initial_speed1 = np.zeros(n)*1.3
  
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
    all_pos_GNM.append(pos)
    avgvelo_GNM.append(velo)
    print("--- %s seconds ---" % (time.time() - start_time))

Average_velocity = np.zeros(len(avgvelo_GNM))

for indices,value in enumerate(avgvelo_GNM):
  Average_velocity[indices] = np.average(np.nan_to_num(np.array(avgvelo_GNM[indices])))

densities = ped_func.density(np.array(ped_num),bounding_area)
#%%
#Export Java code result from excel sheets



def java_code(file_name,time_stepsize):
  f = open(file_name, "r")
  header = f.readline();
  column_name = ['time_step','Pedestrain_ID','X_pos','Y_pos']
  data = pd.DataFrame(columns=column_name)
  values = []
  for ind,row in enumerate(f):
      s = row.split(" ")
      time_steps = int(s[0])
      pedId = int(s[1])
      x_PID = float(s[2])
      y_PID = float(s[3])
      values.append(((time_steps,pedId,x_PID,y_PID)))
  data = pd.DataFrame(values,columns=column_name)
  #rows = pd.Series(list((time_steps,pedId,x_PID,y_PID)),index=data.columns)
  #data = data.append(rows, ignore_index=True)

  datas = pd.DataFrame(columns=column_name)
  for ind,i in enumerate(data.Pedestrain_ID.unique()):
    Current_data = data.loc[data['Pedestrain_ID'] == i].reset_index(drop=True)
    Current_data['Velocity']= np.sqrt( Current_data.X_pos.diff()**2+ Current_data.Y_pos.diff()**2)/time_stepsize
    Current_data['Velocity'] = Current_data['Velocity'].fillna(0)
    datas = pd.concat([datas,  Current_data],sort=False)


  data_within = datas
  new_column_name = ['time_step','Density','Velocity','Std']
  new_data = pd.DataFrame(columns=new_column_name)

  rows = []
  #Build up density velocity 
  for ind, i in enumerate(data_within.time_step.unique()):
    Current_data = data_within.loc[data_within['time_step'] == i].reset_index(drop=True)
    rows.append((i,Current_data.shape[0]/24,Current_data['Velocity'].mean(),Current_data['Velocity'].std()))
  new_data = pd.DataFrame(rows, columns=new_column_name)
  #new_data = new_data[:-20]
  #new_data = new_data.loc[(new_data['Density']>=0.9) & (new_data['Density']<=3.5)]
  new_data.Std = new_data.Std.fillna(0)
  #polyfit = np.poly1d(np.polyfit(new_data.Density,new_data.Velocity,3))
  #sorted = np.sort(new_data.Density)
  #plt.scatter(new_data.Density,new_data.Velocity)gi
  #print(new_data)
  #plt.plot(sorted,polyfit(sorted))
  #plt.scatter(new_data.Density,new_data.Std,marker='o')
  #plt.xlabel('Density $Pm^{-2}$')
  #plt.ylabel('Velocity $ms^{-1}$')
  return np.average(new_data.Velocity.loc[new_data.Density == new_data.Density.max()]),new_data.Velocity.loc[new_data.Density == new_data.Density.max()]

Simulation_avg = np.zeros(len(ped_num))
Simulation_speed = []
for ind,val in enumerate(ped_num):
  Simulation_avg[ind],Speed = java_code("data/postvis8m_%1i.trajectories"%val,0.4)
  Simulation_speed.append(Speed)

# %%
from occ_test_par import avgspeed_vals,avg_speed
# %%

plt.figure()
plt.plot(densities,Simulation_avg/1.34,label = 'GNM model Java code ')
plt.plot(densities,np.array(avgspeed_vals)/1.34,label='Cognitive based model')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Density $Pm^{-2}$')
plt.ylabel('Normalized Velocity $Ms^{-1}$')
plt.plot(densities,np.array(Average_velocity)/1.34,label='GNM model simulation result')
plt.title('Relationship of average speed against density')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.grid()

#%%

plt.figure()
GNM_std = []
for indices,value in enumerate(avgvelo_GNM):
    GNM_std.append(np.std((np.nan_to_num(np.array(avgvelo_GNM[indices])[50:])),axis=1))
plt.plot(densities,(np.average(GNM_std,axis=1))/0.26,label='GNM model',marker='x')
GNM_java_std = []
#%%

Social_std = []
for indices,value in enumerate(avg_speed):
    Social_std.append(np.std(np.linalg.norm(avg_speed[indices],axis=0).T,axis=1))
plt.plot(densities,np.average(Social_std,axis=1)/0.26,label='Cognitive based model',marker='o')
plt.title('Relationship of speed variation against density')
plt.xlabel('Density $Pm^{-2}$')
plt.ylabel('Normalized standard deviation $ms^{-1}$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()

#%%
#Relation_data = pd.DataFrame(columns = ['model','Density','Average_speed'])
#
#for indices,value in enumerate(avgvelo_GNM):
#    row = pd.DataFrame({'model':'GNM','Density':densities[indices],'Average_speed':np.average((np.nan_to_num(np.array(avgvelo_GNM[indices])))[50:],axis=1)/1.34})
#    Relation_data = pd.concat([Relation_data,row],sort=False)
#
#
#
#for indices,value in enumerate(avgvelo_GNM):
#    row = pd.DataFrame({'model':'Cognitive','Density':densities[indices],'Average_speed':np.average(np.linalg.norm(avg_speed[indices],axis=0).T[50:],axis=1)/1.34})
#    Relation_data = pd.concat([Relation_data,row],sort=False)
#plt.figure()
#sns.boxplot(x="Density", y="Average_speed", data=Relation_data, hue='model')
#plt.title('Relationship of normalized average speed against density')
