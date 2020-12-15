
"""
Created on Sun Aug 30 12:25:26 2020

@author: Jiachun Wang
"""

import numpy as np
import skfmm
#pip install scikit-fmm
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage.morphology import binary_dilation
from IPython import display
from scipy.spatial.distance import cdist
import time
import pandas as pd
import ped_func


def GNM_model(y1,y2,N_iT1,N_iT2,xx,yy,grid_point,end_point_x,end_point_y,obstacle_point,epsilon,v1,v2,dt,radius,ped_ind1,ped_ind2,initial_pos1,initial_speed1,initial_pos2,initial_speed2,time_limit=5.5,obstacle=True,kappa=0.6,tau=0.5,P_p=3.59,P_B=9.96,R_p=0.7,R_B=0.25):


  #Store location of pedestrian in the order of x_coord,y_coord,speed
  pos_alltime = []
  #Store velocity each time
  velocity_alltime = []
  #Number of pedestrian exited the grid
  exit_ind = 0
  
  current_time = 0
  #Iteration time step
  time_step = 0
  while current_time <= time_limit:
    #Size of group 1 pedestrians
    n1 = y1.size//3
    #Group 1 pedestrian x_coord
    x_pos1 = y1[:n1]
    #Group 1 pedestrian y_coord
    y_pos1 = y1[n1:2*n1]
    #Group 1 pedestrian speed
    w1 = y1[2*n1:3*n1]
    
    #Size of group 2 pedestrians
    n2 = y2.size//3
    #Group 2 pedestrian x_coord
    x_pos2 = y2[:n2]
    #Group 2 pedestrian y_coord
    y_pos2 = y2[n2:2*n2]
    #Group 2 pedestrian speed
    w2 = y2[2*n2:3*n2]
    
    #Merge position to one array
    y = np.concatenate( (x_pos1,x_pos2,y_pos1,y_pos2) )
    current_time += dt
    time_step += 1
    #Save their position in list
    pos_alltime.append(y.tolist())

    #Approximate direction vector N_iT
    N_grad_T1 = np.array(griddata((xx,yy),N_iT1.reshape(-1,2),(x_pos1,y_pos1),method='nearest'))
    N_grad_T2 = np.array(griddata((xx,yy),N_iT2.reshape(-1,2),(x_pos2,y_pos2),method='nearest'))

    if N_grad_T1.size != 0:

      #Remove the pedestrians that reached destination
      N_grad_T1,y1,v1,exit_ind,ped_ind1 = ped_func.remove_0vec_pe(N_grad_T1,y1,w1,v1,n1,initial_pos1,initial_speed1,exit_ind,ped_ind1,end_point_x,end_point_y,xx,yy,N_iT1)
    n1 = y1.size//3
    x_pos1 = y1[:n1]
    y_pos1 = y1[n1:2*n1]
    w1 = y1[2*n1:3*n1]
    pos1 = np.stack((x_pos1,y_pos1),axis=-1)

    if N_grad_T2.size != 0:
   
      #Remove the pedestrians that reached destination
      N_grad_T2,y2,v2,exit_ind,ped_ind2 = ped_func.remove_0vec_pe(N_grad_T2,y2,w2,v2,n2,initial_pos2,initial_speed2,exit_ind,ped_ind2,end_point_x,end_point_y,xx,yy,N_iT2)
    n2 = y2.size//3
    x_pos2 = y2[:n2]
    y_pos2 = y2[n2:2*n2]
    w2 = y2[2*n2:3*n2]
    pos2 = np.stack((x_pos2,y_pos2),axis=-1)
    
    #Update the position and speed array
    x_pos = np.concatenate((x_pos1,x_pos2))
    y_pos = np.concatenate((y_pos1,y_pos2))
    w = np.concatenate((w1,w2))
    v = np.concatenate((v1,v2))
    pos = np.concatenate((pos1,pos2))


    #Combine the N_iT vector for the two groups
    if (N_grad_T1.size)==0 and (N_grad_T2.size) ==0:
      break
    if (N_grad_T1.size)!=0 and (N_grad_T2.size) !=0:
      N_grad_T = np. concatenate((N_grad_T1,N_grad_T2))
    elif (N_grad_T1.size)!=0:
      N_grad_T = N_grad_T1
    else:
      N_grad_T = N_grad_T2
    
    #Find angle between Gradient N vector and vector from x_i to x_j
    if N_grad_T1.size != 0:
      s_ij1 = np.array([[ped_func.g_tilda(np.cos(kappa*ped_func.find_angle(N_grad_T1[i],ped_func.find_vec(x_pos[j],y_pos[j])-ped_func.find_vec(x_pos1[i],y_pos1[i])) )) for j in range(n1+n2)]   for i in range(n1)])
    
    if N_grad_T2.size != 0:
      s_ij2 = np.array([[ped_func.g_tilda(np.cos(kappa*ped_func.find_angle(N_grad_T2[i],ped_func.find_vec(x_pos[j],y_pos[j])-ped_func.find_vec(x_pos2[i],y_pos2[i])) )) for j in range(n1+n2)]   for i in range(n2)])
    #Combine the angle array for the two groups 
    if (N_grad_T1.size)!=0 and (N_grad_T2.size) !=0:
      s_ij = np.concatenate((s_ij1,s_ij2))
    elif (N_grad_T1.size)!=0:
      s_ij = s_ij1
    else:
      s_ij = s_ij2
    
    #Get N_iP vector
    dist1 = ped_func.cdist(pos,pos)
    pp1 =  ped_func.iterate_f(dist1,P_p,R_p,2*radius,epsilon)*s_ij
    diff_unit1 = (np.array(pos)[:,np.newaxis,:]-np.array(pos)[np.newaxis,:,:]) / abs(dist1[:,:,np.newaxis])
    p_ij = pp1[:,:,np.newaxis]*(diff_unit1)
    # Calculate the forces caused by obstable or without any obstacle, combine N_iT and N_iP
    if obstacle:
      dist2 = ped_func.cdist(pos,obstacle_point)
      pp2 =  ped_func.iterate_f(dist2,P_B,R_B,radius,epsilon)
      diff_unit2 = (np.array(pos)[:,np.newaxis,:]-np.array(obstacle_point)[np.newaxis,:,:]) / abs(dist2[:,:,np.newaxis])
      B_ij = pp2[:,:,np.newaxis]*(diff_unit2)


      N_grad_P = (np.sum(B_ij,axis=1)+ped_func.sum_pij(p_ij))
      N_vec = np.array([ped_func.g_scale(ped_func.g_scale(N_grad_T[i])+ped_func.g_scale(N_grad_P[i])) for i in range(n1+n2)])
    else:
      N_grad_P =ped_func.sum_pij(p_ij)
      N_vec = np.array([ped_func.g_scale(ped_func.g_scale(N_grad_T[i])+ped_func.g_scale(N_grad_P[i])) for i in range(n1+n2)])
      
    #Update the model one time step forward
    D1 = x_pos1+w1*N_vec[:n1,0]*dt
    D2 = y_pos1+w1*N_vec[:n1,1]*dt
    D3 = w1+1/tau*(v1*np.linalg.norm(N_vec[:n1,:],axis=1)-w1)*dt
    velocity_alltime.append(ped_func.find_speed(w*N_vec[:,0],w*N_vec[:,1]))
    y1 = np.concatenate((D1,D2,D3))

    D1_2 = x_pos2+w2*N_vec[n1:,0]*dt
    D2_2 = y_pos2+w2*N_vec[n1:,1]*dt
    D3_2 = w2+1/tau*(v2*np.linalg.norm(N_vec[n1:,:],axis=1)-w2)*dt
    y2 = np.concatenate((D1_2,D2_2,D3_2))

    if N_grad_T1.size != 0:

      #Remove the pedestrians that reached destination
      N_grad_T1,y1,v1,exit_ind,ped_ind1 = ped_func.remove_0vec_pe(N_grad_T1,y1,w1,v1,n1,initial_pos1,initial_speed1,exit_ind,ped_ind1,end_point_x,end_point_y,xx,yy,N_iT1)
      n1 = y1.size//3
      x_pos1 = y1[:n1]
      y_pos1 = y1[n1:2*n1]
      w1 = y1[2*n1:3*n1]
      pos1 = np.stack((x_pos1,y_pos1),axis=-1)

    if N_grad_T2.size != 0:
      #Remove the pedestrians that reached destination
      N_grad_T2,y2,v2,exit_ind,ped_ind2 = ped_func.remove_0vec_pe(N_grad_T2,y2,w2,v2,n2,initial_pos2,initial_speed2,exit_ind,ped_ind2,end_point_x,end_point_y,xx,yy,N_iT2)
      n2 = y2.size//3
      x_pos2 = y2[:n2]
      y_pos2 = y2[n2:2*n2]
      w2 = y2[2*n2:3*n2]
      pos2 = np.stack((x_pos2,y_pos2),axis=-1)
    

  return pos_alltime,velocity_alltime,time_step,exit_ind
