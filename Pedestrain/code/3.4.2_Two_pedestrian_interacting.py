"""
Yadu Raj Bhageria
Imperial College London
Mathematics Department
CID: 00733164
"""
#------------------------------------------------------------------------------#
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#------------------------------------------------------------------------------#
import model_dev as model
#import model
from importlib import reload
reload(model)
#------------------------------------------------------------------------------#
model.reset_model()
model.instructions = False # Change True to see errors in initializing parameters before running the model
model.n = 2
model.tau = 0.5
model.ar = math.radians(0.1)
model.time_step = 0.1
#Setting person 0 at (-3.5,+0) and person 1 at (0,0)
model.x = np.zeros((2,model.n))
model.x[0][0] = - 7.88/2
model.x[1][0] =   0.01
model.x[0][1] =   7.88/2
model.x[1][1] = - 0.01
#Setting destination of person 0 as (7.88/2,0)
model.o = np.zeros((2,model.n))
model.o[0][0] = 7.88/2
model.o[0][1] = -7.88/2
#150 degree horizon
model.H_min = math.radians(75)
model.H_max = math.radians(75)
model.mass = np.ones(2) * 75 #Person of weight 70
model.v_0 = 1.3*np.ones(model.n)
#Array to hold current information of speeds
model.v = np.zeros((2,model.n))
model.v[0][0] = 1.0
model.v[0][1] = -1.0
#Initialize the walls [a,b,c,startval,endval]
model.n_walls = 2
# wall y = 1.75/2
model.walls = np.empty((5,model.n))
model.walls[:,0] = np.array([ 0, 1, 1.75/2, -7.88/2, 7.88/2])
# wall y = -1.75/2
model.walls[:,1] = np.array([ 0, 1, -1.75/2, -7.88/2, 7.88/2])
#------------------------------------------------------------------------------#
model.check_model_ready()
model.initialize_global_parameters()
#------------------------------------------------------------------------------#
#Increment the time in steps of relaxation_time
while (model.t<6):
    model.compute_destinations()
    model.move_pedestrians()
    model.update_model()
    if model.instructions: print(model.t)
    #break
    
    
def read_files(f):
    header = f.readline();
    column_name = ['time_step','Pedestrain_ID','X_pos','Y_pos']
    datas = pd.DataFrame(columns=column_name)
    values = []
    for ind,row in enumerate(f):
        s = row.split(" ")
        time_steps = int(s[0])
        pedId = int(s[1])
        x_PID = float(s[2])
        y_PID = float(s[3])
        values.append(((time_steps,pedId,x_PID,y_PID)))
    datas = pd.DataFrame(values,columns=column_name)
    return datas


    #------------------------------------------------------------------------------#
# IMPORT DATA
#data = np.genfromtxt('data/data_pub_corridor_2people.csv', delimiter=',')
f = open('data/postvis2ped_0.4.trajectories', "r")
datas = read_files(f)
f.close()
f = open('data/postvis2ped_0.7.trajectories', "r")
datas2 = read_files(f)
f.close()
f = open('data/postvis2ped_1.1.trajectories', "r")
datas3 = read_files(f)
f.close()
f = open('data/postvis2ped_2.trajectories', "r")
datas4 = read_files(f)
fig = plt.figure()
#plt.plot( data1[:,0], data1[:,1], 'r', label = 'Published Results')
for ind,i in enumerate(datas.Pedestrain_ID.unique()):
  ped_data = datas.loc[datas['Pedestrain_ID'] == i].reset_index(drop=True)
  plt.plot(ped_data['X_pos'],ped_data['Y_pos'],label = '$R_p$ = 0.4 ')
  ped_data = datas2.loc[datas2['Pedestrain_ID'] == i].reset_index(drop=True)
  plt.plot(ped_data['X_pos'],ped_data['Y_pos'],label = '$R_p$ = 0.7 ')
  #ped_data = datas3.loc[datas3['Pedestrain_ID'] == i].reset_index(drop=True)
  #plt.plot(ped_data['X_pos'],ped_data['Y_pos'],label = '$R_p$ = 1.1 ')
  ped_data2 = datas4.loc[datas4['Pedestrain_ID'] == i].reset_index(drop=True)
  plt.plot(ped_data2['X_pos'],ped_data2['Y_pos'],label = '$R_p$ = 2')
# plt.plot(x_position[:,ind],y_position[:,ind],'--',label = 'Simulation result Ped %1.3f'%i)
plt.xlim([-7.88/2,7.88/2])
plt.ylim([-1.75/2,1.75/2])
#------------------------------------------------------------------------------#
#plot_pub = plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
plt.plot(model.x_full[0,0,:],model.x_full[1,0,:], linestyle = "--", label = 'P1 ->')
plt.plot(model.x_full[0,1,:],model.x_full[1,1,:], linestyle = "--", label = 'P2 <-')
plt.axhline(1.75/2,color = 'k')
plt.axhline(-1.75/2,color ='k')
plt.title('Pedestrian travel trajectory with different $R_p$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
# model.plot_current_positions(fig, ['blue','red'])
#plt.savefig('results/corridor_2people.png')

f = open('data/postvis2ped_pot_3.59.trajectories', "r")
datas = read_files(f)
f.close()
f = open('data/postvis2ped_pot_5.59.trajectories', "r")
datas2 = read_files(f)
f.close()
f = open('data/postvis2ped_pot_8.59.trajectories', "r")
datas3 = read_files(f)
f.close()
f = open('data/postvis2ped_pot_11.59.trajectories', "r")
datas4 = read_files(f)
fig = plt.figure()
#plt.plot( data1[:,0], data1[:,1], 'r', label = 'Published Results')
for ind,i in enumerate(datas.Pedestrain_ID.unique()):
  ped_data = datas.loc[datas['Pedestrain_ID'] == i].reset_index(drop=True)
  plt.plot(ped_data['X_pos'],ped_data['Y_pos'],label = '$P_p$ = 3.59 ')
  ped_data = datas2.loc[datas2['Pedestrain_ID'] == i].reset_index(drop=True)
  plt.plot(ped_data['X_pos'],ped_data['Y_pos'],label = '$P_p$ = 5.59 ')
  #ped_data = datas3.loc[datas3['Pedestrain_ID'] == i].reset_index(drop=True)
  #plt.plot(ped_data['X_pos'],ped_data['Y_pos'],label = '$P_p$ = 8.59 ')
  ped_data2 = datas4.loc[datas4['Pedestrain_ID'] == i].reset_index(drop=True)
  plt.plot(ped_data2['X_pos'],ped_data2['Y_pos'],label = '$P_p$ = 11.59')
# plt.plot(x_position[:,ind],y_position[:,ind],'--',label = 'Simulation result Ped %1.3f'%i)
plt.xlim([-7.88/2,7.88/2])
plt.ylim([-1.75/2,1.75/2])

#------------------------------------------------------------------------------#
#plot_pub = plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
plt.plot(model.x_full[0,0,:],model.x_full[1,0,:], linestyle = "--", label = 'P1 ->')
plt.plot(model.x_full[0,1,:],model.x_full[1,1,:], linestyle = "--", label = 'P2 <-')
plt.axhline(1.75/2,color = 'k')
plt.axhline(-1.75/2,color ='k')
plt.title('Pedestrian travel trajectory with different $P_p$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
# model.plot_current_positions(fig, ['blue','red'])
#plt.savefig('results/corridor_2people.png')





f.close()
