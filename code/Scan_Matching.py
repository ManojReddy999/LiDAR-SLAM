import numpy as np
import matplotlib.pyplot as plt
from load_data import encoder_counts, encoder_stamps, imu_angular_velocity, imu_stamps, lidar_angle_increment,lidar_angle_max,lidar_angle_min,lidar_range_max,lidar_range_min,lidar_ranges,lidar_stamsp,dataset
from functions import create_point_cloud, Rz2d, icp

mpt = 0.0022   # meters per tic
dist = mpt*np.mean(encoder_counts,axis=0)

encoder_stamps_prev = np.zeros(encoder_stamps.shape)
encoder_stamps_prev[1:,] = encoder_stamps[:-1,]
dt_encoder_stamps = encoder_stamps - encoder_stamps_prev

vel = dist/dt_encoder_stamps

closest_indices = []
for i in encoder_stamps:
    diff = np.abs(imu_stamps-i)
    closest_indices.append(np.argmin(diff))

ang_vel = imu_angular_velocity[2]
ang_vel = np.array([ang_vel[i] for i in closest_indices])

x_init = np.array([0,0,0]).astype(float)
x_state = np.zeros((len(encoder_counts[0]),3),dtype=float)
x_state[0,:] = x_init
for i in range(len(encoder_counts[0])-1):
    x_init += dt_encoder_stamps[i+1,]*np.array([vel[i+1,]*np.cos(x_init[2]),vel[i+1,]*np.sin(x_init[2]),ang_vel[i+1,]])
    x_state[i+1,:] = x_init


angles = np.arange(lidar_angle_min,lidar_angle_max+lidar_angle_increment[0,0]-1e-4,lidar_angle_increment[0,0])
angles = np.tile(angles.reshape(1081,1), (1,lidar_ranges.shape[1]))

xcoords = 0.15 + lidar_ranges * np.cos(angles)
ycoords = lidar_ranges * np.sin(angles)
pc = create_point_cloud(xcoords,ycoords)

length = min(x_state.shape[0],pc.shape[1])   # number of iterations

x_icp = np.zeros((length,2),dtype=float)
T = np.eye(3)
matrix_size = (3,3)
T_list = np.empty((length,*matrix_size))
T_list[0] = T

T_icp = np.empty((length-1,*matrix_size))

for i in range(length-1):
    source_pc = pc[:,i+1,:]
    target_pc = pc[:,i,:]
    T0 = np.eye(3)
    T1 = np.eye(3)
    T0[:2, -1] = x_state[i][:2]
    T1[:2, -1] = x_state[i+1][:2]
    T0[:-1, :-1] = Rz2d(x_state[i][2])
    T1[:-1, :-1] = Rz2d(x_state[i+1][2])
    T0_inv = np.linalg.inv(T0)
    T01 = T0_inv @ T1
    T01_icp,_ = icp(source_pc, target_pc, T01,3,50)
    T_icp[i] = T01_icp
    # T = T@T01_icp

    diff = T01[:-1,:-1] - T01_icp[:-1,:-1]
    if np.linalg.norm(diff,'fro') < 0.005:
        T = T@T01_icp
    else:
        T = T@T01
    
    T_list[i+1] = T
    x_icp[i+1,:] = T[:-1,-1]

fig1 = plt.figure()
plt.plot(x_icp[:,0],x_icp[:,1])
plt.plot(x_state[:,0],x_state[:,1])
plt.savefig(f'Scan_matching_{dataset}.png')