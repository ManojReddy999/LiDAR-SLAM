import numpy as np
import matplotlib.pyplot as plt
from functions import bresenham2D, sigmoid, create_point_cloud
from Scan_Matching import length, pc, T_list, x_icp,x_state
from load_data import dataset,lidar_ranges, lidar_angle_min, lidar_angle_max, lidar_angle_increment

x_state = x_state

# Map parameters
map_resolution = 0.05  # meters per cell
map_width = 50  # meters
map_height = 50  # meters
map_origin_x = -20  # meters
map_origin_y = -20  # meters

# Initialize map
map_sizex = int(map_width / map_resolution)
map_sizey = int(map_height / map_resolution)
occupancy_grid = np.zeros((map_sizex, map_sizey), dtype=np.int8)  # 0 represents unknown


for i in range(length):

    angle = np.arange(lidar_angle_min,lidar_angle_max+lidar_angle_increment[0,0]-1e-4,lidar_angle_increment[0,0])
    lidar_range = lidar_ranges[:,i]
    indices = np.logical_and((lidar_range < 30), (lidar_range > 0.1))
    lidar_range = lidar_range[indices]
    angle = angle[indices]

    xcoords = 0.15 + lidar_range * np.cos(angle)
    ycoords = lidar_range * np.sin(angle)
    pc = create_point_cloud(xcoords,ycoords)

    homogeneous_points = np.hstack((pc, np.ones((len(pc), 1))))
    transformed_points = homogeneous_points @ T_list[i].T
    transformed_points = transformed_points[:, :2]
    robot_pose = T_list[i][:2,2]
    start_cell = np.array([robot_pose[0]+20, robot_pose[1]+20]) / map_resolution
    transformed_points = (transformed_points + 20)/map_resolution

    for end_cell in transformed_points:
        
    # Use Bresenham2D to get cells between robot position and point
        line_cells = bresenham2D(start_cell[0], start_cell[1], end_cell[0], end_cell[1])
        indGood = np.logical_and((line_cells[0] < map_sizex), (line_cells[1] < map_sizey))
        line_cells = line_cells[:,indGood]

    # Mark cells as free or occupied
        free_cells = (line_cells[0][:-1].astype(int), line_cells[1][:-1].astype(int))
        occupancy_grid[free_cells] = -np.log(9)
        occupancy_grid[int(line_cells[0,-1]), int(line_cells[1,-1])] = np.log(9)  # Occupied

occupancy_grid = sigmoid(occupancy_grid)

#plot map
fig = plt.figure()
# plt.clf()
plt.plot((x_icp[:,0]+20)/map_resolution,(x_icp[:,1]+20)/map_resolution)
plt.imshow(occupancy_grid.T,origin='lower',cmap="viridis");
plt.title("Occupancy grid map")
plt.savefig(f'occupancy_grid_{dataset}.png')