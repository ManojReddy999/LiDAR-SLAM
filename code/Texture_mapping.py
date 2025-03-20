import matplotlib.pyplot as plt
import numpy as np
import cv2
from load_data import rgb_stamps, encoder_stamps,disp_stamps
from run_project import dataset
from Scan_Matching import T_list

pose_indices = []
disp_indices = []
for i in rgb_stamps:
    diff1 = np.abs(encoder_stamps-i)
    diff2 = np.abs(disp_stamps-i)
    pose_indices.append(np.argmin(diff1))
    disp_indices.append(np.argmin(diff2))

# Map parameters
map_resolution = 0.05  # meters per cell
map_width = 50  # meters
map_height = 50  # meters
map_origin_x = -20  # meters
map_origin_y = -20  # meters

# Initialize map
map_sizex = int(map_width / map_resolution)
map_sizey = int(map_height / map_resolution)
texture_grid = np.zeros((map_sizex, map_sizey, 3), dtype=np.uint8)

disp_path = "/home/mmkr/Desktop/ECE276A_PR2/dataRGBD/Disparity20/"
rgb_path = "/home/mmkr/Desktop/ECE276A_PR2/dataRGBD/RGB20/"

def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)

for i in range(len(rgb_stamps)):  
    # load RGBD image
    imd = cv2.imread(f"{disp_path}disparity20_{disp_indices[i]}.png",cv2.IMREAD_UNCHANGED) # (480 x 640)
    imc = cv2.imread(f"{rgb_path}rgb20_{i+1}.png")[...,::-1] # (480 x 640 x 3)

    # print(i)

    # convert from disparity from uint16 to double
    disparity = imd.astype(np.float32)

    # get depth
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd

    # calculate u and v coordinates 
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
    #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

    # get 3D coordinates 
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z

    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    rgbv = np.round((v * 526.37 + 16662.0)/fy)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])
    rgbu_flat = rgbu[valid]
    rgbv_flat = rgbv[valid]

    x = x[valid]
    y = y[valid]
    z = z[valid]

    x,y,z = z,-x,-y

    regular_frame = np.stack((x,y,z,np.ones_like(z)),axis=-1)

    T_camera = np.array([[ 0.93569047, -0.02099846,  0.35219656, 0.18],
        [ 0.01965239,  0.99977951,  0.00739722, 0.005],
        [-0.35227423,  0.        ,  0.93589682, 0.36],
        [0.,  0.        ,  0., 1]])

    body_frame = regular_frame @ T_camera.T
    body_frame_hom = body_frame[:, [0, 1, 3]]
    world_frame = body_frame_hom @ T_list[pose_indices[i]].T
    z_world = body_frame[:,2]

    xy_world = world_frame[:,:2]

    z_indices = z_world < -0.15
    rgbu_flat = rgbu_flat[z_indices]
    rgbv_flat = rgbv_flat[z_indices]

    xy_world = (xy_world[z_indices] + 20)/map_resolution
    z_world = (z_world[z_indices] + 20)/map_resolution

    goodInd = np.logical_and((xy_world[:,0] < map_sizex), (xy_world[:,1] < map_sizey))
    xy_world = xy_world[goodInd]

    texture_grid[xy_world[:,0].astype(int),xy_world[:,1].astype(int),:] = imc[rgbv_flat[goodInd].astype(int),rgbu_flat[goodInd].astype(int),:]

fig = plt.figure()
# plt.clf()
plt.imshow(texture_grid.transpose(1,0,2),origin='lower',cmap="viridis");
plt.savefig(f'texture_grid_{dataset}.png')