
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result, icp, Rz


yaw_angles = np.arange(0, 2*np.pi, np.pi/20)
rotations = np.array([Rz(yaw) for yaw in yaw_angles])


if __name__ == "__main__":
  obj_name = 'drill' # drill or liq_container
  num_pc = 1 # number of point clouds

  source_pc = read_canonical_model(obj_name)

  for i in range(num_pc):
    target_pc = load_pc(obj_name, i)

    # estimated_pose, you need to estimate the pose with ICP
    pose = np.eye(4)
    source_centroid = np.mean(source_pc, axis=0)
    target_centroid = np.mean(target_pc, axis=0)
    pose[:3, 3] = source_centroid - target_centroid

    best_rmse = float('inf')
    best_transformation = None

    for initial_rotation in rotations: 
      pose[:3, :3] = initial_rotation
      pose,rmse = icp(source_pc, target_pc, pose,50)

      if rmse < best_rmse:
        best_rmse = rmse
        best_transformation = pose

    # visualize the estimated result
    visualize_icp_result(source_pc, target_pc,best_transformation)

