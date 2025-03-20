import os
import scipy.io as sio
import numpy as np
import open3d as o3d


def read_canonical_model(model_name):
  '''
  Read canonical model from .mat file
  model_name: str, 'drill' or 'liq_container'
  return: numpy array, (N, 3)
  '''
  model_fname = os.path.join('./data', model_name, 'model.mat')
  model = sio.loadmat(model_fname)

  cano_pc = model['Mdata'].T / 1000.0 # convert to meter

  return cano_pc


def load_pc(model_name, id):
  '''
  Load point cloud from .npy file
  model_name: str, 'drill' or 'liq_container'
  id: int, point cloud id
  return: numpy array, (N, 3)
  '''
  pc_fname = os.path.join('./data', model_name, '%d.npy' % id)
  pc = np.load(pc_fname)

  return pc


def visualize_icp_result(source_pc, target_pc, pose):
  '''
  Visualize the result of ICP
  source_pc: numpy array, (N, 3)
  target_pc: numpy array, (N, 3)
  pose: SE(4) numpy array, (4, 4)
  '''
  source_pcd = o3d.geometry.PointCloud()
  source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
  source_pcd.paint_uniform_color([0, 0, 1])

  target_pcd = o3d.geometry.PointCloud()
  target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
  target_pcd.paint_uniform_color([1, 0, 0])

  source_pcd.transform(pose)

  o3d.visualization.draw_geometries([source_pcd, target_pcd])



def Rz(theta):

  c = np.cos(theta)
  s = np.sin(theta)
  return np.array([[c, -s, 0],
                   [s, c, 0],
                   [0, 0, 1]])


# def find_correspondences(source, target, transformation):
#     # Transform the source point cloud
#     transformed_source = transform_point_cloud(source, transformation)

#     # Find nearest neighbors in the target for each source point (brute-force)
#     differences = target[:, None, :3] - transformed_source[:, :3]
#     squared_distances = np.sum(differences ** 2, axis=-1)
#     distances = np.sqrt(squared_distances)
#     indices = np.argmin(distances, axis=1)
#     distances = distances[np.arange(len(source)), indices]  # Keep only the minimum distances

#     return distances, indices

from scipy.spatial import KDTree


def estimate_transformation(source, target):
    # Calculate centroids
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)

    # Center the point clouds
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    # Compute the covariance matrix
    Q = sum(np.outer(mi, zi) for mi, zi in zip(target_centered, source_centered))

    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(Q)

    # Compute rotation matrix
    R = U @ Vt

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # Compute translation vector
    t = target_centroid - (R @ source_centroid[:, np.newaxis]).T

    # Construct the transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    return transformation


def icp(source, target,transformation, iterations,tolerance=1e-6):

    prev_error = 0
    # Build KD-tree for target
    target_tree = KDTree(target)
            
    for i in range(iterations):

        # Transform the source point cloud
        transformed_source = (transformation[:3,:3]@source.T).T + transformation[:3,3]   # R = transformation[:3,:3]   p = transformation[:3,3]


        # Query KD-tree for nearest neighbors
        distances, indices = target_tree.query(transformed_source)
        target_new = target[indices,:]
        # 2. Estimate Transformation 
        new_transformation = estimate_transformation(source, target_new)

        error = np.mean(distances ** 2)  # Root mean square error (RMSE)

        # 5. Check Convergence 
        if np.abs(error - prev_error) < tolerance:
            break

        prev_error = error
        transformation = new_transformation

    return transformation, error 

