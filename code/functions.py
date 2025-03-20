import os
import scipy.io as sio
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import cv2


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

def Rz2d(theta):

  c = np.cos(theta)
  s = np.sin(theta)
  return np.array([[c, -s],
                   [s, c]])


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


def estimate_transformation(source, target,n):
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
    transformation = np.eye(n)
    transformation[:-1, :-1] = R
    transformation[:-1, -1] = t

    return transformation


def icp(source, target,transformation,dimension, iterations,tolerance=1e-6):

    prev_error = 0
    # Build KD-tree for target
    target_tree = KDTree(target)
            
    for i in range(iterations):

        # Transform the source point cloud
        homogeneous_source = np.hstack((source, np.ones((len(source), 1))))
        transformed_source = homogeneous_source @ transformation.T
        transformed_source = transformed_source[:, :-1]


        # Query KD-tree for nearest neighbors
        distances, indices = target_tree.query(transformed_source)
        target_new = target[indices,:]
        # 2. Estimate Transformation 
        new_transformation = estimate_transformation(source, target_new,dimension)

        error = np.mean(distances ** 2)  # Root mean square error (RMSE)

        # 5. Check Convergence 
        if np.abs(error - prev_error) < tolerance:
            break

        prev_error = error
        transformation = new_transformation

    return transformation, error 


def create_point_cloud(xcoords,ycoords):
    # m = xcoords.shape[0]
    # n = xcoords.shape[1]
    # # ones = np.ones((m,n))
    point_cloud = np.stack((xcoords,ycoords),axis=-1)
    return point_cloud

# def create_homogeneous(x,y):
#     m = xcoords.shape[0]
#     n = xcoords.shape[1]
#     ones = np.ones((m,n))
#     homogeneous = np.stack((xcoords,ycoords,ones),axis=-1)
#     return homogeneous

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))


def pose2_to_transformation_matrix(pose2):
   
    # Extract translation (x, y) and rotation (theta) from the pose
    x, y, theta = pose2.x(), pose2.y(), pose2.theta()
    
    # Construct the rotation matrix
    transformation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])

    return transformation_matrix