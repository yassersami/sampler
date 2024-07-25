
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

from sklearn.cluster import DBSCAN
from numba import jit, prange



# * ---------------- Tools functions ----------------
def near_cube_border(point, radius):
    # Check distance to each face of the cube
    distances = np.array([
        point[0], 1 - point[0],  # Distance to the left / right face (x = 0 or x = 1)
        point[1], 1 - point[1],  # Distance to the back / front face (y = 0 or y = 1)
        point[2], 1 - point[2]   # Distance to the bottom / top face (z = 0 or z = 1)
    ])
    
    # Check if any distance is less than the radius
    near_faces = distances < radius

    return near_faces 

def remove_very_close_points(cluster_points, same_sphere_tol=1e-3):
    if len(cluster_points) == 1:
        return cluster_points
    
    remaining_points = [cluster_points[0]]

    # Remove points closer than same_sphere_tol
    for i, point in enumerate(cluster_points[1:]):
        distances = np.linalg.norm(remaining_points - point, axis=1)
        if np.all(distances > same_sphere_tol):
            remaining_points.append(point)

    return np.array(remaining_points)



# * ---------------- Simples clusters ----------------
def compute_two_sphere_intersection_volume(center1, center2, radius):
    d = np.linalg.norm(center1 - center2)

    h = 2 * radius - d
    volume = np.pi * h ** 2 * (radius - h / 3)
    return volume

def compute_simple_case_volume(cluster_points, radius, out_of_cube):
    sphere_volume = (4 / 3) * np.pi * radius ** 3
    acum_volume = 0
    if len(cluster_points) == 1:
        # One sphere case
        acum_volume += sphere_volume
    elif len(cluster_points) == 2:
        # There is a simple equation for the volume of intersection of two spheres
        intersection = compute_two_sphere_intersection_volume(cluster_points[0], cluster_points[1], radius)
        volume = 2 * sphere_volume - intersection
        acum_volume += volume

    return acum_volume



# * ---------------- Special clusters ----------------
# Compute all clusters in contact with the border of space ([0,1]^3) or clusters with more than 2 spheres


@jit(nopython=True)
def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

@jit(nopython=True)
def count_pave_volume(points: np.ndarray, radius: float, pave: List[tuple], depth: int, max_depth: int) -> float:

    min_x, max_x = pave[0]
    min_y, max_y = pave[1]
    min_z, max_z = pave[2]
    pave_points = np.array([
        [min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z], [min_x, max_y, max_z],
        [max_x, min_y, min_z], [max_x, min_y, max_z], [max_x, max_y, min_z], [max_x, max_y, max_z]
    ])
    
    is_pave_contact_sphere = False
    for point in points:
        distance_pave_point = np.array([distance(point, pave_point) for pave_point in pave_points])
        if np.max(distance_pave_point) < radius: # if pave is in the sphere
            return (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        elif np.min(distance_pave_point) < radius: # elif pave is in contact with the sphere
            is_pave_contact_sphere = True
            if depth > max_depth:
                return (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

    volume = 0
    if not is_pave_contact_sphere:
        return volume

    # Subdivide the pave into 8 smaller paves
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    mid_z = (min_z + max_z) / 2
    l_subdivision = np.array([
        ((min_x, mid_x), (min_y, mid_y), (min_z, mid_z)),
        ((min_x, mid_x), (min_y, mid_y), (mid_z, max_z)),
        ((min_x, mid_x), (mid_y, max_y), (min_z, mid_z)),
        ((min_x, mid_x), (mid_y, max_y), (mid_z, max_z)),
        ((mid_x, max_x), (min_y, mid_y), (min_z, mid_z)),
        ((mid_x, max_x), (min_y, mid_y), (mid_z, max_z)),
        ((mid_x, max_x), (mid_y, max_y), (min_z, mid_z)),
        ((mid_x, max_x), (mid_y, max_y), (mid_z, max_z))
    ])
    
    for sub_pave in l_subdivision:
        volume += count_pave_volume(points, radius, sub_pave, depth + 1, max_depth)
    
    return volume

@jit(nopython=True, parallel=True)
def octree(centers: np.ndarray, radius: float, max_depth: int) -> float:
    X_centers = np.array([points[0] for points in centers])
    Y_centers = np.array([points[1] for points in centers])
    Z_centers = np.array([points[2] for points in centers])
    # Create the box where the cluster is
    min_x, max_x = np.min(X_centers) - radius, np.max(X_centers) + radius
    min_y, max_y = np.min(Y_centers) - radius, np.max(Y_centers) + radius
    min_z, max_z = np.min(Z_centers) - radius, np.max(Z_centers) + radius

    # Ensure the cube is inside the [0, 1]^3 cube
    min_x, max_x = max(min_x, 0), min(max_x, 1)
    min_y, max_y = max(min_y, 0), min(max_y, 1)
    min_z, max_z = max(min_z, 0), min(max_z, 1)

    # Subdivide the cube into smaller paves to avoid firsts recursions
    n_subdivision = 128 # per side
    x_segment = (max_x - min_x) / n_subdivision
    y_segment = (max_y - min_y) / n_subdivision
    z_segment = (max_z - min_z) / n_subdivision

    def f_point_X(i):
        e = min_x + i * x_segment
        return (e, e + x_segment)
    
    def f_point_Y(i):
        e = min_y + i * y_segment
        return (e, e + y_segment)
    
    def f_point_Z(i):
        e = min_z + i * z_segment
        return (e, e + z_segment)

    l_subdivision = np.array([
        (f_point_X(i), f_point_Y(j), f_point_Z(k)) 
        for i in range(n_subdivision) 
        for j in range(n_subdivision) 
        for k in range(n_subdivision)
    ])
    
    volume = 0
    for pave in prange(len(l_subdivision)):
        volume += count_pave_volume(centers, radius, l_subdivision[pave], 1, max_depth)
    
    return volume


# * ---------------- Mains functions ----------------
def compute_cluster_volume(cluster_points: np.ndarray, radius: float, same_sphere_tol: float, max_depth: int) -> float:
    cluster_points = remove_very_close_points(cluster_points, same_sphere_tol)

    # Check for points near [0,1]^3 cube borders
    cluster_borders = [near_cube_border(point, radius) for point in cluster_points]
    out_of_cube = any([any(point_borders) for point_borders in cluster_borders])
    
    if len(cluster_points) <= 2 and not out_of_cube:
        return compute_simple_case_volume(cluster_points, radius, out_of_cube)

    else: # Approximation with octree (subdivising cube recursively)
        volume = octree(cluster_points, radius, max_depth) 
        return volume

# TODO : Parallelize this function (prange) -> this implies to transform precedents functions to be compatible with numba
def covered_space_upper(points: np.ndarray, radius: float, params_volume: Dict):
    same_sphere_tol = params_volume['same_sphere_tol']
    max_depth = params_volume['max_depth']
    
    points = np.unique(points, axis=0)

    # Clustering of touching spheres
    max_distance = 2 * radius
    dbscan = DBSCAN(eps=max_distance, min_samples=1)
    clusters = dbscan.fit_predict(points)

    print("Calculating volume...")

    acum_volume = 0
    for cluster_id in tqdm(np.unique(clusters)):
        cluster_points = points[clusters == cluster_id]
        acum_volume += compute_cluster_volume(cluster_points, radius, same_sphere_tol, max_depth)
        
    print(f"Volume calculated : {acum_volume}")
    return acum_volume
