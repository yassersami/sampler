import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from scipy.special import gamma, betainc
from sklearn.cluster import DBSCAN
from numba import jit, prange


# * ---------------- Tools functions ----------------
def near_cube_border(point, radius):
    # Check distance to each face of the cube
    distances = np.array([point[i] for i in range(len(point))])
    distances = np.append(distances, 1 - distances) # Add the distance to the opposite face

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
def volume_n_sphere_cap(radius, a, n):
    if a >= 0:
        volume = (1/2) * np.pi**(n/2) / gamma(n/2 + 1) * radius**n * betainc((n + 1)/2, 1/2, 1 - (a**2 / radius**2))
    else:
        volume = np.pi**(n/2) / gamma(n/2 + 1) * radius**n - volume_n_sphere_cap(radius, -a, n)
    return volume

def compute_two_sphere_intersection_volume(center1, center2, radius, n):
    d = np.linalg.norm(center1 - center2)

    if d >= 2 * radius:
        return 0
    elif d <= abs(radius - radius):
        return volume_n_sphere(radius, n)
    
    c1 = (d**2 + radius**2 - radius**2) / (2 * d)
    c2 = (d**2 - radius**2 + radius**2) / (2 * d)
    
    cap_volume1 = volume_n_sphere_cap(radius, c1, n)
    cap_volume2 = volume_n_sphere_cap(radius, c2, n)
    
    intersection_volume = cap_volume1 + cap_volume2
    return intersection_volume

def volume_n_sphere(radius, n):
    return np.pi**(n/2) * radius**n / gamma(n/2 + 1)

def compute_simple_case_volume(cluster_points, radius, n)-> np.ndarray:
    sphere_volume = volume_n_sphere(radius, n)
    acum_volume = 0
    if len(cluster_points) == 1:
        # One sphere case
        acum_volume += sphere_volume
    elif len(cluster_points) == 2:
        # There is a simple equation for the volume of intersection of two spheres
        intersection = compute_two_sphere_intersection_volume(cluster_points[0], cluster_points[1], radius, n)
        volume = 2 * sphere_volume - intersection
        acum_volume += volume

    return np.array([acum_volume, acum_volume], dtype=np.float64)

# * ---------------- Special clusters ----------------
# Compute all clusters in contact with the border of space ([0,1]^3) or clusters with more than 2 spheres

@jit(nopython=True)
def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2)**2))

@jit(nopython=True)
def custom_min(arr):
    min_val = arr[0]
    for val in arr:
        if val < min_val:
            min_val = val
    return min_val

@jit(nopython=True)
def custom_max(arr):
    max_val = arr[0]
    for val in arr:
        if val > max_val:
            max_val = val
    return max_val

@jit(nopython=True)
def generate_pave_points(min_coords, max_coords):
    dim = len(min_coords)
    n_points = 2**dim
    pave_points = np.zeros((n_points, dim))

    for i in range(n_points):
        for j in range(dim):
            if (i & (1 << j)) != 0: # La puissance de ChatGPT, ça vérifie si le j-ième bit est à 1 et donc de déduire si on prend le min ou le max
                pave_points[i, j] = max_coords[j]
            else:
                pave_points[i, j] = min_coords[j]

    return pave_points

@jit(nopython=True)
def subdivide_pave(min_coords, max_coords):
    dim = len(min_coords)
    mid_coords = (min_coords + max_coords) / 2
    sub_paves = []

    # Generate all combinations of sub-pave corners
    for i in range(2**dim):
        new_min = min_coords.copy()
        new_max = max_coords.copy()
        for j in range(dim):
            if (i // (2**j)) % 2 == 0:
                new_max[j] = mid_coords[j]
            else:
                new_min[j] = mid_coords[j]
        sub_paves.append([(new_min[k], new_max[k]) for k in range(dim)])

    return sub_paves

@jit(nopython=True)
def generate_subdivisions(min_coords, max_coords, n_subdivision):
    dim = len(min_coords)
    subdivisions = []
    step = (max_coords - min_coords) / n_subdivision

    for index in range(n_subdivision**dim):
        coords = []
        for d in range(dim):
            div = n_subdivision**(dim - d - 1)
            coord_index = (index // div) % n_subdivision
            coord_min = min_coords[d] + coord_index * step[d]
            coord_max = coord_min + step[d]
            coords.append((coord_min, coord_max))
        subdivisions.append(coords)
    
    return subdivisions


@jit(nopython=True)
def count_pave_volume(points: np.ndarray, radius: float, pave: List[Tuple[float, float]], depth: int, max_depth: int)-> np.ndarray:
    min_coords = np.array([p[0] for p in pave])
    max_coords = np.array([p[1] for p in pave])
    pave_points = generate_pave_points(min_coords, max_coords)
    
    is_pave_contact_sphere = False
    for point in points:
        distance_pave_point = np.array([distance(point, pave_point) for pave_point in pave_points])
        if custom_max(distance_pave_point) < radius: # if pave is in the sphere
            volume_pave = np.prod(max_coords - min_coords)
            return np.array([volume_pave, volume_pave], dtype=np.float64)
        elif custom_min(distance_pave_point) < radius: # if pave is in contact with the sphere
            is_pave_contact_sphere = True
            if depth > max_depth: 
                return np.array([0, np.prod(max_coords - min_coords)], dtype=np.float64) # only place where there is a difference to compute volume : it gets bounds
    
    volume = np.array([0, 0], dtype=np.float64)
    if not is_pave_contact_sphere:
        return volume
    
    # Subdivide the pave into 2^dim smaller paves
    sub_paves = subdivide_pave(min_coords, max_coords)
    
    for sub_pave in sub_paves:
        volume += count_pave_volume(points, radius, sub_pave, depth + 1, max_depth)
    
    return volume

@jit(nopython=True, parallel=True)
def octree(centers: np.ndarray, radius: float, max_depth: int, dim: int, n_subdivision: int)-> np.ndarray:
    min_coords = np.array([custom_min(centers[:, i]) for i in range(dim)]) - radius
    max_coords = np.array([custom_max(centers[:, i]) for i in range(dim)]) + radius

    # Ensure the hypercube is inside the [0, 1]^dim cube
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, 1)
    
    # Subdivide the hypercube into smaller paves to avoid first recursions
    subdivision = generate_subdivisions(min_coords, max_coords, n_subdivision)
    
    
    volume = np.array([0, 0], dtype=np.float64)
    for pave in prange(len(subdivision)):
        volume += count_pave_volume(centers, radius, subdivision[pave], 1, max_depth)
    
    return volume

# * ---------------- Mains functions ----------------
def compute_cluster_volume(cluster_points: np.ndarray, radius: float, same_sphere_tol: float, max_depth: int, dim: int, n_subdivision: int)-> np.ndarray:
    cluster_points = remove_very_close_points(cluster_points, same_sphere_tol)

    # Check for points near [0,1]^3 cube borders
    cluster_borders = [near_cube_border(point, radius) for point in cluster_points]
    out_of_cube = any([any(point_borders) for point_borders in cluster_borders])
    
    if len(cluster_points) <= 2 and not out_of_cube:
        return compute_simple_case_volume(cluster_points, radius, dim)

    else: # Approximation with octree (subdivising cube recursively)
        volume = octree(cluster_points, radius, max_depth, dim, n_subdivision) 
        return volume

# TODO : Parallelize this function (prange) -> this implies to transform precedents functions to be compatible with numba
def covered_space_bound(points: np.ndarray, radius: float, params_volume: Dict, dim: int)-> np.ndarray:
    same_sphere_tol = params_volume['same_sphere_tol']
    max_depth = params_volume['max_depth']
    n_subdivision = params_volume['n_subdivision']
    
    points = np.unique(points, axis=0)

    # Clustering of touching spheres
    max_distance = 2 * radius
    dbscan = DBSCAN(eps=max_distance, min_samples=1)
    clusters = dbscan.fit_predict(points)
    acum_volume = np.array([0, 0], dtype=np.float64)
    for cluster_id in np.unique(clusters):
        cluster_points = points[clusters == cluster_id]
        acum_volume += compute_cluster_volume(cluster_points, radius, same_sphere_tol, max_depth, dim, n_subdivision)
        
    print(f"Volume calculated. Lower bound : {acum_volume[0]} - Upper bound : {acum_volume[1]}")
    return acum_volume
