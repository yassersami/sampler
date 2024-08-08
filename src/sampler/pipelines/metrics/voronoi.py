import numpy as np
from scipy.spatial import ConvexHull, Voronoi, Delaunay
from numba import njit
from scipy.optimize import linprog
import itertools
import tqdm as tqdm

@njit(fastmath=True)
def intersect_hypercube(point1, point2, dim, tol):
    intersections = np.empty((2*dim, dim), dtype=np.float64)
    count = 0
    for i in range(dim):
        diff = point2[i] - point1[i]
        if abs(diff) > tol: # Not parallel to the hypercube face
            for bound in (0, 1): 
                t = (bound - point1[i]) / diff 
                if 0-tol < t < 1+tol:
                    intersection = point1 + t * (point2 - point1)
                    if np.all(intersection > -tol) and np.all(intersection < 1 + tol):
                        intersections[count] = intersection
                        count += 1
    return intersections[:count]

@njit
def is_arr_include(arr_check, arr_ref):
    for array in arr_check:
        found = False
        for item in arr_ref:
            if np.array_equal(array, item):
                found = True
                break
        if not found:
            return False
    return True

@njit(fastmath=True)
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)
@njit
def filter(points, threshold, dim):
    num_points = len(points)
    filtered_points = np.empty((num_points, dim), dtype=np.float64)
    count = 0
    threshold_power = -np.log10(threshold)
    factor = 10 ** threshold_power

    for i in range(num_points):
        point = np.floor(points[i] * factor + 0.5) / factor
        add_point = True
        for j in range(count):
            if distance(point, filtered_points[j]) < threshold:
                add_point = False
                break
        if add_point:
            filtered_points[count] = point
            count += 1

    return filtered_points[:count]

def get_volume_voronoi(points: np.ndarray, dim: int, tol: float=1e-3, isFilter: bool=True):
    '''
    Compute the volume of the Voronoi diagram of a set of points in a hypercube of dimension dim.
    The volume is computed by clipping the Voronoi regions with the hypercube and computing the volume of the clipped regions with the ConvexHull method.

    Parameters
    ----------
    points : np.array The points to compute the Voronoi diagram.
    dim : int The dimension of the hypercube. (parameter redundant with the dimension of the points)
    tol : float, optional The tolerance to handle numerical errors.
    isFilter : bool, the function will filter the points to remove duplicates and close points (recommended for dim>3).

    See notebooks to understand more about the method and the parameters (and to improve it !).
    '''
    if dim>4: 
        print("The dimension is too high, the computation may take a long time !")

    # Add far points to assure interesting regions are not infinite
    ref_far_points = np.array([
        np.array([(i >> j) & 1 for j in range(dim)]) for i in range(2 ** dim)
    ], dtype=float)

    scaled_reference = 100 * np.sqrt(dim)
    ref_far_points *= scaled_reference
    ref_far_points -= scaled_reference / 2

    extended_points = np.vstack([points, ref_far_points])
    vor_extended = Voronoi(extended_points) # Get the Voronoi diagram

    hypercube_points = np.array([ # Get the hypercube points
        np.array([(i >> j) & 1 for j in range(dim)]) for i in range(2 ** dim)
    ], dtype=float)

    vertices_regions_inside = [] # List of vertices defining the regions inside the hypercube
    vertices_inside = np.array([v for v in vor_extended.vertices if np.all(v > 0 - tol) and np.all(v < 1 + tol)], dtype=np.float64)

    for reg in tqdm.tqdm(vor_extended.regions): 
        if -1 in reg or len(reg) == 0: # if region is infinite or empty, ignore it
            continue

        polytope = np.array([vor_extended.vertices[i] for i in reg], dtype=np.float64)

        if is_arr_include(polytope, vertices_inside): # if the region is inside the hypercube doesn't need to compute intersection (clip)
            vertices_regions_inside.append(polytope)
            continue

        if isFilter: # advice to use it when the number of points is high : round values & remove close points
            polytope = filter(polytope, tol, dim)


        # * Add points of the hypercube inside the polytope (region) if within the region
        delaunay = Delaunay(polytope)
        for p in hypercube_points:
            if delaunay.find_simplex(p)>=0:
                polytope = np.vstack([polytope, p])
                hypercube_points = np.delete(hypercube_points, np.where((hypercube_points == p).all(axis=1)), axis=0)
        # ? Other method, maybe faster/efficient ?
        # num_vertices = len(polytope)
        # c = np.zeros(num_vertices)
        # A = np.vstack((polytope.T, np.ones(num_vertices)))
        # for p in hypercube_points:
        #     b = np.append(p, 1)
        #     res = linprog(c, A_eq=A, b_eq=b, bounds=(0, None))
        #     if res.success and np.all(res.x >= 0):
        #         polytope = np.vstack([polytope, p])

        all_intersections = np.empty((0, dim), dtype=np.float64)

        # TODO : Upgrade this part to compute smartly the intersection between the hypercube and the polytope.
        # ?      I tried to use only ridge_vertices but either I did a mistake or it's not enough to compute the intersection ! 
        # ?      Set small number of points with dim > 2 and look the difference between plot according to the method used
        # polytope_convexHull = ConvexHull(polytope)
        # polytope_ridge_vertices = polytope_convexHull.simplices
        # for ridge_index in polytope_ridge_vertices:
        #     point1, point2 = polytope[ridge_index[0]], polytope[ridge_index[1]]
        #     inter_points = intersect_hypercube(point1, point2, dim, tol)
        #     if inter_points.size > 0:
        #         all_intersections = np.vstack((all_intersections, inter_points))

        for ridge in vor_extended.ridge_vertices: 
            valid_ridge = np.array([v for v in ridge if v != -1 and v in reg]) # Get the valid ridge vertices in the region
            if len(valid_ridge) > 1:
                for simplex in itertools.combinations(valid_ridge, 2): # Make all combinations of the valid ridge vertices (expensive !)
                    point1, point2 = vor_extended.vertices[simplex[0]], vor_extended.vertices[simplex[1]]
                    inter_points = intersect_hypercube(point1, point2, dim, tol)
                    if inter_points.size > 0:
                        all_intersections = np.vstack((all_intersections, inter_points))
   

        polytope = polytope[(polytope > 0 - tol).all(axis=1) & (polytope < 1 + tol).all(axis=1)] # Keep only points inside the hypercube
        if len(all_intersections) > 0:
            pol = np.vstack([polytope, np.vstack(all_intersections)])
        else:
            pol = np.vstack([polytope])
        vertices_regions_inside.append(pol)

    if isFilter:
        for i, region in enumerate(vertices_regions_inside):
            vertices_regions_inside[i] = filter(region, tol, dim)


    l_volume = np.array([ConvexHull(reg, qhull_options='Q12 Qc Qs').volume for reg in vertices_regions_inside]) # Add some parameters to compute volume and avoid errors
    print("Quantity of error volume : ", 1-np.sum(l_volume) )
    return l_volume

