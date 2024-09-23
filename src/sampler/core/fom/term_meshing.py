import numpy as np
from scipy.spatial import Delaunay


class NDTDensityEstimator:
    """
    The class uses recursive subdivision of the unit hypercube [0, 1]^p
    space, counting the number of points in each subregion, and then computes the
    density as a function of the sample count in the region that contains the
    input point.
    """
    class Node:
        def __init__(self, bounds):
            self.bounds = bounds      # Region bounds in [0, 1]^p space (list of tuples)
            self.children = None      # To store subregions
            self.points = []          # Points within this region
            self.sample_count = 0     # Number of points in this region

    def __init__(self, max_depth=10, min_samples=1):
        """
        Initialize the density estimator.
        
        Parameters:
        - max_depth: Maximum depth of the tree (controls granularity).
        - min_samples: Minimum number of samples in a node before it is not subdivided.
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def _split_region(self, bounds):
        """
        Split the region into 2^p subregions.
        """
        p = len(bounds)  # Dimensionality of the space
        midpoints = [(low + high) / 2 for low, high in bounds]
        subregions = []
        
        # Generate all combinations of splits across dimensions
        for i in range(2**p):
            new_bounds = []
            for j in range(p):
                if (i >> j) & 1:
                    new_bounds.append((midpoints[j], bounds[j][1]))
                else:
                    new_bounds.append((bounds[j][0], midpoints[j]))
            subregions.append(new_bounds)
        return subregions

    def _point_in_region(self, point, bounds):
        """
        Check if a point lies within a region.
        """
        return all(low <= point[i] <= high for i, (low, high) in enumerate(bounds))

    def _build_tree(self, node, points, depth):
        """
        Recursively build the ndtree.
        """
        node.sample_count = len(points)
        
        # Stop subdividing if max depth is reached or not enough samples
        if depth >= self.max_depth or len(points) <= self.min_samples:
            node.points = points
            return node

        # Split the current region into 2^p subregions
        subregions = self._split_region(node.bounds)
        node.children = []
        
        for region in subregions:
            subregion_points = [p for p in points if self._point_in_region(p, region)]
            child_node = self.Node(region)
            node.children.append(self._build_tree(child_node, subregion_points, depth + 1))

        return node

    def fit(self, X):
        """
        Fit the model to the data points X in [0,1]^p space.
        
        Parameters:
        - X: numpy array of shape (n_samples, p), where n_samples is the number
          of points and p is the dimensionality of the space.
        """
        p = X.shape[1]
        self.root = self.Node([(0, 1)] * p)  # The entire [0,1]^p space
        self._build_tree(self.root, X, 0)

    def _find_region(self, node, x):
        """
        Find the leaf node (region) that contains point x.
        """
        if node.children is None:
            return node  # Leaf node

        for child in node.children:
            if self._point_in_region(x, child.bounds):
                return self._find_region(child, x)

        return node

    def predict_density(self, X):
        """
        Predict a density score for each point in X.
        
        Parameters:
        - X: numpy array of shape (n_samples, p).
        
        Returns:
        - Densities: numpy array of shape (n_samples,) with density scores between 0 and 1.
        """
        densities = np.zeros(X.shape[0])

        # Traverse tree and estimate density for each point
        for i, x in enumerate(X):
            leaf_node = self._find_region(self.root, x)

            # Normalize density by sample count in the region compared to the overall root node
            densities[i] = leaf_node.sample_count / self.root.sample_count

        # Normalize densities between 0 and 1
        if densities.max() > 0:
            densities /= densities.max()

        return densities


class DelaunayDensityEstimator:
    """
    This class estimates the density of a point based on Delaunay triangulation
    of a dataset in [0, 1]^p space. It divides the space into p-dimensional
    simplices (polytopes with (p+1) vertices) and computes the density for any
    input point based on the volume of the simplex it falls into. A larger
    simplex volume indicates a sparser, less dense region, resulting in a lower
    density score. Optionally, the density score can be adjusted by the
    closeness of the point to the simplex vertices, with closer proximity
    indicating higher density, weighted by a tunable `closeness_weight`. The
    final density score is a combination of the inverse simplex volume and this
    closeness measure, normalized between 0 and 1.
    """
    def __init__(self, closeness_weight=0.1):
        """
        Initialize the density estimator.
        
        Parameters:
        - closeness_weight: A factor to weight the importance of the distance to vertices
          in the density score. A smaller value means less importance compared to the simplex volume.
        """
        self.delaunay = None
        self.volumes = None
        self.closeness_weight = closeness_weight

    def _compute_simplex_volumes(self):
        """
        Compute the volume of each simplex in the Delaunay triangulation.
        """
        simplices = self.delaunay.simplices
        points = self.delaunay.points
        
        p = points.shape[1]
        n_simplices = simplices.shape[0]
        volumes = np.zeros(n_simplices)
        
        for i, simplex in enumerate(simplices):
            vertices = points[simplex]
            # Calculate volume of the simplex in p-dimensional space
            matrix = vertices[1:] - vertices[0]
            volume = np.abs(np.linalg.det(matrix)) / np.math.factorial(p)
            volumes[i] = volume
        
        return volumes

    def fit(self, X):
        """
        Fit the model using the input data X in [0,1]^p space.
        
        Parameters:
        - X: numpy array of shape (n_samples, p), where n_samples is the number
          of points and p is the dimensionality of the space.
        """
        self.delaunay = Delaunay(X)
        self.volumes = self._compute_simplex_volumes()

    def _simplex_closeness(self, simplex_vertices, x):
        """
        Compute a closeness measure of point x to the vertices of the simplex.
        The closeness score is the inverse of the average distance from x to the vertices.
        """
        distances = np.linalg.norm(simplex_vertices - x, axis=1)
        avg_distance = np.mean(distances)
        return 1.0 / (avg_distance + 1e-8)  # To avoid division by zero

    def predict_density(self, X):
        """
        Predict a density score for each point in X.
        
        Parameters:
        - X: numpy array of shape (n_samples, p).
        
        Returns:
        - Densities: numpy array of shape (n_samples,) with density scores between 0 and 1.
        """
        densities = np.zeros(X.shape[0])

        # For each point, find the corresponding simplex and compute the density
        for i, x in enumerate(X):
            simplex_index = self.delaunay.find_simplex(x)

            # If the point is outside the convex hull, assign zero density
            if simplex_index == -1:
                densities[i] = 0.0
                continue

            # Get the simplex volume
            simplex_volume = self.volumes[simplex_index]

            # Get the vertices of the simplex
            simplex_vertices = self.delaunay.points[self.delaunay.simplices[simplex_index]]

            # Compute the closeness score to the vertices (optional refinement)
            closeness_score = self._simplex_closeness(simplex_vertices, x)

            # Inverse of volume indicates density; larger volumes = lower density
            density_score = 1.0 / (simplex_volume + 1e-8)  # Prevent division by zero
            
            # Combine volume-based density and closeness score
            total_score = (1.0 - self.closeness_weight) * density_score + \
                          self.closeness_weight * closeness_score
            
            densities[i] = total_score

        # Normalize densities between 0 and 1
        if densities.max() > 0:
            densities /= densities.max()

        return densities
