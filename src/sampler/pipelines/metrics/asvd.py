from typing import Union, List, Dict, Tuple, Callable
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from scipy.special import factorial
from scipy.integrate import nquad, IntegrationWarning
from scipy.stats import lognorm


class ASVD:
    """
    Augmented (Space) Simplex Volume Distribution (ASVD) class.

    This class calculates and analyzes fractional vertex star volumes in both
    original and augmented spaces. These volumes are also known as Voronoi
    volumes or Donald volumes in certain contexts.

    The class performs the following steps:
    1. Applies DBSCAN clustering to the input data in the original feature space.
    2. Conducts Delaunay triangulation on each identified cluster.
    3. Computes simplex volumes.
    4. Computes fractional vertex star volumes.

    Attributes:
        features (List[str]): Names of feature columns.
        targets (List[str]): Names of target columns.
        vertices_x (np.ndarray): Vertex coordinates in the original feature space.
        vertices_xy (np.ndarray): Vertex coordinates in the augmented space (features + targets).
        cluster_labels (np.ndarray): Cluster label for each vertex.
        simplices_idx (np.ndarray): Indices of simplices from Delaunay triangulation.
        simplices_x (np.ndarray): Simplex coordinates in the original feature space.
        simplices_xy (np.ndarray): Simplex coordinates in the augmented space.
        simplices_volumes_x (np.ndarray): Simplex volumes in the original feature space.
        simplices_volumes_xy (np.ndarray): Simplex volumes in the augmented space.
        stars_volumes_x (np.ndarray): Fractional vertex star volumes in the original feature space.
        stars_volumes_xy (np.ndarray): Fractional vertex star volumes in the augmented space.

    Note:
        - Suffix '_x' denotes attributes in the original feature space.
        - Suffix '_xy' denotes attributes in the augmented space (features + targets).
        - Steps 3. and 4. are computed in both the original feature space and
          the augmented space (features + targets).
    """

    def __init__(
        self, data: pd.DataFrame, features: List[str], targets: List[str],
        use_func: bool=False, func: Callable=None
    ):
        """
        Initialize the ASVD object.

        Parameters:
        data: Dataframe (n, p+k) (or (n, p) if use_func is True) of samples that will
            be future (augmented) vertices.
            vertices = data[features], augmented_vertices = data[features + targets]
        use_func: Boolean indicating whether to use a custom function
        func: Custom function to compute targets (if use_func is True)
        """
        self.features = features
        self.targets = targets
        self.set_vertices(data, use_func, func)

        # Check if there are enough vertices for Delaunay triangulation
        if data.shape[0] < len(self.features) + 1:
            # Not enough vertices, set volumes to zero
            self.simplices_volumes_x = np.array([0])
            self.simplices_volumes_xy = np.array([0])
            self.stars_volumes_x = np.array([0])
            self.stars_volumes_xy = np.array([0])
        else:
            # Proceed with normal computation
            self.set_clusters()
            self.set_simplices()
            self.compute_simplices_volumes()
            self.compute_stars_volumes()

    def set_vertices(self, data, use_func, func):
        # Set vertices
        vertices_x = data[self.features].values
        if not use_func:
            vertices_y = data[self.targets].values
        else:
            vertices_y = np.array([func(vertex).ravel() for vertex in vertices_x])
        vertices_xy = np.column_stack((vertices_x, vertices_y))
        # New attributes
        self.vertices_x = vertices_x
        self.vertices_xy = vertices_xy

    def set_clusters(self):
        # Automatically determine DBSCAN parameters
        n_neighbors = min(len(self.vertices_x) - 1, 10)  # Use 10 neighbors or less if fewer points
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(self.vertices_x)
        distances, _ = nbrs.kneighbors(self.vertices_x)
        
        # Sort distances to the nth neighbor (farthest neighbor)
        distances = np.sort(distances[:, -1])
        
        # Find the elbow point for epsilon
        kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
        epsilon = distances[kneedle.elbow] if kneedle.elbow else np.median(distances)

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=epsilon, min_samples=3).fit(self.vertices_x)
        labels = clustering.labels_

        # Store clustering information as attributes
        self.cluster_labels = labels
        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.cluster_sizes = np.bincount(labels[labels >= 0])

    def set_simplices(self):
        # Initialize lists to store simplices and their indices
        all_simplices_idx = []
        all_simplices_x = []
        all_simplices_xy = []

        # Process each cluster
        for cluster_id in range(self.n_clusters):
            # Get indices of points in this cluster
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            cluster_points = self.vertices_x[cluster_indices]

            # Skip clusters with too few points for triangulation
            if len(cluster_points) < len(self.features) + 1:
                continue

            # Create Delaunay triangulation for this cluster
            tri = Delaunay(cluster_points)
            
            # Map local indices to global indices
            global_simplices_idx = cluster_indices[tri.simplices]

            # Add to the lists
            all_simplices_idx.append(global_simplices_idx)
            all_simplices_x.append(self.vertices_x[global_simplices_idx])
            all_simplices_xy.append(self.vertices_xy[global_simplices_idx])

        # Combine all simplices
        self.simplices_idx = np.vstack(all_simplices_idx) if all_simplices_idx else np.array([])
        self.simplices_x = np.vstack(all_simplices_x) if all_simplices_x else np.array([])
        self.simplices_xy = np.vstack(all_simplices_xy) if all_simplices_xy else np.array([])

    def compute_simplices_volumes(self):
        # Compute original and augmented simplex volume
        simplices_volumes_x = compute_simplices_volumes(self.simplices_x)
        simplices_volumes_xy = compute_simplices_volumes(self.simplices_xy)

        # New attributes
        self.simplices_volumes_x = simplices_volumes_x
        self.simplices_volumes_xy = simplices_volumes_xy

    def compute_stars_volumes(self):
        # Set vertices for which to do computation
        vertices_idx = np.arange(self.vertices_x.shape[0])

        # Compute original and augmented fractional vertex star volume
        df_fvs_volumes_x = compute_stars_volumes(
            vertices_idx, self.simplices_idx, self.simplices_volumes_x
        )
        df_fvs_volumes_xy = compute_stars_volumes(
            vertices_idx, self.simplices_idx, self.simplices_volumes_xy
        )
        stars_volumes_x = df_fvs_volumes_x.values.ravel()
        stars_volumes_xy = df_fvs_volumes_xy.values.ravel()

        # New attributes
        self.stars_volumes_x = stars_volumes_x  # fractional vertex star volume
        self.stars_volumes_xy = stars_volumes_xy  # augmented fractional vertex star volume

    def get_augmentation(self, use_star: bool):
        """ Compute augmentation ratio of volumes """
        if use_star:
            # Focus on fractional star volumes
            volumes = self.stars_volumes_x
            volumes_xy = self.stars_volumes_xy
        else: 
            # Use simplex volumes
            volumes = self.simplices_volumes_x
            volumes_xy = self.simplices_volumes_xy

        # Compute total volumes
        sum_volumes = volumes.sum()
        sum_volumes_xy = volumes_xy.sum()

        augmentation = 1 if sum_volumes==0 else sum_volumes_xy / sum_volumes
        return augmentation

    def get_scores(self, use_star: bool):
        """
        Computes the following metrics for assessing the quality of simplicial
        complexes in representing response curves:

        1. Augmentation Ratio:
        Quantifies the extent of captured variation by calculating the ratio of
        total volume augmentation. Higher values indicate better representation.
        This metric serves as a discrete approximation of the generalized 1D
        function arc length integral:

                    AR = 1/(b-a) * ∫[a to b] √(1 + (dy/dx)²) dx

        2. Distribution Characteristics:
        - 'Lognormal σ': Sigma parameter of the fitted log-normal distribution
                         to volumes.
        - 'Std Dev': Standard deviation of volumes.

        3. Volume key statistics at 75th and 90th percentiles.
        """
        if use_star:
            # Focus on fractional star volumes
            volumes = self.stars_volumes_x
            volumes_xy = self.stars_volumes_xy
        else: 
            # Use simplex volumes
            volumes = self.simplices_volumes_x
            volumes_xy = self.simplices_volumes_xy

        # Stats for volumes from 0 to 75th and 90th percentiles
        q3, cumsum_q3 = get_cum_vol(volumes, 75)
        d9, cumsum_d9 = get_cum_vol(volumes, 90)

        # Get log normal distribution parameters
        lognormal_sigma = fit_lognormal(volumes)

        star_scores = {
            'Count': volumes.shape[0],
            'Clusters': self.n_clusters,
            'Augmentat°': self.get_augmentation(use_star),
            'Lognormal σ': lognormal_sigma,
            'Std Dev': np.std(volumes),
            'sum': volumes.sum(),
            '3rd Quartile': q3,
            'Q3 Cum. Vol': cumsum_q3,
            '9th Decile': d9,
            'D9 Cum. Vol': cumsum_d9,
        }
        return star_scores

    def get_statistics(self):
        # Simplex Volume scores dicts
        simplices_scores_x = describe_volumes(self.simplices_volumes_x)
        simplices_scores_xy = describe_volumes(self.simplices_volumes_xy)
        simplices_scores_xy['sum_augm'] = self.get_augmentation(use_star=False)

        # Fractional Vertex Star Volume scores dicts
        stars_scores_x = describe_volumes(self.stars_volumes_x)
        stars_scores_xy = describe_volumes(self.stars_volumes_xy)
        stars_scores_xy['sum_augm'] = self.get_augmentation(use_star=True)

        df_scores = pd.DataFrame({
            ('simplices', 'volumes_x'): simplices_scores_x,
            ('simplices', 'volumes_xy'): simplices_scores_xy,
            ('stars', 'volumes_x'): stars_scores_x,
            ('stars', 'volumes_xy'): stars_scores_xy,
        })

        return df_scores


def compute_simplex_volume(simplex):
    """
    Calculate the volume of a simplex in n-dimensional space.
    
    :param simplex: numpy array of shape (p+1, p+k) where rows represent the vertices of the
    simplex and columns their coordinates.
    :return: volume of the simplex
    """
    p_plus_1, p_plus_k = simplex.shape
    
    # Translate simplex so that the first vertex is at the origin
    translated_simplex = simplex[1:] - simplex[0]
    
    # Compute the Gram matrix
    gram_matrix = np.dot(translated_simplex, translated_simplex.T)
    
    # Compute the volume using the square root of the determinant of the Gram matrix
    volume = np.sqrt(np.linalg.det(gram_matrix)) / factorial(p_plus_1 - 1)

    if np.isnan(volume):
        warnings.warn(f"NaN volume detected for simplex: \n{simplex}")
        return 0
    
    return volume


def compute_simplices_volumes(simplices):
    """
    Compute the volumes of (p+1)-simplices in (p+k)-dimensional space.
    
    simplices: array of shape (n_simplices, p+1, p+k). List of simplices, where
    each simplex is a list of (p+1) vertices with (p+k) coordinates.
    """
    volumes = [compute_simplex_volume(simplex) for simplex in simplices]
    volumes = np.array(volumes)
    return volumes


def compute_stars_volumes(
    vertices_idx: np.ndarray, simplices_idx: np.ndarray, volumes: np.ndarray
):
    """
    Calculate the Fractional Vertex Star Volumes (FVSV) for each vertex in a set of simplices.

    This function computes what is also known as the Voronoi volume or Donald volume
    in some contexts. It allocates a fraction of each simplex's volume to its vertices.

    Note: There is a known issue where the assigned star volumes can vary depending on
    the layout of the simplices generated by the Delaunay function. This can result in
    inconsistent volumes for the same vertex neighborhood.

    Parameters:
    vertices_idx (array-like): Array of vertices indices.
    simplices_idx (array-like): Array of simplices, each represented by a list of vertices indices.
    volumes (array-like): Array of volumes corresponding to each simplex.
                          Note that volumes.shape[0] == simplices_idx.shape[0]

    Returns:
    pd.DataFrame: DataFrame containing the fractional vertex star volumes for each given vertex.
    """
    vertex_star_volumes = {vertex: 0.0 for vertex in vertices_idx}

    # Distribute 1/(p+1) simplex volume to each vertex
    fraction = 1 / simplices_idx.shape[1]  # p+1 vertices per simplex
    for simplex, volume in zip(simplices_idx, volumes):
        for vertex in simplex:
            vertex_star_volumes[vertex] += volume * fraction

    df_stars_volumes = pd.DataFrame.from_dict(
        vertex_star_volumes, orient='index', columns=['fractional_volume']
    )
    return df_stars_volumes


def compute_response_curve_augmentation(
    predictor: Callable[[np.ndarray], np.ndarray],
    region: List[Tuple[float, float]],
    n_points: int = 1000,
    error_tolerance: float = 1e-3
) -> Tuple[float, float]:
    """
    Compute the p-volume of the response curve in (p+k)-dimensional space and the augmentation ratio.

    Parameters:
    predictor (Callable): A callable function that takes a numpy array of shape (n_samples, n_features)
                          and returns a numpy array of shape (n_samples, n_targets).
    p, k (int): Size of features and targets dimensions respectively.
    region (list of tuples): List of (min, max) tuples for each feature dimension.
                             Defaults to [0, 1] for each dimension if not provided.
    n_points (int): Number of points for progress bar estimation. Default is 1000.

    Returns:
    Tuple[float, float]: The computed p-volume of the response curve and the augmentation ratio.
    """
    p = len(region)
    k = predictor(np.array(region)[:, [0]].T).size

    def _integrand(*args):
        """
        Compute the integrand for volume calculation.
        """
        x = np.array(args).reshape(1, -1)
        y = predictor(x).flatten()
        
        # Compute the Jacobian matrix
        jacobian = np.zeros((k, p))
        epsilon = 1e-6
        for i in range(p):
            x_plus = x.copy()
            x_plus[0, i] += epsilon
            y_plus = predictor(x_plus).flatten()
            jacobian[:, i] = (y_plus - y) / epsilon
        
        # Compute the Gram determinant
        gram_matrix = np.dot(jacobian.T, jacobian)
        return np.sqrt(np.linalg.det(np.eye(p) + gram_matrix))

    # Compute the volume using numerical integration with custom options
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=IntegrationWarning)
        volume, _ = nquad(_integrand, region, opts={
            'limit': n_points,
            'epsabs': error_tolerance,
            'epsrel': error_tolerance
        })
    # Compute the augmentation ratio
    region_volume = np.prod([r[1] - r[0] for r in region])
    augmentation = volume / region_volume
    
    return augmentation


def get_cum_vol(volumes: np.ndarray, percentile: int):
    """ Get percentile of volumes and cumulated volume up to percentile. """
    p_vol = np.percentile(volumes, percentile)
    volumes_to_p = volumes[volumes <= p_vol]
    cumulated_volume_to_p = np.sum(volumes_to_p)
    return p_vol, cumulated_volume_to_p


def describe_volumes(volumes: np.ndarray) -> dict:
    """
    Calculate various metrics for the volumes of augmented and original simplices.
    """
    # Key Percentiles
    q3, q3_cumsum = get_cum_vol(volumes, 75)
    d9, d9_cumsum = get_cum_vol(volumes, 90)

    # Get log normal distribution shape
    lognormal_sigma = fit_lognormal(volumes)

    # Compile results
    dict_scores = {
        'count': len(volumes),
        'mean': np.mean(volumes),
        'std': np.std(volumes),
        'lognormal σ': lognormal_sigma,
        'min': np.min(volumes),
        '25%': np.percentile(volumes, 25),
        '50%': np.percentile(volumes, 50),  # median
        '75%': q3,
        '90%': d9,
        'max': np.max(volumes),
        'q3_cumsum': q3_cumsum,
        'd9_cumsum': d9_cumsum,
        'sum': np.sum(volumes),
    }

    return dict_scores

def fit_lognormal(data):
    """
    Fit a lognormal distribution to the given data.

    The location parameter is fixed at 0 (floc=0), which constrains the
    distribution to start at 0. This is often appropriate for data that cannot
    be negative, such as volumes.
    
    Note:
    - shape: Also known as the log-scale parameter (sigma)
    - location: Fixed at 0 in this case
    - scale: Related to the median of the distribution
    """
    # Check if any negative values
    if np.any(data < 0):
        raise ValueError("Negative values not allowed for lognormal distribution with floc=0")
    # Shift data slightly away from 0
    epsilon = 1e-10
    data = np.maximum(data, epsilon)
    shape, loc, scale = lognorm.fit(data, floc=0)
    return shape


if __name__ == '__main__':
    # Example usage: compute_stars_volumes
    vertices_idx = np.array([0, 1, 2, 3])
    simplices_idx = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3]])
    volumes = np.array([1.0, 1.5, 2.0])

    df_stars_volumes = compute_stars_volumes(vertices_idx, simplices_idx, volumes)
    print(f'df_stars_volumes: \n{df_stars_volumes}')

    # Example of usage: ASVD
    np.random.seed(42)
    data = pd.DataFrame({
        'x1': np.random.rand(100),
        'x2': np.random.rand(100),
        'x3': np.random.rand(100),
        'y1': np.random.rand(100),
        'y2': np.random.rand(100)
    })
    features = ['x1', 'x2', 'x3']
    targets = ['y1', 'y2']

    # Using only data
    asvd_data = ASVD(data, features, targets)
    df_stats_data = asvd_data.compute_statistics()
    print(f'df_stats_data: \n{df_stats_data}')

    # Using a custom function
    def f_expl(points):
        """Example function: f(x, y) = x^2 + y^2"""
        points = np.atleast_2d(points)
        return np.sum(np.square(points), axis=1, keepdims=True)

    asvd_func = ASVD(data, features, targets, use_func=True, func=f_expl)
    df_stats_func = asvd_func.compute_statistics()
    print(f'df_stats_func: \n{df_stats_func}')
    
    # Example of usage: compute_response_curve_augmentation
    from sklearn.linear_model import LinearRegression
    
    # Create a simple linear model
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([[1, 2], [2, 4], [3, 6]])
    model = LinearRegression().fit(X, y)
    
    # Compute response curve volume
    volume = compute_response_curve_augmentation(
        predictor=model.predict,
        region=[(0, 3), (1, 4)]
    )
    
    print(f"The response curve volume is: {volume}")
