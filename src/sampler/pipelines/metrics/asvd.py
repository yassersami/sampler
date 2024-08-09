import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.special import factorial
from scipy.stats import skew, kurtosis
from scipy.integrate import nquad, IntegrationWarning
from typing import Union, List, Dict, Tuple, Callable
import warnings

class ASVD:
    """
    Augmented (Space) Simplex Volume Distribution (ASVD) class.

    - This class handles the calculation and analysis of fractional vertex star volumes,
    also known as Voronoi volumes or Donald volumes in some contexts.
    - _x or _xy suffix indicates wheter original or augmented space
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
        features: List of feature column names
        targets: List of target column names
        use_func: Boolean indicating whether to use a custom function
        func: Custom function to compute targets (if use_func is True)
        """
        self.features = features
        self.targets = targets
        self.curve_volume = {}
        self.curve_augmentation = {}
        self.set_vertices(data, use_func, func)
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

    def set_simplices(self):
        # Create Delaunay triangulation
        tri = Delaunay(self.vertices_x)
        simplices_idx = tri.simplices
        
        # Set original and augmented simplices coordinates
        simplices_x = self.vertices_x[simplices_idx]
        simplices_xy = self.vertices_xy[simplices_idx]
    
        # New attributes
        self.simplices_idx = simplices_idx
        self.simplices_x = simplices_x
        self.simplices_xy = simplices_xy

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

    def compute_scores(self):
        """
        Computes the following metrics for assessing the quality of simplicial complexes in representing response curves:

        1. Volumetric Statistics:
        Computes mean volume per vertex and total volume of the simplicial complex.
        
        2. Augmentation Ratio (AR):
        Quantifies the extent of captured variation by calculating the ratio of total volume augmentation. 
        Higher values indicate better representation. This metric serves as a discrete approximation of 
        the generalized 1D function arc length integral:

        AR = 1/(b-a) * ∫[a to b] √(1 + (dy/dx)²) dx

        3. Relative Standard Deviation (RSD):
        Measures the uniformity of sample distribution, normalized by the mean to eliminate bias 
        from varying simplex counts. Computed for:
        a) Augmented space along the target function curve (RSD_xy)
        b) Original feature space (RSD_x)
        Lower values indicate more uniform distribution.

        4. RSD Augmentation (RSDA):
        Defined as RSDA = RSD_xy - RSD_x
        Negative values indicate improved uniformity on the response curve relative to the feature space.
        Positive values suggest decreased uniformity post-augmentation.

        """
        simplices_sum_x = self.simplices_volumes_xy.sum()
        simplices_sum_xy = self.simplices_volumes_x.sum()
        simplices_mean_x = self.simplices_volumes_x.mean()
        simplices_mean_xy = self.simplices_volumes_xy.mean()
        sum_augm = np.nan if simplices_sum_x==0 else simplices_sum_xy / simplices_sum_x
        rsd_x = self.simplices_volumes_x.std() / simplices_mean_x
        rsd_xy = self.simplices_volumes_xy.std() / simplices_mean_xy
        rsd_augm = np.zeros_like(rsd_x)
        rsd_augm = np.where(rsd_x != 0, rsd_xy / rsd_x, np.nan)
        riqr_x = (np.percentile(self.simplices_volumes_x, 75) - np.percentile(self.simplices_volumes_x, 25))/simplices_mean_x
        riqr_xy = (np.percentile(self.simplices_volumes_xy, 75) - np.percentile(self.simplices_volumes_xy, 25))/simplices_mean_xy
        return {
            "count": self.vertices_x.shape[0],
            "sum_x": simplices_sum_x,
            "sum_xy": simplices_sum_xy,
            "mean_x": self.stars_volumes_x.mean(),  # Mean over number of vertices
            "mean_xy": self.stars_volumes_xy.mean(),  # Mean over number of aumengted vertices
            "sum_augm": sum_augm,
            "rsd_x": rsd_x,
            "rsd_xy": rsd_xy,
            "rsd_augm": rsd_augm,
            "riqr_x": riqr_x,
            "riqr_xy": riqr_xy,
        }

    def compute_statistics(self):
        # Simplex Volume scores dicts
        simplices_scores_x = describe_volumes(self.simplices_volumes_x)
        simplices_scores_xy = describe_volumes(self.simplices_volumes_xy)
        simplices_scores_xy["sum_augm"] = simplices_scores_xy['sum'] / simplices_scores_x['sum']
        simplices_scores_xy["rsd_augm"] = simplices_scores_xy['rsd'] - simplices_scores_x['rsd']
        # Fractional Vertex Star Volume scores dicts
        stars_scores_x = describe_volumes(self.stars_volumes_x)
        stars_scores_xy = describe_volumes(self.stars_volumes_xy)
        stars_scores_xy["sum_augm"] = stars_scores_xy['sum'] / stars_scores_x['sum']
        stars_scores_xy["rsd_augm"] = stars_scores_xy['rsd'] - stars_scores_x['rsd']

        df_scores = pd.DataFrame({
            ('simplices', 'volumes_x'): simplices_scores_x,
            ('simplices', 'volumes_xy'): simplices_scores_xy,
            ('stars', 'volumes_x'): stars_scores_x,
            ('stars', 'volumes_xy'): stars_scores_xy,
        })
        # Reorder rows
        df_scores = insert_row_in_order(df_scores, [("sum_augm", 2), ('rsd_augm', 6)])
    
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


def describe_volumes(volumes: np.ndarray) -> pd.DataFrame:
    """
    Calculate various metrics for the volumes of augmented and original simplices.
    """
    # Calculate descriptive statistics
    df_scores = pd.DataFrame(volumes).describe()

    # Add metrics
    df_scores.loc["sum", 0] = volumes.sum()
    df_scores.loc["rsd", 0] = volumes.std() / volumes.mean()  # relative standard deviation (RSD) or coefficient of variation (CV)
    df_scores.loc["iqr", 0] = np.percentile(volumes, 75) - np.percentile(volumes, 25)  #  interquartile range (IQR)
    df_scores.loc["skewness", 0] = skew(volumes)
    df_scores.loc["kurtosis", 0] = kurtosis(volumes)
    
    df_scores = insert_row_in_order(df_scores, [("sum", 1), ("rsd", 4)])
    dict_scores = df_scores.iloc[:, 0].to_dict()
    return dict_scores


def insert_row_in_order(
    df: pd.DataFrame, 
    rows_and_positions: List[Tuple[Union[str, int], int]]
) -> pd.DataFrame:
    """
    Insert specified rows into a DataFrame at given positions.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    rows_and_positions (List[Tuple[Union[str, int], int]]): A list of tuples, 
        each containing a row name/index and its desired position.
    """
    current_order = df.index.tolist()
    
    # Sort rows_and_positions by position in ascending order
    rows_and_positions.sort(key=lambda x: x[1])
    
    for row_name, position in rows_and_positions:
        if row_name not in df.index:
            raise ValueError(f"Row '{row_name}' not found in DataFrame. \n{df}")
        current_order.remove(row_name)
        position = min(position, len(current_order))  # Ensure position is within bounds
        current_order.insert(position, row_name)
    
    return df.loc[current_order]


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
        warnings.filterwarnings("ignore", category=IntegrationWarning)
        volume, _ = nquad(_integrand, region, opts={
            'limit': n_points,
            'epsabs': error_tolerance,
            'epsrel': error_tolerance
        })
    # Compute the augmentation ratio
    region_volume = np.prod([r[1] - r[0] for r in region])
    augmentation = volume / region_volume
    
    return augmentation


if __name__ == "__main__":
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
