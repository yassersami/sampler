import numpy as np
import pandas as pd
from typing import Union
from scipy.spatial import Delaunay
from scipy.special import factorial
from scipy.stats import skew, kurtosis
from typing import List, Dict, Tuple, Callable

class ASVD:
    """
    Augmented (Space) Simplex Volume Distribution (ASVD) class.

    This class handles the calculation and analysis of fractional vertex star volumes,
    also known as Voronoi volumes or Donald volumes in some contexts.
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
        self.data = data
        self.use_func = use_func
        self.func = func
        self.compute_asvd()

    def compute_asvd(self):
        # Set vertices
        vertices_x = self.data[self.features].values
        if not self.use_func:
            vertices_y = self.data[self.targets].values
        else:
            vertices_y = np.array([self.func(vertex).ravel() for vertex in vertices_x])
        vertices_xy = np.column_stack((vertices_x, vertices_y))

        # Create Delaunay triangulation
        tri = Delaunay(vertices_x)
        simplices_idx = tri.simplices

        # # Set original simplices coordinates
        # simplices_x = np.array([vertices_x[simplex_idx] for simplex_idx in simplices_idx])
        # # Set augmented simplices coordinates
        # simplices_xy = np.array([vertices_xy[simplex_idx] for simplex_idx in simplices_idx])
        
        # Set original simplices coordinates
        simplices_x = vertices_x[simplices_idx]
        # Set augmented simplices coordinates
        simplices_xy = vertices_xy[simplices_idx]

        # Compute simplex volume
        simplices_volumes_x = compute_simplices_volumes(simplices_x)
        simplices_volumes_xy = compute_simplices_volumes(simplices_xy)

        # Compute fractional vertex star volume
        vertices_idx = np.arange(vertices_x.shape[0])
        df_fvs_volumes_x = compute_fvs_volumes(vertices_idx, simplices_idx, simplices_volumes_x)
        df_fvs_volumes_xy = compute_fvs_volumes(vertices_idx, simplices_idx, simplices_volumes_xy)
        stars_volumes_x = df_fvs_volumes_x.values.ravel()
        stars_volumes_xy = df_fvs_volumes_xy.values.ravel()

        # Set new attributes at the end to avoid self prefixes for readability
        self.simplices_idx = simplices_idx
        self.simplices_x = simplices_x
        self.simplices_xy = simplices_xy
        self.simplices_volumes_x = simplices_volumes_x  # simplex volume
        self.simplices_volumes_xy = simplices_volumes_xy  # augmented simplex volume
        self.stars_volumes_x = stars_volumes_x  # fractional vertex star volume
        self.stars_volumes_xy = stars_volumes_xy  # augmented fractional vertex star volume

    def compute_statistics(self):
        # Fractional Vertex Star Volume scores dicts
        stars_scores_x = compute_asvd_scores(self.stars_volumes_x)
        stars_scores_xy = compute_asvd_scores(self.stars_volumes_xy)
        stars_scores_xy["augmentation"] = self.stars_volumes_xy.sum() / self.stars_volumes_x.sum()

        df_scores = pd.DataFrame({
            'volumes_x': stars_scores_x,
            'volumes_xy': stars_scores_xy,
        })
        # Reorder rows
        df_scores = insert_row_in_order(df_scores, [("sum", 1), ("augmentation", 2)])
    
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


def compute_fvs_volumes(
    vertices_idx: np.ndarray, simplices_idx: np.ndarray, volumes: np.ndarray
):
    """
    Calculate the Fractional Vertex Star Volumes (FVSV) for each vertex in a set of simplices.

    This function computes what is also known as the Voronoi volume or Donald volume
    in some contexts. It allocates a fraction of each simplex's volume to its vertices.

    Parameters:
    vertices_idx (array-like): Array of vertices indices.
    simplices_idx (array-like): Array of simplices, each represented by a list of vertices indices.
    volumes (array-like): Array of volumes corresponding to each simplex.
                          Note that volumes.shape[0] == simplices_idx.shape[0]

    Returns:
    pd.DataFrame: DataFrame containing the fractional vertex star volumes for each given vertex.

    Example:
    >>> vertices_idx = np.array([0, 1, 2, 3])
    >>> simplices_idx = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3]])
    >>> volumes = np.array([1.0, 1.5, 2.0])
    >>> result = compute_fvs_volumes(vertices_idx, simplices_idx, volumes)
    >>> print(result)
                 fractional_volume
    0                     1.000000
    1                     0.833333
    2                     1.500000
    3                     1.166667
    """
    # Number of vertices per simplex
    num_vertices = simplices_idx.shape[1]

    vertex_star_volumes = {vertex: 0.0 for vertex in vertices_idx}

    for simplex, volume in zip(simplices_idx, volumes):
        volume_fraction = volume / num_vertices
        for vertex in simplex:
            vertex_star_volumes[vertex] += volume_fraction

    df_stars_volumes = pd.DataFrame.from_dict(
        vertex_star_volumes, orient='index', columns=['fractional_volume']
    )
    return df_stars_volumes


def compute_asvd_scores(volumes: np.ndarray) -> pd.DataFrame:
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



if __name__ == "__main__":
    # Example usage
    vertices_idx = np.array([0, 1, 2, 3])
    simplices_idx = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3]])
    volumes = np.array([1.0, 1.5, 2.0])

    df_stars_volumes = compute_fvs_volumes(vertices_idx, simplices_idx, volumes)
    print(f'df_stars_volumes: \n{df_stars_volumes}')

    # Example usage
    volumes = np.random.rand(100)  # Example data

    scores = compute_asvd_scores(volumes)
    print(f'scores: \n{scores}')

    # Example of usage
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

    # Initialize ASVD object
    asvd = ASVD(data, features, targets)

    # Compute statistics
    stats = asvd.compute_statistics()
    print(f'stats: \n{stats}')

    # Using a custom function
    def f_expl(points):
        """Example function: f(x, y) = x^2 + y^2"""
        points = np.atleast_2d(points)
        return np.sum(np.square(points), axis=1, keepdims=True)

    asvd_func = ASVD(data, features, targets, use_func=True, func=f_expl)
    stats_func = asvd_func.compute_statistics()
    print(f'stats_func: \n{stats_func}')
