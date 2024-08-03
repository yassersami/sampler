import numpy as np
import pandas as pd
from typing import Union
from scipy.spatial import Delaunay
from scipy.special import factorial
from scipy.stats import skew, kurtosis


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
    each simplex is a list of (p+1) vertices in (p+k)-dimensional space
    """
    volumes = [compute_simplex_volume(simplex) for simplex in simplices]
    volumes = np.array(volumes)
    return volumes


def assvd(f, points):
    # Create Delaunay triangulation
    tri = Delaunay(points)
    simplices_idx = tri.simplices

    # Create simplices_up
    augmented_simplices = np.array([
        [(*point, *f(point).ravel()) for point in points[simplex_idx]]
        for simplex_idx in simplices_idx
    ])
    original_simplices = np.array([
        [point for point in points[simplex_idx]]
        for simplex_idx in simplices_idx
    ])

    # Compute volumes
    augmented_volumes = compute_simplices_volumes(augmented_simplices)
    original_volumes = compute_simplices_volumes(original_simplices)
    return augmented_volumes, original_volumes


def calculate_fvsv(
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
    >>> result = calculate_fvsv(vertices_idx, simplices_idx, volumes)
    >>> print(result)
       fractional_vertex_star_volume
    0                     1.000000
    1                     0.833333
    2                     1.500000
    3                     1.166667
    """
    vertex_star_volumes = {vertex: 0.0 for vertex in vertices_idx}
    
    # Number of vertices per simplex
    num_vertices = simplices_idx.shape[1]
    
    for simplex, volume in zip(simplices_idx, volumes):
        volume_fraction = volume / num_vertices
        for vertex in simplex:
            vertex_star_volumes[vertex] += volume_fraction

    df_vertex_star_volumes = pd.DataFrame.from_dict(vertex_star_volumes, orient='index', columns=['fvsv'])
    return df_vertex_star_volumes


def assvd_scores(augmented_volumes: np.ndarray, original_volumes: np.ndarray) -> pd.DataFrame:
    """
    Calculate various metrics for the volumes of augmented and original simplices.
    """
    # Create a DataFrame for volumes
    df_volumes = pd.DataFrame({
        'augmented': augmented_volumes,
        'original': original_volumes
    })

    # Calculate descriptive statistics
    df_scores = df_volumes.describe()
    # Add metrics
    df_scores.loc["sum", "augmented"] = augmented_volumes.sum()
    df_scores.loc["cv", "augmented"] = augmented_volumes.std() / augmented_volumes.mean()
    df_scores.loc["iqr", "augmented"] = np.percentile(augmented_volumes, 75) - np.percentile(augmented_volumes, 25)
    df_scores.loc["skewness", "augmented"] = skew(augmented_volumes)
    df_scores.loc["kurtosis", "augmented"] = kurtosis(augmented_volumes)
    df_scores.loc["augmentation", "augmented"] = augmented_volumes.sum() / original_volumes.sum()
    
    df_scores.loc["sum", "original"] = original_volumes.sum()
    df_scores.loc["cv", "original"] = original_volumes.std() / original_volumes.mean()
    df_scores.loc["iqr", "original"] = np.percentile(original_volumes, 75) - np.percentile(original_volumes, 25)
    df_scores.loc["skewness", "original"] = skew(original_volumes)
    df_scores.loc["kurtosis", "original"] = kurtosis(original_volumes)
    
    # Reorder rows
    df_scores = insert_row_in_order(df_scores, "sum", 1)

    return df_scores


def insert_row_in_order(
    df: pd.DataFrame, row_name: Union[str, int], position: int
) -> pd.DataFrame:
    """
    Insert a specified row into a DataFrame at a given position.
    """
    if row_name not in df.index:
        raise ValueError(f"Row '{row_name}' not found in DataFrame.")
    
    current_order = df.index.tolist()
    current_order.remove(row_name)
    new_order = current_order[:position] + [row_name] + current_order[position:]
    
    return df.loc[new_order]


if __name__ == "__main__":
    # Example usage
    vertices_idx = np.array([0, 1, 2, 3])
    simplices_idx = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3]])
    volumes = np.array([1.0, 1.5, 2.0])

    df_vertex_star_volumes = calculate_fvsv(vertices_idx, simplices_idx, volumes)
    print(df_vertex_star_volumes)

    # Example usage
    augmented_volumes = np.random.rand(100)  # Example data
    original_volumes = np.random.rand(100)  # Example data

    df_result = assvd_scores(augmented_volumes, original_volumes)
    print(df_result)