from typing import List, Tuple, Dict, Union, Optional
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import distance

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import MeanShift
from scipy.optimize import shgo

RANDOM_STATE = 42


def compute_sigmoid_local_density(
    x: np.ndarray, dataset_points: np.ndarray, decay_dist: float=0.04
) -> np.ndarray:
    """
    Compute the decay score and apply a sigmoid transformation to obtain a 
    density-like value.

    Parameters:
    - decay_dist (delta): A float representing the characteristic distance at 
    which the decay effect becomes significant. This parameter controls the 
    rate at which the influence of a reference point decreases with distance. 
    Lower values allow for the inclusion of farther points, effectively 
    shifting the sigmoid curve horizontally. Note that for x_half = delta * 
    np.log(2), we have np.exp(-x_half/delta) = 1/2, indicating the distance 
    at which the decay effect reduces to half its initial value.
    
    Note: an efficient decay with sigmoid is at decay_dist = 0.04

    Returns:
    - A float representing the transformed score after applying the decay and 
    sigmoid functions.

    Explanation:
    - Sigmoid Effect: The sigmoid function is applied to the decay score to 
    compress it into a range between 0 and 1. This transformation makes the 
    score resemble a density measure, where values close to 1 indicate a 
    densely populated area.
    - Sigmoid Parameters: The sigmoid position and sigmoid speed are fixed at 5
    and 1, respectively. The sigmoid_pos determines the midpoint of the sigmoid
    curve, while the sigmoid_speed controls its steepness. In our context, these
    parameters do not have significantly different effects compared to
    decay_dist, hence they are fixed values. Also adequate sigmoid parameters
    are very hard to find.
    """
    x = np.atleast_2d(x)

    # Compute for each x_row distances to every dataset point
    # Element at position [i, j] is d(x_i, dataset_points_j)
    distances = distance.cdist(x, dataset_points, metric="euclidean")

    # Compute the decay score weights for all distances
    # decay score =0 for big distance, =1 for small distance
    decay_scores = np.exp(-distances / decay_dist)

    # Sum the decay scores across each row (for each point)
    # At each row the sum is supported mainly by closer points having greater effect
    cumulated_scores = decay_scores.sum(axis=1)

    # Fix sigmoid parameters
    pos = 5
    speed = 1

    # Apply sigmoid to combined scores to get scores in [0, 1]
    transformed_decays = 1 / (1 + np.exp(-(cumulated_scores - pos) * speed))

    return transformed_decays


class KDEModel:
    """
    A cool website to visualize KDE.
    https://mathisonian.github.io/kde/
    """
    def __init__(self, kernel='gaussian'):
        self.kernel = kernel
        self.kde = None
        self.data = None
        self.modes = None
        self.bandwidth = None
        self.min_density = None

    def search_bandwidth(
        self, X, log_bounds=(-4, 0), n_iter=20, cv=5, random_state=RANDOM_STATE
    ):
        """
        TODO: Find a better score than likelihood to select bandwidth.
        
        For example:
        - The optimal bandwidth is the largest possible width for which the
        minimum density over the design space [0, 1]^p is equal to 0. In terms
        of class attributes:
        
                best_bandwidth = max({bandwidth | min_density < tol})
        
        - The optimal bandwidth is the largest possible width for which the
        density at the modes is above a threshold equal to 0.8. Since the
        locations of the modes do not change, it's computationaly more
        convenient to express this in terms of maximum density rather than
        minimum density. Formally:
        
                best_bandwidth = max({bandwidth | density_at_modes > 0.8})
        """
        # Define a range of bandwidths to search over using log scale
        bandwidths = np.logspace(log_bounds[0], log_bounds[1], 100)

        # Use RandomizedSearchCV to find the best bandwidth
        random_search = RandomizedSearchCV(
            KernelDensity(kernel=self.kernel),
            {'bandwidth': bandwidths},
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
            # scoring=None => use KDE.score which is log-likelihood
        )
        random_search.fit(X)

        # Get best bandwidth
        best_bandwidth = random_search.best_params_['bandwidth']
        print(f"KDEModel.search_bandwidth -> Optimal bandwidth found: {best_bandwidth}")
    
        return best_bandwidth

    def fit(self, X: np.ndarray, bandwidth: Optional[float] = None, **kwargs):
        # Store the training data
        self.data = X

        # Use the specified bandwidth or the best one found
        if bandwidth is None:
            bandwidth = self.search_bandwidth(X, **kwargs)
        self.bandwidth = bandwidth
        
        # Fit the KDE model
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth)
        self.kde.fit(X)

    def update_modes(self):
        """
        Update modes of data clusters using Mean shift to later compute maximum
        modes density.
        """
        if self.kde is None:
            raise RuntimeError("Model must be fitted before searching for max density.")
        
        mean_shift = MeanShift(bandwidth=self.bandwidth)
        mean_shift.fit(self.data)
        
        # Get modes (highest density points)
        self.modes = mean_shift.cluster_centers_
    
    def update_min_density(self, shgo_iters: int = 5, shgo_n: int = 1000):
        """
        Update minimum density in [0, 1]^p space using SHGO minimizer to search
        for minimum density (anti-modes minimal density).
        """
        if self.kde is None:
            raise RuntimeError("Model must be fitted before searching for min density.")

        print("KDEModel.update_min_density -> Searching for minimum density")

        result = shgo(
            self.predict_proba, bounds=[(0, 1)]*self.data.shape[1],
            iters=shgo_iters, n=shgo_n, sampling_method='simplicial'
        )
        self.min_density = result.fun

        print(f"KDEModel.update_min_density -> Minimum density: {self.min_density}")

    def predict_proba(self, X):
        if self.kde is None:
            raise RuntimeError("Model must be fitted before predicting.")
        log_density = self.kde.score_samples(X)
        return np.exp(log_density)

    def predict_anti_density_score(self, X):
        """
        Score encourages regions with low density.

        Note: this is not (1 - density_score) as we normalize by different
        quantities: (1 - min_density) != max_density.
        """
        X = np.atleast_2d(X)
        if self.min_density is None:
            raise RuntimeError("min_density must be updated before predicting scores.")
        densities = self.predict_proba(X)
        return (1 - densities) / (1 - self.min_density)


class OutlierExcluder:
    def __init__(
        self, features: List[str], targets: List[str]
    ):
        
        self.features = features
        self.targets = targets
        # Points to be avoided as they result in erroneous simulations
        self.ignored_df = pd.DataFrame(columns=self.features)
    
    def update_outliers_set(self, df: pd.DataFrame):
        """ Add bad points (in feature space) to the set of ignored points. """
        # Identify rows where any target column has NaN
        ignored_rows = df[df[self.targets].isna().any(axis=1)]

        # Extract feature values from these rows
        new_ignored_df = ignored_rows[self.features]

        # Concatenate the new ignored points with the existing ones
        self.ignored_df = pd.concat([self.ignored_df, new_ignored_df]).drop_duplicates()
        self.ignored_df = self.ignored_df.reset_index(drop=True)

    def detect_outlier_proximity(
        self, x: np.ndarray, exclusion_radius: float = 1e-5
    ) -> np.ndarray:
        """
        Proximity Exclusion Condition: Determine if the given point is
        sufficiently distant from any point with erroneous simulations. Points
        located within a specified exclusion radius around known problematic
        points are excluded from further processing.
        
        This condition is necessary because surrogate GP does not update around
        failed samples.
        """
        x = np.atleast_2d(x)
        
        if self.ignored_df.empty:
            return np.zeros(x.shape[0], dtype=bool)
    
        # Compute distances based on the specified metric
        distances = distance.cdist(x, self.ignored_df.values, "euclidean")

        # Determine which points should be ignored based on tolerance
        should_ignore = np.any(distances < exclusion_radius, axis=1)

        # Log ignored points
        for i, ignore in enumerate(should_ignore):
            if ignore:
                print(f"OutlierProximityHandler.should_ignore -> Point {x[i]} was ignored.")

        return should_ignore
