from typing import List, Tuple, Dict, Any, Literal, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import distance

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import MeanShift
from scipy.optimize import shgo

from .term_base import FittableFOMTerm, ModelFOMTerm
from .term_gpr import SurrogateGPRTerm

RANDOM_STATE = 42


class SigmoidLocalDensityTerm(FittableFOMTerm):
    """
    A fittable FOM term that computes the sigmoid local density.
    """
    fit_config: ClassVar[Dict[str, bool]] = {'X_only': True, 'drop_nan': False}

    def __init__(self, decay_dist: float = 0.04):
        self.decay_dist = decay_dist
        self.dataset_points = None
    
    def fit(self, X: np.ndarray) -> None:
        self.dataset_points = X

    def _predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decay score and apply a sigmoid transformation to obtain a 
        density-like value.

        Parameters:
        - decay_dist (delta): A float representing the characteristic distance
        at which the decay effect becomes significant. This parameter controls
        the rate at which the influence of a reference point decreases with
        distance. Lower values allow for the inclusion of farther points,
        effectively shifting the sigmoid curve horizontally. Note that for
        x_half = delta * np.log(2), we have np.exp(-x_half/delta) = 1/2,
        indicating the distance at which the decay effect reduces to half its
        initial value.
        
        Note: an efficient decay with sigmoid is at decay_dist = 0.04

        Returns:
        - A float representing the transformed score after applying the decay
        and sigmoid functions.

        Explanation:
        - Sigmoid Effect: The sigmoid function is applied to the decay score to
        compress it into a range between 0 and 1. This transformation makes the
        score resemble a density measure, where values close to 1 indicate a
        densely populated area.
        - Sigmoid Parameters: The sigmoid position and sigmoid speed are fixed
        at 5 and 1, respectively. The sigmoid_pos determines the midpoint of the
        sigmoid curve, while the sigmoid_speed controls its steepness. In our
        context, these parameters do not have significantly different effects
        compared to decay_dist, hence they are fixed values. Also adequate
        sigmoid parameters are very hard to find.
        """
        X = np.atleast_2d(X)

        # Compute for each x_row distances to every dataset point
        # Element at position [i, j] is d(x_i, dataset_points_j)
        distances = distance.cdist(X, self.dataset_points, metric='euclidean')

        # Compute the decay score weights for all distances
        # decay score =0 for big distance, =1 for small distance
        decay_scores = np.exp(-distances / self.decay_dist)

        # Sum the decay scores across each row (for each point)
        # At each row the sum is supported mainly by closer points having greater effect
        cumulated_scores = decay_scores.sum(axis=1)

        # Fix sigmoid parameters
        pos = 5
        speed = 1

        # Apply sigmoid to combined scores to get scores in [0, 1]
        # sigmoid_decays is 1 when it's a bad region that is already densely populated
        sigmoid_decays = 1 / (1 + np.exp(-(cumulated_scores - pos) * speed))
        
        # Score is 0 for bad regions, and equal to one for good empty regions
        scores = 1 - sigmoid_decays

        return scores
    
    def get_parameters(self) -> Dict[str, float]:
        return {'decay_dist': self.decay_dist}


class OutlierProximityTerm(FittableFOMTerm):
    
    fit_config: ClassVar[Dict[str, bool]] = {'X_only': False, 'drop_nan': False}

    def __init__(self, exclusion_radius: float = 1e-5):
        self.exclusion_radius = exclusion_radius
        self.outlier_points = None

    @property
    def score_signs(self) -> List[Literal[1, -1]]:
        return [-1]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Detect outliers to avoid (in feature space). """
        y = np.atleast_2d(y.T).T  # Ensure y is 2D with shape (n_samples, n_targets)
        
        # Identify rows where any target column has NaN
        nan_mask = np.isnan(y).any(axis=1)

        # Extract feature values from these rows
        self.outlier_points = X[nan_mask]

    def _predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Proximity Avoidance Condition: Determine if the given point is
        sufficiently distant from any point with erroneous simulations. Points
        located within a specified exclusion radius around known problematic
        points render a low score (0) to be excluded from further processing.
        
        This condition is necessary because surrogate GPR does not fit on failed
        samples.
        """
        X = np.atleast_2d(X)

        if self.outlier_points.size == 0:
            return np.zeros(X.shape[0])

        # Compute distances based on the specified metric
        distances = distance.cdist(X, self.outlier_points, 'euclidean')

        # Determine which points should be ignored based on tolerance
        should_avoid = np.any(distances < self.exclusion_radius, axis=1)

        # Score is -1 if bad row that FOM should avoid else 0
        score = 0 - should_avoid.astype(float)  # 0 - value to avoid negative 0

        return score
    
    def get_parameters(self) -> Dict[str, Any]:
        return {'exclusion_radius': self.exclusion_radius}


class KDEModel:
    """
    A cool website to visualize KDE.
    https://mathisonian.github.io/kde/
    """
    def __init__(self, kernel='gaussian'):
        self.kernel = kernel
        self.bandwidth = None
        self.kde = None
        self.data = None
        self.is_trained = False
        self.modes = None
        self.max_density = None
        self.min_density = None

    def search_bandwidth(
        self, X, log_bounds=(-4, 0), n_iter=20,
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
            cv=5,  # Using cv here has no sense
            random_state=RANDOM_STATE,
            # scoring=None => use KDE.score which is log-likelihood
        )
        random_search.fit(X)

        # Get best bandwidth
        best_bandwidth = random_search.best_params_['bandwidth']
        print(f"KDEModel.search_bandwidth -> Optimal bandwidth found: {best_bandwidth}")
    
        return best_bandwidth

    def fit(self, X: np.ndarray, bandwidth: float, **kwargs):

        # Fit the KDE model
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth, **kwargs)
        self.kde.fit(X)

        # Update attributes
        self.data = X  # store training data
        self.bandwidth = bandwidth
        self.is_trained = True

    def update_modes(self):
        """
        Update modes of data clusters using Mean shift to later compute maximum
        modes density.
        """
        if self.kde is None:
            raise RuntimeError("Model must be fitted before searching for max density.")

        print(f"{self.__class__.__name__} -> Searching for modes...")

        mean_shift = MeanShift(bandwidth=self.bandwidth)
        mean_shift.fit(self.data)

        # Get modes (highest density points)
        self.modes = mean_shift.cluster_centers_

    def update_max_density(self):
        """
        Update maximum density in [0, 1]^p space using KDE density at modes.
        """
        if self.modes is None:
            raise RuntimeError("Modes must be updated before searching for max density.")

        self.max_density = self.predict_proba(self.modes).max()

    def update_min_density(self, shgo_iters: int = 5, shgo_n: int = 1000):
        """
        Update minimum density in [0, 1]^p space using SHGO minimizer to search
        for minimum density (anti-modes minimal density).
        """
        if self.kde is None:
            raise RuntimeError("Model must be fitted before searching for min density.")

        print(f"{self.__class__.__name__} -> Searching for maximum std...")

        def scalar_predict_proba(X):
            # SHGO minimizes the density
            return self.predict_proba(X.reshape(1, -1))[0]

        result = shgo(
            scalar_predict_proba,
            bounds=[(0, 1)]*self.data.shape[1],
            n=shgo_n,
            iters=shgo_iters,
            sampling_method='simplicial'
        )
        self.min_density = result.fun

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


class OutlierKDETerm(ModelFOMTerm, KDEModel):

    required_args = []
    fit_config = {'X_only': False, 'drop_nan': False}
    dependencies = ['surrogate_gpr']

    @property
    def score_signs(self) -> List[Literal[1, -1]]:
        return [-1]

    def fit(self, X: np.ndarray, y: np.ndarray, surrogate_gpr: SurrogateGPRTerm, **kwargs):

        # Get lenght scale from fitted GPR
        bandwidth = surrogate_gpr.kernel_.length_scale

        # Drop samplesfrom X where y is NaN
        X_outliers = X[np.isnan(y).any(axis=1)]

        # Must provide bandwidth to fit KDE
        KDEModel.fit(self, X_outliers, bandwidth, **kwargs)

        # Update maximum and minimum densities
        self.update_modes()
        self.update_max_density()
        self.update_min_density()

    def _predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Score is -1 if sample is in high density outlier region that FOM should
        avoid, else 0.
        """
        normalized_density = (self.predict_proba(X) - self.min_density) / (self.max_density - self.min_density)
        return 0 - normalized_density

    def get_model_params(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}

        return {
            'bandwidth': self.bandwidth,
            'max_density': self.max_density
        }

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'kernel': self.kernel,
            'bandwidth': self.bandwidth,
            'max_density': self.max_density
        }
