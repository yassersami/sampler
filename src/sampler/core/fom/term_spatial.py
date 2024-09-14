from typing import List, Tuple, Dict, Any, Literal, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import distance

from .term_base import FittableFOMTerm


class SigmoidDensityTerm(FittableFOMTerm):
    """
    A fittable FOM term that computes sigmoid density based on decay scores.
    """
    fit_config = {'X_only': True, 'drop_nan': False}

    def __init__(self, score_weights: float, bandwidth: float = 0.04):
        super().__init__(score_weights)
        self.bandwidth = bandwidth
        self.dataset_points = None

    @property
    def score_signs(self) -> Dict[str, Literal[1, -1]]:
        return {score_name: -1 for score_name in self.score_names}
    
    def fit(self, X: np.ndarray) -> None:
        self.dataset_points = X

    def _predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decay score and apply a sigmoid transformation to obtain a 
        density-like value.

        Parameters:
        - bandwidth (float): A float representing the characteristic distance
        at which the decay effect becomes significant. This parameter controls
        the rate at which the influence of a reference point decreases with
        distance. Lower values allow for the inclusion of farther points,
        effectively shifting the sigmoid curve horizontally. Note that for
        x_half = bandwidth * np.log(2), we have np.exp(-x_half/bandwidth) = 1/2,
        indicating the distance at which the decay effect reduces to half its
        initial value.
        
        Note: efficient emprical value found bandwidth = 0.04

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
        compared to bandwidth, hence they are fixed values. Also adequate
        sigmoid parameters are very hard to find.
        """
        X = np.atleast_2d(X)

        # Compute for each x_row distances to every dataset point
        # Element at position [i, j] is d(x_i, dataset_points_j)
        distances = distance.cdist(X, self.dataset_points, metric='euclidean')

        # Compute the decay score weights for all distances
        # decay score =0 for big distance, =1 for small distance
        decay_scores = np.exp(-distances / self.bandwidth)

        # Sum the decay scores across each row (for each point)
        # At each row the sum is supported mainly by closer points having greater effect
        cumulated_scores = decay_scores.sum(axis=1)

        # Fix sigmoid parameters
        pos = 5
        speed = 1

        # Apply sigmoid to combined scores to get scores in [0, 1]
        # sigmoid_decays is 1 when it's a bad region that is already densely populated
        sigmoid_decays = 1 / (1 + np.exp(-(cumulated_scores - pos) * speed))
        
        # Score is -1 for bad regions (already dense), and 0 for good empty regions
        scores = 0 - sigmoid_decays

        return scores
    
    def get_parameters(self) -> Dict[str, float]:
        params = FittableFOMTerm.get_parameters(self)
        params.update({'bandwidth': self.bandwidth})
        return params


class OutlierProximityTerm(FittableFOMTerm):
    
    fit_config = {'X_only': False, 'drop_nan': False}

    def __init__(self, score_weights: float, exclusion_radius: float = 1e-5):
        super().__init__(score_weights)
        self.exclusion_radius = exclusion_radius
        self.outlier_points = None

    @property
    def score_signs(self) -> Dict[str, Literal[1, -1]]:
        return {score_name: -1 for score_name in self.score_names}

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
        params = FittableFOMTerm.get_parameters(self)
        params.update({'exclusion_radius': self.exclusion_radius})
        return params
