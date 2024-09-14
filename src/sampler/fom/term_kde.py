from typing import List, Tuple, Dict, Any, Literal, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import MeanShift
from scipy.optimize import shgo

from .term_base import ModelFOMTerm, RANDOM_STATE
from .term_gpr import SurrogateGPRTerm


class KDEModel:
    """
    A cool website to visualize KDE.
    https://mathisonian.github.io/kde/
    """
    def __init__(self):
        self.kernel = 'gaussian'
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

    def __init__(self, score_weights: float):
        ModelFOMTerm.__init__(self, score_weights)
        KDEModel.__init__(self)

    @property
    def score_signs(self) -> Dict[str, Literal[1, -1]]:
        return {score_name: -1 for score_name in self.score_names}

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
        params = ModelFOMTerm.get_parameters(self)
        params.update({
            'kernel': self.kernel,
            'bandwidth': self.bandwidth,
            'max_density': self.max_density
        })
        return params
