from typing import List, Tuple, Dict, Any, Literal, Union, ClassVar
import warnings
import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.cluster import MeanShift
from scipy.optimize import shgo

from .term_base import ModelFOMTerm, BANDWIDTH_BOUNDS, RANDOM_STATE
from .term_gpr import SurrogateGPRTerm


class KDEModel:
    """
    A cool website to visualize KDE.
    https://mathisonian.github.io/kde/
    """
    def __init__(self):
        self.kernel_name = 'gaussian'
        self.model = None
        self.data = None
        self.is_trained = False
        self.modes = None
        self.max_density = None
        self.min_density = None

    def search_cv(self, X: np.ndarray) -> float:
        """
        - The optimal bandwidth is the largest possible width for which the
        density at the modes is above a threshold equal to 0.8. Since the
        locations of the modes do not change, it's computationaly more
        convenient to express this in terms of maximum density rather than
        minimum density. Formally:

                best_bandwidth = max({bandwidth | density_at_modes > 0.8})
        """

        print(f"{self.__class__.__name__} -> Searching bandwidth using CV-log-likelihood...")

        # Define a range of bandwidths to search over using log scale
        bandwidths = np.logspace(np.log10(BANDWIDTH_BOUNDS[0]), np.log10(BANDWIDTH_BOUNDS[1]), 100)

        # Use RandomizedSearchCV to find the best bandwidth
        random_search = RandomizedSearchCV(
            KernelDensity(kernel=self.kernel_name),
            {'bandwidth': bandwidths},
            n_iter=20,
            cv=5,
            random_state=RANDOM_STATE,
            # scoring=None => use KDE.score which is log-likelihood
        )
        random_search.fit(X)

        # Get best bandwidth
        best_bandwidth = random_search.best_params_['bandwidth']

        return best_bandwidth

    def compute_cv(self, X):
        """ Compute 5-fold cross-validation score for the KDE. """
        # Ensure X is a 2D array
        X = np.atleast_2d(X)

        # Compute 5-fold cross-validation score
        cv_scores = cross_val_score(
            self.model,
            X,
            cv=5,
            scoring=lambda estimator, X: estimator.score(X)
        )

        # Return the mean score
        return np.mean(cv_scores)

    def fit(self, X: np.ndarray, bandwidth: float):
        # Check if X is empty
        if X.shape[0] == 0:
            self.is_trained = False
        else:
            # Fit the KDE model
            # bandwidth doesn't change during training
            # Fit is for tree-based algorithm to use for efficient density estimation
            self.model = KernelDensity(kernel=self.kernel_name, bandwidth=bandwidth)
            self.model.fit(X)

            # Update attributes
            self.data = X  # store training data
            self.is_trained = True

    def update_modes(self):
        """
        Update modes of data clusters using Mean shift to later compute maximum
        modes density.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before searching for max density.")

        print(f"{self.__class__.__name__} -> Searching for modes...")

        mean_shift = MeanShift(bandwidth=self.model.bandwidth_)
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
        if self.model is None:
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
        if self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        log_density = self.model.score_samples(X)
        return np.exp(log_density)


class OutlierKDETerm(ModelFOMTerm, KDEModel):

    required_args = []
    fit_config = {'X_only': False, 'drop_nan': False}
    dependencies = ['surrogate_gpr']

    def __init__(self, score_weights: float, bandwidth_config: Dict[str, Union[bool, float, str]]):
        ModelFOMTerm.__init__(self, score_weights)
        KDEModel.__init__(self)
        self._validate_bandwidth_config(bandwidth_config)
        self.bandwidth_config = bandwidth_config

    @property
    def score_signs(self) -> Dict[str, Literal[1, -1]]:
        return {score_name: -1 for score_name in self.score_names}

    def _validate_bandwidth_config(self, bandwidth_config):
        """
        Validate the bandwidth configuration.
        """
        # Check if all required keys are present
        required_keys = ['use_gpr_kernel', 'use_grid_search', 'heuristic']
        if not all(key in bandwidth_config for key in required_keys):
            raise ValueError(
                f"Bandwidth config must contain all of these keys: {required_keys}"
            )

        use_gpr_kernel = bandwidth_config['use_gpr_kernel']
        use_grid_search = bandwidth_config['use_grid_search']
        heuristic = bandwidth_config['heuristic']

        # 1. Check if use keys are both boolean
        if not isinstance(use_gpr_kernel, bool) or not isinstance(use_grid_search, bool):
            raise ValueError(
                "'use_gpr_kernel' and 'use_grid_search' must be boolean values"
            )

        # 2. Check if use keys are not true at the same time
        if use_gpr_kernel and use_grid_search:
            raise ValueError(
                "'use_gpr_kernel' and 'use_grid_search' cannot both be True"
            )

        # 3. Check heuristic value when both use keys are false
        if not use_gpr_kernel and not use_grid_search:
            if isinstance(heuristic, str):
                if heuristic not in ['scott', 'silverman']:
                    raise ValueError(
                        "'heuristic' must be 'scott', 'silverman', "
                        f"or a float within {BANDWIDTH_BOUNDS} range"
                    )
            elif isinstance(heuristic, (int, float)):
                if not BANDWIDTH_BOUNDS[0] <= heuristic <= BANDWIDTH_BOUNDS[1]:
                    raise ValueError(
                        "When 'heuristic' is a number, it must be "
                        f"within {BANDWIDTH_BOUNDS} range"
                    )
            else:
                raise ValueError(
                    "'heuristic' must be 'scott', 'silverman', or "
                    f"a float within {BANDWIDTH_BOUNDS} range"
                )

    def get_bandwidth(self, X: np.ndarray, surrogate_gpr: SurrogateGPRTerm) -> float:
        if self.bandwidth_config['use_gpr_kernel']:
            # Get lenght scale from fitted GPR
            bandwidth = surrogate_gpr.model.kernel_.length_scale

        elif self.bandwidth_config['use_grid_search']:
            # Find optimal bandwidth using RandomizedSearchCV
            bandwidth = self.search_cv(X)

        else:
            # Use heursitic
            bandwidth = self.bandwidth_config['heuristic']

        return bandwidth

    def fit(self, X: np.ndarray, y: np.ndarray, surrogate_gpr: SurrogateGPRTerm):

        # Drop samplesfrom X where y is NaN
        X_outliers = X[np.isnan(y).any(axis=1)]

        # Get bandwidth
        bandwidth = self.get_bandwidth(X, surrogate_gpr)

        # Must provide bandwidth to fit KDE
        KDEModel.fit(self, X_outliers, bandwidth)

        if self.is_trained:
            # Update maximum and minimum densities
            self.update_modes()
            self.update_max_density()
            self.update_min_density()

    def _predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Score is -1 if sample is in high density outlier region that FOM should
        avoid, else 0.
        """
        # Return zeros (best value) if the model is not trained,
        # as no outlier samples are encountered yet
        if not self.is_trained:
            return np.zeros(X.shape[0])

        normalized_density = (self.predict_proba(X) - self.min_density) / (self.max_density - self.min_density)
        return 0 - normalized_density

    def get_model_params(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}

        return {
            'bandwidth': self.model.bandwidth_,
            'cv_log_likelihood': self.compute_cv(self.data),
            'max_density': self.max_density,
            'min_density': self.min_density
        }

    def get_parameters(self) -> Dict[str, Any]:
        params = ModelFOMTerm.get_parameters(self)
        params.update({
            'bandwidth_config': self.bandwidth_config,
            'kernel': self.kernel_name,
            **self.get_model_params()
        })
        return params
