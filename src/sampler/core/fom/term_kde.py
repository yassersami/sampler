from typing import List, Tuple, Dict, Any, Literal, Union, Optional
import warnings
import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from scipy.optimize import shgo

from .term_base import ModelFOMTerm, BANDWIDTH_BOUNDS, RANDOM_STATE
from .term_gpr import SurrogateGPRTerm
from .kde_utils import brentq_bandwidth, search_cv_bandwidth, mean_shift_modes


class KDEModel:
    """
    A cool website to visualize KDE.
    https://mathisonian.github.io/kde/
    """
    def __init__(self):
        self.kernel_name = 'gaussian'
        self.model: KernelDensity = None
        self.data = None
        self.is_trained = False
        self.modes = None
        self.max_density = None
        self.min_density = None

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

    def update_max_density(self):
        """
        Update maximum density in [0, 1]^p space using KDE density at modes.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be fitted before searching for max density.")

        print(f"{self.__class__.__name__} -> Searching for maximum density...")

        modes = mean_shift_modes(self.data, self.model.bandwidth_)
        self.max_density = self.predict_proba(modes).max()

    def update_min_density(self):
        """
        Update minimum density in [0, 1]^p space using SHGO minimizer to search
        for minimum density (anti-modes minimal density).
        """
        if not self.is_trained:
            raise RuntimeError("Model must be fitted before searching for min density.")

        print(f"{self.__class__.__name__} -> Searching for minimum density...")

        # Fix SHGO parameters
        shgo_iters = 5
        shgo_n = 1000

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


class KDETerm(KDEModel, ModelFOMTerm):

    fit_config = {'X_only': False, 'drop_nan': False}
    bw_config_keys = ['use_brentq', 'use_search_cv', 'use_heuristic']

    def __init__(self,
        score_weights: float,
        heuristic: Union[float, str],
        null_density_proportion: float,
        bandwidth_config: Dict[str, Union[bool, float, str]]
    ):
        KDEModel.__init__(self)
        ModelFOMTerm.__init__(self, score_weights)

        self._validate_heuristic(heuristic)
        self._validate_null_density_proportion(null_density_proportion)
        self._validate_bandwidth_config(bandwidth_config)

        self.bandwidth_config = bandwidth_config
        self.heuristic = heuristic
        self.null_density_proportion = null_density_proportion

    @property
    def score_signs(self) -> Dict[str, Literal[1, -1]]:
        return {score_name: -1 for score_name in self.score_names}

    def _validate_bandwidth_config(self, bandwidth_config):
        """
        Validate the bandwidth configuration.
        """
        intro = f"{self.__class__.__name__}: "

        # Check if all required keys are present
        missing_keys = [key for key in self.bw_config_keys if key not in bandwidth_config]
        if missing_keys:
            raise ValueError(intro + f"Bandwidth config is missing keys: {missing_keys}")

        # Check if boolean keys are indeed boolean
        non_boolean_keys = [
            key for key in self.bw_config_keys
            if not isinstance(bandwidth_config[key], bool)
        ]
        if non_boolean_keys:
            raise ValueError(intro + f"These keys must be boolean: {non_boolean_keys}")

        # Check if mutually exclusive keys are not true at the same time
        true_keys = [key for key in self.bw_config_keys if bandwidth_config[key]]
        if len(true_keys) > 1:
            raise ValueError(intro + f"Only one of these can be True: {true_keys}")

    def _validate_heuristic(self, heuristic):
        # Check if heuristic is valid string option
        if isinstance(heuristic, str):
            if heuristic not in ['scott', 'silverman']:
                raise ValueError(
                    "'heuristic' must be 'scott', 'silverman', "
                    f"or a float within {BANDWIDTH_BOUNDS} range"
                )

        # Check if heuristic is valid number
        elif isinstance(heuristic, (int, float)):
            if not BANDWIDTH_BOUNDS[0] <= heuristic <= BANDWIDTH_BOUNDS[1]:
                raise ValueError(
                    "When 'heuristic' is a number, it must be "
                    f"within {BANDWIDTH_BOUNDS} range"
                )

        # Else, raise error
        else:
            raise ValueError(
                "'heuristic' must be 'scott', 'silverman', or "
                f"a float within {BANDWIDTH_BOUNDS} range"
            )

    def _validate_null_density_proportion(self, null_density_proportion):
        # Check if null_density_proportion is a number
        if not isinstance(null_density_proportion, (int, float)):
            raise ValueError("'null_density_proportion' must be a number")

        # Check if null_density_proportion is valid number
        if null_density_proportion > 0.5:
            raise ValueError(
                "'null_density_proportion' must preferably be in [0, 0.5] range. "
                f"Got: {null_density_proportion}"
            )

    def get_bandwidth(self, X: np.ndarray) -> float:
        intro = f"{self.__class__.__name__} -> "

        if self.bandwidth_config['use_brentq']:
            # Find optimal bandwidth using BrentQ
            print(intro + f"Searching optimal bandwidth using BrentQ...")
            previous_bw = self.model.bandwidth_ if self.is_trained else None
            bandwidth = brentq_bandwidth(X, self.kernel_name, self.null_density_proportion, previous_bw)

        elif self.bandwidth_config['use_search_cv']:
            # Find optimal bandwidth using RandomizedSearchCV
            print(intro + "Searching bandwidth using CV-log-likelihood...")
            bandwidth = search_cv_bandwidth(X, self.kernel_name)

        elif self.bandwidth_config['use_heuristic']:
            # Use heursitic
            bandwidth = self.heuristic

        return bandwidth

    def fit(self, X: np.ndarray, bandwidth: float) -> None:
        KDEModel.fit(self, X, bandwidth)

        # Update maximum and minimum densities
        self.update_max_density()
        self.update_min_density()

    def predict_score(self, X: np.ndarray) -> np.ndarray:
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
            'max_density': self.max_density,
            'min_density': self.min_density
        }

    def get_parameters(self) -> Dict[str, Any]:
        params = ModelFOMTerm.get_parameters(self)
        params.update({
            'bandwidth_config': self.bandwidth_config,
            'kernel': self.kernel_name,
            **self.get_model_params(),
        })
        if self.is_trained:
            params.update({
                'n_train': self.data.shape[0],
            })
        return params


class OutlierKDETerm(KDETerm):

    dependencies = ['surrogate_gpr']
    bw_config_keys = KDETerm.bw_config_keys + ['use_gpr_kernel']

    def get_bandwidth(self, X: np.ndarray, dependency_terms: Dict[str, ModelFOMTerm]) -> float:
        intro = f"{self.__class__.__name__} -> "

        if self.bandwidth_config['use_gpr_kernel']:
            # Parse dependency terms
            surrogate_gpr: SurrogateGPRTerm = dependency_terms['surrogate_gpr']

            if surrogate_gpr is None:
                raise ValueError(
                    intro + "'surrogate_gpr' term must be applied when "
                    "'use_gpr_kernel' is True."
                )
            # Get lenght scale from fitted GPR
            bandwidth = surrogate_gpr.model.kernel_.length_scale

        else:
            bandwidth = super().get_bandwidth(X)

        return bandwidth

    def fit(self, X: np.ndarray, y: np.ndarray, dependency_terms: Dict[str, ModelFOMTerm]):
        """
        Fit the KDE model on outlier data with a bandwidth determined from the
        full dataset.

        This method performs the following steps:
        1. Identify outliers as samples where y contains NaN values.
        2. If any outliers are encountred, proceed next steps.
        3. Determine the bandwidth based on bandwidth_config, using the entire
           dataset (X) to ensure it's suitable for the whole design space.
        4. Fit the KDE term using only the outlier data points.
        """
        # Drop samples from X where y is NaN
        X_outliers = X[np.isnan(y).any(axis=1)]

        if X_outliers.shape[0] != 0:
            # Get bandwidth using all data
            bandwidth = self.get_bandwidth(X, dependency_terms)

            # Fit KDE on outliers only
            super().fit(X_outliers, bandwidth)


class FeatureKDETerm(KDETerm):

    def fit(self, X: np.ndarray, y: np.ndarray):
        """ Fit the KDE model on all samples in feature space. """
        # Get bandwidth using all data
        bandwidth = self.get_bandwidth(X)

        # Fit KDE on outliers only
        super().fit(X, bandwidth)


class TargetKDETerm(KDETerm):

    fit_config = {'X_only': False, 'drop_nan': True}
    # All possible terms with regression model 
    dependencies = ['surrogate_gpr']

    def get_regression_model(self, dependency_terms: Dict[str, ModelFOMTerm]) -> ModelFOMTerm:
        # Check that exactly one term is not None
        non_none_terms = {
            name: term for name, term in dependency_terms.items()
            if term is not None
        }

        if len(non_none_terms) == 1:
            return next(iter(non_none_terms.values()))
        else:
            raise ValueError(
                f"{self.__class__.__name__} -> Expected exactly one non-None term, "
                f"but found {len(non_none_terms)}: {list(non_none_terms)}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray, dependency_terms: Dict[str, ModelFOMTerm]):
        """ Fit the KDE model on all samples in target space. """

        # Parse dependency terms
        self.regression_model = self.get_regression_model(dependency_terms)

        # Get bandwidth using all data
        bandwidth = self.get_bandwidth(y)  # different space, can't use GP kernel option

        # Fit KDE on outliers only
        super().fit(y, bandwidth)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """ Predict density score of y response of X point."""
        # KDE is fitted in target space, so must map X to target space first
        y_pred = self.regression_model.model.predict(X)
        y_pred = np.atleast_2d(y_pred.T).T
        return super().predict_score(y_pred)
