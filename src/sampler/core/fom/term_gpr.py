from typing import List, Tuple, Dict, Union, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import NotFittedError
from scipy.optimize import shgo

from ..data_processing.scalers import interest_probability
from .term_base import ModelFOMTerm, MultiScoreMixin, KERNELS, RANDOM_STATE


class SurrogateGPR():
    def __init__(
        self,
        shgo_n: int,
        shgo_iters: int,
        interest_region: Dict[str, Tuple[float, float]],
        kernel: str,
    ):
        self.model = GaussianProcessRegressor(
            kernel=KERNELS[kernel], random_state=RANDOM_STATE
        )

        self.is_trained = False
        self.shgo_n = shgo_n
        self.shgo_iters = shgo_iters
        self.interest_region = np.array(list(interest_region.values()))

        # To normalize std of coverage function to be between 0 and 1
        self.max_std = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Check if y is 1D or 2D
        if y.ndim == 1:
            n_targets = 1
        elif y.ndim == 2:
            n_targets = y.shape[1]
        else:
            raise ValueError(f"Unexpected shape for y: {y.shape}")

        # Validate that bounds match the target dimension
        if len(self.interest_region) != n_targets:
            raise ValueError(
                f"Number of bounds ({len(self.interest_region)}) "
                f"does not match the number of targets ({n_targets})"
            )

        print(f"{self.__class__.__name__} -> Fitting model...")
        self.model.fit(X, y)
        self.is_trained = True

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the combined standard deviation for a GP regressor with multiple
        targets.
        """
        # Predict standard deviations for each target
        _, y_std = self.model.predict(X, return_std=True)

        # If 1D, it's already the standard deviation for a single target
        if y_std.ndim == 1:
            return y_std
        # If 2D, combine standard deviations using the mean across targets
        return y_std.mean(axis=1)
    
    def update_max_std(self):
        """
        Update maximum standard deviation of the Gaussian Process for points
        between 0 and 1.
        """
        if not self.is_trained:
            raise NotFittedError("The model must be trained before calling update_max_std.")

        print(f"{self.__class__.__name__} -> Searching for maximum std...")

        search_error = 0.01

        def get_opposite_std(X):
            """Opposite of std to be minimized."""
            X = np.atleast_2d(X)
            return -1 * self.predict_std(X)

        result = shgo(
            get_opposite_std,
            bounds=[(0, 1)]*self.model.n_features_in_,
            n=self.shgo_n,
            iters=self.shgo_iters,
            sampling_method='simplicial'
        )

        max_std = -1 * result.fun
        max_std = min(1.0, max_std * (1 + search_error))
        self.max_std = max_std

    def predict_interest_score(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the probability of being in the region of interest.
        """
        X = np.atleast_2d(X)

        y_mean, y_std = self.model.predict(X, return_std=True)

        probabilities = interest_probability(y_mean, y_std, self.interest_region)

        return probabilities

    def predict_std_score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if self.max_std is None:
            raise NotFittedError("max_std must be updated before predicting scores.")
        return self.predict_std(X) / self.max_std


class SurrogateGPRTerm(SurrogateGPR, MultiScoreMixin, ModelFOMTerm):

    required_args = ['interest_region']
    fit_config = {'X_only': False, 'drop_nan': True}
    all_scores= ['interest', 'std']
    
    def __init__(
        self,
        # Config kwargs
        score_config: Dict[str, bool],
        score_weights: Dict[str, float],
        shgo_n: int,
        shgo_iters: int,
        kernel: str,
        # kwargs required from FOM attributes 
        interest_region: Dict[str, Tuple[float, float]],
    ):

        SurrogateGPR.__init__(self,
            shgo_n=shgo_n,
            shgo_iters=shgo_iters,
            interest_region=interest_region,
            kernel=kernel,
        )

        MultiScoreMixin.__init__(self, score_config)  # Set score names
        score_names = MultiScoreMixin.get_active_scores(self)

        ModelFOMTerm.__init__(self, score_weights, score_names)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        SurrogateGPR.fit(self, X, y)
        if self.score_config['std'] and self.is_trained:
            # Update max_std of current surrogate GP
            self.update_max_std()

    def predict_score(self, X: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        X = np.atleast_2d(X)

        scores = []

        if self.score_config['interest']:
            scores.append(self.predict_interest_score(X))

        if self.score_config['std']:
            scores.append(self.predict_std_score(X))

        return tuple(scores) if len(scores) > 1 else scores[0]

    def get_model_params(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}

        # Get kernel parameters
        kernel_params = {
            k: v for k, v in self.model.kernel_.get_params().items()
            if not k.endswith('_bounds')
        }
        return {
            **kernel_params,
            'lml': self.model.log_marginal_likelihood_value_,
            'max_std': self.max_std,
        }

    def get_parameters(self) -> Dict:
        params = ModelFOMTerm.get_parameters(self)
        params.update({
            'is_trained': self.is_trained,
            **self.get_model_params()
        })

        if self.is_trained:
            params.update({
                'n_features': self.model.n_features_in_,
                'n_train': self.model.X_train_.shape[0],
            })

        return params
