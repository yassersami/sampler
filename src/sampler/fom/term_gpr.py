from typing import List, Tuple, Dict, Union, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import NotFittedError
from scipy.optimize import shgo

from .term_base import ModelFOMTerm, KERNEL, RANDOM_STATE


class SurrogateGPR(GaussianProcessRegressor):
    def __init__(
        self, 
        interest_region: Dict[str, Tuple[float, float]],
        shgo_n: int,
        shgo_iters: int,
        **kwargs
    ):
        super().__init__(kernel=KERNEL, random_state=RANDOM_STATE, **kwargs)

        self.is_trained = False
        self.interest_region = interest_region
        self.shgo_n = shgo_n
        self.shgo_iters = shgo_iters

        self.lowers = [region[0] for region in interest_region.values()]
        self.uppers = [region[1] for region in interest_region.values()]

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
        if len(self.lowers) != n_targets or len(self.uppers) != n_targets:
            raise ValueError(
                f"Number of bounds ({len(self.lowers)}/{len(self.uppers)}) "
                f"does not match the number of targets ({n_targets})"
            )

        print(f"{self.__class__.__name__} -> Fitting model...")
        super().fit(X, y)
        self.is_trained = True

    def get_std(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the combined standard deviation for a GP regressor with multiple
        targets.
        """
        # Predict standard deviations for each target
        _, y_std = self.predict(X, return_std=True)

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
            return -1 * self.get_std(X)

        result = shgo(
            get_opposite_std,
            bounds=[(0, 1)]*self.n_features_in_,
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
        CDF: cumulative distribution function P(X <= x)
        """
        X = np.atleast_2d(X)

        y_mean, y_std = self.predict(X, return_std=True)

        point_norm = norm(loc=y_mean, scale=y_std)

        probabilities = point_norm.cdf(self.uppers) - point_norm.cdf(self.lowers)

        if y_mean.ndim == 1:
            return probabilities
        return np.prod(probabilities, axis=1)

    def predict_std_score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if self.max_std is None:
            raise NotFittedError("max_std must be updated before predicting scores.")
        return self.get_std(X) / self.max_std


class SurrogateGPRTerm(ModelFOMTerm, SurrogateGPR):
    
    required_args = ['interest_region']
    fit_config = {'X_only': False, 'drop_nan': True}
    
    def __init__(
        self,
        apply_interest: bool,
        apply_std: bool,
        interest_region: Dict[str, Tuple[float, float]],
        shgo_n: int,
        shgo_iters: int,
        **gpr_kwargs
    ):
        SurrogateGPR.__init__(
            self, interest_region=interest_region,
            shgo_n=shgo_n, shgo_iters=shgo_iters,
            **gpr_kwargs
        )
        self.apply_interest = apply_interest
        self.apply_std = apply_std
    
    @property
    def score_names(self) -> List[str]:
        score_names = []
        if self.apply_interest:
            score_names.append('gpr_interest')
        if self.apply_std:
            score_names.append('gpr_std')
        return score_names

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        SurrogateGPR.fit(self, X, y)
        if self.apply_std:
            # Update max_std of current surrogate GP
            self.update_max_std()

    def _predict_score(self, X: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        X = np.atleast_2d(X)

        scores = []
        
        if self.apply_interest:
            scores.append(self.predict_interest_score(X))
        
        if self.apply_std:
            scores.append(self.predict_std_score(X))
        
        if not scores:
            raise AttributeError("At least one of apply_interest or apply_std must be True")
        
        return tuple(scores) if len(scores) > 1 else scores[0]

    def get_model_params(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        kernel_params = {
            k: v for k, v in self.kernel_.get_params().items()
            if not k.endswith('_bounds')
        }
        return {
            **kernel_params,
            'lml': self.log_marginal_likelihood_value_,
            'max_std': self.max_std,
        }

    def get_parameters(self) -> Dict:
        params = {
            'apply_interest': self.apply_interest,
            'apply_std': self.apply_std,
            'kernel_str': str(self.kernel_),
            'is_trained': self.is_trained,
        }

        if self.is_trained:
            params.update({
                'kernel': self.kernel_.get_params(),
                'log_marginal_likelihood': self.log_marginal_likelihood_value_,
                'max_std': self.max_std,
                'n_features': self.n_features_in_,
                'n_train': self.X_train_.shape[0],
            })

        return params
