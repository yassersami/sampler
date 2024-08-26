from typing import List, Tuple, Dict, Union, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.gaussian_process import (
    GaussianProcessRegressor, GaussianProcessClassifier
)
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
from sklearn.utils.validation import check_is_fitted
from scipy.linalg import solve
from scipy.optimize import shgo

from .base import FOMTermRegistry, FittableFOMTerm

RANDOM_STATE = 42

# TODO:
# - adapt bounds and predict_interest_score for 1D y
# - updat_max_std has problems with 2D X
class SurrogateGPR(GaussianProcessRegressor):
    def __init__(
        self, 
        interest_region: Dict[str, Tuple[float, float]],
        shgo_n: int = 1000,
        shgo_iters: int = 5,
        **kwargs
    ):
        kernel = RationalQuadratic(length_scale_bounds=(1e-5, 10))
        super().__init__(kernel=kernel, random_state=RANDOM_STATE, **kwargs)

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
        print(
            "SurrogateGPR.update_max_std -> Searching for the maximum standard "
            "deviation of the surrogate..."
        )
        if not self.is_trained:
            raise RuntimeError("The model must be trained before calling update_max_std.")

        search_error = 0.01

        def get_opposite_std(x):
            """Opposite of std to be minimized."""
            x = np.atleast_2d(x)
            y_std = self.get_std(x)
            return -1 * y_std

        bounds = [(0, 1)]*self.n_features_in_
        result = shgo(
            get_opposite_std, bounds, n=self.shgo_n, iters=self.shgo_iters,
            sampling_method='simplicial'
        )

        max_std = -1 * result.fun
        max_std = min(1.0, max_std * (1 + search_error))
        self.max_std = max_std

        print(f"SurrogateGPR.update_max_std -> Maximum GP std: {max_std}")
    
    def predict_interest_score(self, X: np.ndarray) -> np.ndarray:
        '''
        Computes the probability of being in the region of interest.
        CDF: cumulative distribution function P(X <= x)
        '''
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
            raise RuntimeError("max_std must be updated before predicting scores.")
        return self.get_std(X) / self.max_std


@FOMTermRegistry.register("surrogate_gpr")
class SurrogateGPRTerm(FittableFOMTerm, SurrogateGPR):
    
    required_args = ["interest_region"]
    fit_params = {'X_only': False, 'drop_nan': True}
    
    def __init__(
        self,
        apply: bool,
        apply_interest: bool,
        apply_std: bool,
        interest_region: Dict[str, Tuple[float, float]],
        shgo_n: int,
        shgo_iters: int,
        **gpr_kwargs
    ):
        FittableFOMTerm.__init__(self, apply=apply)
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

    def predict_score(self, X: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.apply_interest and self.apply_std:
            return self.predict_interest_score(X), self.predict_std_score(X)
        elif self.apply_interest:
            return self.predict_interest_score(X)
        elif self.apply_std:
            return self.predict_std_score(X)
        else:
            raise AttributeError("At least one of apply_interest or apply_std must be True")

    def get_parameters(self, add_details: bool = False):
        params = {
            'kernel': self.kernel_.get_params(),
            'kernel_str': str(self.kernel_),
        }

        if self.is_trained:
            params.update({
                'log_marginal_likelihood': self.log_marginal_likelihood_value_,
                'n_features': self.n_features_in_,
                'n_train': self.X_train_.shape[0],
            })

        if not add_details:
            return params

        params.update({
            'alpha': self.alpha,
            'optimizer': self.optimizer,
            'n_restarts_optimizer': self.n_restarts_optimizer,
            'normalize_y': self.normalize_y,
            'copy_X_train': self.copy_X_train,
            'random_state': self.random_state,
        })

        if self.is_trained:
            params.update({
                'noise_level': self.noise_,
                'optimizer_results': {
                    'fun': self._optimizer_results.fun,
                    'nit': self._optimizer_results.nit,
                    'message': self._optimizer_results.message,
                }
            })

        return params
    

class GPCModel(GaussianProcessClassifier):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_a(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Return estimates of the latent function for X.
        
        Notes:
        ------
        - For binary classification (n_classes = 2), the output shape is
        (n_samples,).
        - For multi-class classification, the output shape is (n_samples,
        n_classes) when multi_class="one_vs_rest", and is shaped (n_samples,
        n_classes*(n_classes - 1)/2) when multi_class="one_vs_one". In other
        terms, There are as many columns as trained Binary GPC sub-models.
        - The number of classes (n_classes) is determined by the number of
        unique target values in the training data.
        """
        check_is_fitted(self)

        if self.n_classes_ > 2:  # Multi-class case
            f_stars = []
            std_f_stars = []
            for estimator, kernel in zip(self.base_estimator_.estimators_, self.kernel_.kernels):
                result = self._binary_predict_a(estimator, kernel, X, return_std)
                if not return_std:
                    f_stars.append(result)
                else:
                    f_stars.append(result[0])
                    std_f_stars.append(result[1])

            if not return_std:
                return np.array(f_stars).T

            return np.array(f_stars).T, np.array(std_f_stars).T
        else:  # Binary case
            return self._binary_predict_a(self.base_estimator_, self.kernel_, X, return_std)

    @staticmethod
    def _binary_predict_a(estimator, kernel, X, return_std):
        """ Return mean and std of the latent function estimates for X. """
        check_is_fitted(estimator)

        # Based on Algorithm 3.2 of GPML
        K_star = kernel(estimator.X_train_, X)  # K_star = k(x_star)
        f_star = K_star.T.dot(estimator.y_train_ - estimator.pi_)  # Line 4
        if not return_std:
            return f_star

        v = solve(estimator.L_, estimator.W_sr_[:, np.newaxis] * K_star)  # Line 5
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        var_f_star = kernel.diag(X) - np.einsum("ij,ij->j", v, v)

        return f_star, np.sqrt(var_f_star)


class InlierOutlierGPC(GPCModel):
    def __init__(self, kernel=None):
        if kernel is None:
            kernel = 1.0 * RBF(length_scale=1.0)
        super().__init__(kernel=kernel, random_state=RANDOM_STATE)
        self.is_trained = False

        # Map class labels to indices
        self.class_to_index = {"outlier": 0, "inlier": 1}
        self.index_to_class = {
            index: label for label, index in self.class_to_index.items()
        }

    def fit(self, X, y):
        # Determine inlier or outlier status based on NaN presence in y
        y = np.where(
            np.isnan(y).any(axis=1),
            self.class_to_index["outlier"],  # 0
            self.class_to_index["inlier"]   # 1
        )
        # Check if there are two classes in y
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.is_trained = False
            warnings.warn("No outliers have been detected yet. The model will not be fitted.")
            return self
        else:
            self.is_trained = True
            # Fit the Gaussian Process Classifier with the modified target values
            super().fit(X, y)
        return self
    
    def get_inlier_bstd(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate Bernoulli Standard Deviation weighed by the inlier proba for
        each input sample.
        
        This function computes the score using the formula:
        
                f(P) = alpha * P * sqrt(P*(1-P))
    
        Where:
        - P = P(x is inlier) is the probability that a sample is an inlier.
        - sqrt(P*(1-P)) is Bernoulli standard deviation. It quantifies the
        predication uncertainty in the predicted class outcome (0, 1), not the
        uncertainty provided by latent function values (logit space).
        - alpha is a scaling factor such that the maximum value of f over the
        interval [0, 1] is 1. With P_max = argmax[0, 1](f) = 3/4.
        """
        X = np.atleast_2d(X)
        
        # Return ones (best value) if the model is not trained as no outliers are encountered yet
        if not self.is_trained:
            return np.ones(X.shape[0])

        # Set scaling factor alpha
        alpha = ( (3/4)**3 * (1/4) )**(-0.5)

        # Predict inlier probabilities
        proba = self.predict_proba(X)[:, self.class_to_index["inlier"]]

        # Compute the score using the given formula
        score = alpha * proba * np.sqrt(proba * (1 - proba))

        return score

    def predict_inlier_entropy(self, X):
        """
        Compute the entropy weighed by the inlier proba for each input sample.
        
        This function computes the score using the formula:
        
            f(P) = alpha * P * (-P*log(P) - (1-P)*log(1-P))

        Note:
        - Entropy is a measure of uncertainty. Higher values indicate more
        uncertainty. For binary classification, the maximum entropy is
        ln(2) ≈ 0.693.
        """
        # Get probabilities for the "inlier" class
        proba = self.predict_proba(X)[:, self.class_to_index["inlier"]]

        # Clip probabilities to avoid log(0)
        proba = np.clip(proba, 1e-15, 1 - 1e-15)

        # Compute entropy
        entropy = -proba * np.log(proba) - (1 - proba) * np.log(1 - proba)
        score = 1/0.1858 * proba * entropy

        return score
