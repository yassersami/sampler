from typing import List, Tuple, Dict, Union, Optional
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.gaussian_process import (
    GaussianProcessRegressor, GaussianProcessClassifier
)
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF
from scipy.optimize import shgo

RANDOM_STATE = 42


class SurrogateGPR(GaussianProcessRegressor):
    def __init__(
        self, 
        features: List[str], 
        targets: List[str],
        interest_region: Dict[str, Tuple[float, float]],
        kernel=None,
        random_state=RANDOM_STATE,
        **kwargs
    ):
        if kernel is None:
            kernel = RationalQuadratic(length_scale_bounds=(1e-5, 2))
        
        super().__init__(kernel=kernel, random_state=random_state, **kwargs)

        self.features = features
        self.targets = targets
        self.interest_region = interest_region
        
        self.lowers = [region[0] for region in interest_region.values()]
        self.uppers = [region[1] for region in interest_region.values()]

        # To normalize std of coverage function to be between 0 and 1
        self.max_std = None
    
    def predict_interest_proba(self, x: np.ndarray) -> np.ndarray:
        '''
        Computes the probability of being in the region of interest.
        CDF: cumulative distribution function P(X <= x)
        '''
        x = np.atleast_2d(x)

        y_hat, y_std = self.predict(x, return_std=True)

        point_norm = norm(loc=y_hat, scale=y_std)

        probabilities = point_norm.cdf(self.uppers) - point_norm.cdf(self.lowers)
        
        if y_hat.ndim == 1:
            return probabilities
        return np.prod(probabilities, axis=1)

    def get_std(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the combined standard deviation for a GP regressor with multiple
        targets.

        Note: In previous implementations, the mean of the Gaussian Process (GP)
        standard deviations was computed for the Figure of Merit (FOM). However,
        the maximum standard deviation was determined using only the first
        target's standard deviation (y_std[:, 1]). Then the max_std was computed
        using max of stadard deviations, but this approach was still lacking
        consistency. This function unifies the approach by consistently
        combining the standard deviations for both the determination of the
        maximum standard deviation and the application of FOM constraints.
        """
        # Predict standard deviations for each target
        _, y_std = self.predict(x, return_std=True)

        # Combine standard deviations using the mean
        y_std_combined = y_std.mean(axis=1)

        return y_std_combined

    def get_norm_std(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        if self.max_std is None:
            raise RuntimeError("max_std must be updated before predicting scores.")
        return self.get_std(x) / self.max_std
    
    def update_max_std(self, shgo_iters: int = 5, shgo_n: int = 1000):
        """
        Update maximum standard deviation of the Gaussian Process for points
        between 0 and 1.
        """
        print(
            "SurrogateGPR.update_max_std -> Searching for the maximum standard "
            "deviation of the surrogate..."
        )
        search_error = 0.01

        def get_opposite_std(x):
            """Opposite of std to be minimized."""
            x = np.atleast_2d(x)
            y_std = self.get_std(x)
            return -1 * y_std

        bounds = [(0, 1)]*len(self.features)
        result = shgo(
            get_opposite_std, bounds, iters=shgo_iters, n=shgo_n, sampling_method='simplicial'
        )

        max_std = -1 * result.fun
        max_std = min(1.0, max_std * (1 + search_error))
        self.max_std = max_std

        print(f"SurrogateGPR.update_max_std -> Maximum GP std: {max_std}")


class InlierOutlierGPC(GaussianProcessClassifier):
    def __init__(self, kernel=None, random_state=RANDOM_STATE):
        if kernel is None:
            kernel = 1.0 * RBF(length_scale=1.0)
        super().__init__(kernel=kernel, random_state=random_state)
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
