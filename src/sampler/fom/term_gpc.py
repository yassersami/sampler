from typing import List, Tuple, Dict, Union, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from scipy.linalg import solve
from scipy.optimize import shgo

from .term_base import FittableFOMTerm

RANDOM_STATE = 42
KERNEL = RationalQuadratic(length_scale_bounds=(1e-5, 10))


class LatentGPC(GaussianProcessClassifier):
    
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
        n_classes) when multi_class='one_vs_rest', and is shaped (n_samples,
        n_classes*(n_classes - 1)/2) when multi_class='one_vs_one'. In other
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
        var_f_star = kernel.diag(X) - np.einsum('ij,ij->j', v, v)

        return f_star, np.sqrt(var_f_star)


class BinaryLatentGPC(LatentGPC):
    def __init__(
        self,
        positive_class: str,
        negative_class: int,
        shgo_n: int,
        shgo_iters: int,
        **gpc_kwargs
    ):
        super().__init__(**gpc_kwargs)
        self.is_trained = False
        self.max_std = None
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.shgo_n = shgo_n
        self.shgo_iters = shgo_iters

        # Map class labels to indices
        self.class_to_index = {self.negative_class: 0, self.positive_class: 1}
        self.index_to_class = {
            index: label for label, index in self.class_to_index.items()
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Check if there are two classes in y
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)

        if num_classes < 2:
            self.is_trained = False
            warnings.warn(
                f"No {self.negative_class}s have been detected yet. "
                "The model will not be fitted."
            )
        elif num_classes == 2:
            self.is_trained = True
            # Fit the Gaussian Process Classifier
            super().fit(X, y)
        else:
            raise ValueError(
                f"Expected 2 classes, but got {num_classes} classes. "
                f"This model only supports binary classification."
            )
    
    def update_max_std(self):
        """
        Update maximum standard deviation of the Gaussian Process for points
        between 0 and 1.
        """
        print(f"{self.__class__.__name__}.update_max_std -> Searching for the maximum std...")
        if not self.is_trained:
            raise NotFittedError("The model must be trained before calling update_max_std.")

        search_error = 0.01

        def get_opposite_std(X):
            """Opposite of std to be minimized."""
            X = np.atleast_2d(X)
            _, y_std = self.predict_a(X, return_std=True)
            return -1 * y_std

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

        print(f"{self.__class__.__name__}.update_max_std -> Maximum GP std: {max_std}")

    def get_std_score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if self.max_std is None:
            raise NotFittedError("max_std must be updated before predicting scores.")
        _, y_std = self.predict_a(X, return_std=True)
        return y_std / self.max_std

    def get_positive_bstd(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate Bernoulli Standard Deviation weighed by the positive class
        probability for each input sample.
        
        This function computes the score using the formula:
        
                f(P) = alpha * P * sqrt(P*(1-P))
    
        Where:
        - P = P(x is positive_class) is the probability that a sample belongs to
          the positive class.
        - sqrt(P*(1-P)) is Bernoulli standard deviation. It quantifies the
          prediction uncertainty in the predicted class outcome (0, 1), not the
          uncertainty provided by latent function values (logit space).
        - alpha is a scaling factor such that the maximum value of f over the
          interval [0, 1] is 1. With P_max = argmax[0, 1](f) = 3/4.
        """
        X = np.atleast_2d(X)

        # Set scaling factor alpha

        # Predict positive class probabilities
        proba = self.predict_proba(X)[:, self.class_to_index[self.positive_class]]

        # Compute positively weighted Bernoulli std
        max_0_1 = np.sqrt((3/4)**3 * (1/4))  # Maximum value on ]0, 1[
        bernoulli_std = np.sqrt(proba * (1 - proba))

        score = 1/max_0_1 * proba * bernoulli_std

        return score

    def predict_positive_entropy(self, X):
        """
        Compute the entropy weighed by the positive class probability for each
        input sample.
        
        This function computes the score using the formula:
        
                f(P) = alpha * P * (-P*log(P) - (1-P)*log(1-P))

        Note: 
        - Entropy is a measure of uncertainty. Higher values indicate more
          uncertainty. For binary classification, the maximum entropy is
          ln(2) ≈ 0.693.
        """
        X = np.atleast_2d(X)

        # Get probabilities for the positive class
        proba = self.predict_proba(X)[:, self.class_to_index[self.positive_class]]

        # Clip probabilities to avoid log(0)
        proba = np.clip(proba, 1e-15, 1 - 1e-15)

        # Compute positively weighted binary entropy (Shanon)
        max_0_1 = 0.616949  # Maximum value on ]0, 1[
        entropy = -proba * np.log2(proba) - (1 - proba) * np.log2(1 - proba)

        score = 1/max_0_1 * proba * entropy

        return score


# TODO: finish BinaryLatentGPCTerm and split surrogate into GPR and GPR
class BinaryLatentGPCTerm(FittableFOMTerm, BinaryLatentGPC):
    
    required_args = []
    fit_params = {'X_only': False, 'drop_nan': False}
    
    def __init__(
        self,
        apply_proba: bool,
        apply_std: bool,
        apply_bstd: bool,
        apply_entropy: bool,
        positive_class: str,
        negative_class: str,
        shgo_n: int,
        shgo_iters: int,
        **gpc_kwargs
    ):
        BinaryLatentGPC.__init__(
            self,
            positive_class=positive_class,
            negative_class=negative_class,
            shgo_n=shgo_n,
            shgo_iters=shgo_iters,
            **gpc_kwargs
        )
        self.apply_proba = apply_proba
        self.apply_std = apply_std
        self.apply_bstd = apply_bstd
        self.apply_entropy = apply_entropy
        
    @property
    def score_names(self) -> List[str]:
        score_names = []
        if self.apply_proba:
            score_names.append(f'{self._term_name}_{self.positive_class}_proba')
        if self.apply_std:
            score_names.append(f'{self._term_name}_std')
        if self.apply_bstd:
            score_names.append(f'{self._term_name}_{self.positive_class}_bstd')
        if self.apply_entropy:
            score_names.append(f'{self._term_name}_{self.positive_class}_entropy')
        return score_names

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        BinaryLatentGPC.fit(self, X, y)
        if self.apply_std and self.is_trained:
            # Update max_std of current GPC
            self.update_max_std()

    def predict_score(self, X: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        X = np.atleast_2d(X)

        # Return ones (best value) if the model is not trained,
        # as no negative class samples are encountered yet
        if not self.is_trained:
            count_active_scores = sum([self.apply_proba, self.apply_std, self.apply_bstd, self.apply_entropy])
            scores = tuple([np.ones(X.shape[0])]*count_active_scores)
            return scores if len(scores) > 1 else scores[0]

        scores = []
        
        if self.apply_proba:
            proba = self.predict_proba(X)[:, self.class_to_index[self.positive_class]]  # 1
            scores.append(proba)
        
        if self.apply_std:
            scores.append(self.get_std_score(X))
        
        if self.apply_bstd:
            scores.append(self.get_positive_bstd(X))
        
        if self.apply_entropy:
            scores.append(self.predict_positive_entropy(X))
        
        if not scores:
            raise AttributeError(
                "At least one of apply_bstd, apply_entropy, apply_proba, "
                "or apply_std must be True"
            )
        
        return tuple(scores) if len(scores) > 1 else scores[0]

    def get_parameters(self, add_details: bool = False):
        params = {
            'apply_proba': self.apply_proba,
            'apply_std': self.apply_std,
            'apply_bstd': self.apply_bstd,
            'apply_entropy': self.apply_entropy,
            'kernel': self.kernel_.get_params(),
            'kernel_str': str(self.kernel_),
            'is_trained': self.is_trained,
        }

        if self.is_trained:
            params.update({
                'max_std': self.max_std,
                'log_marginal_likelihood': self.log_marginal_likelihood_value_,
                'n_features': self.n_features_in_,
                'n_train': self.base_estimator_.X_train_.shape[0],
                'classes': self.classes_,
            })

        if not add_details:
            return params

        params.update({
            'optimizer': self.optimizer,
            'n_restarts_optimizer': self.n_restarts_optimizer,
            'max_iter_predict': self.max_iter_predict,
            'warm_start': self.warm_start,
            'copy_X_train': self.copy_X_train,
            'random_state': self.random_state,
        })

        if self.is_trained:
            params.update({
                'n_classes': self.n_classes_,
            })

        return params


class InterestGPCTerm(BinaryLatentGPCTerm):

    required_args = BinaryLatentGPCTerm.required_args + ['interest_region']

    def __init__(
        self,
        apply_proba: bool,
        apply_std: bool,
        apply_bstd: bool,
        apply_entropy: bool,
        shgo_n: int,
        shgo_iters: int,
        interest_region: Dict[str, Tuple[float, float]],
        **gpc_kwargs
    ):
        super().__init__(
            apply_proba = apply_proba,
            apply_std = apply_std,
            apply_bstd = apply_bstd,
            apply_entropy = apply_entropy,
            positive_class='interest',
            negative_class='no_interest',
            shgo_n=shgo_n,
            shgo_iters=shgo_iters,
            kernel=KERNEL,
            random_state=RANDOM_STATE,
            **gpc_kwargs
        )
        self.interest_region = interest_region

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Determine class status based on interest region appartenance of y values

        # Compute interest condition
        interest_cond = np.all(
            [
                (y[:, j] > val[0]) & (y[:, j] < val[1])
                for j, val in enumerate(self.interest_region.values())
            ],
            axis=0
        )

        # Set 'quality' column based on interest conditions
        y = np.where(
            interest_cond,
            self.class_to_index[self.negative_class],  # 0
            self.class_to_index[self.positive_class]   # 1
        )
        super().fit(X, y)


class OutlierGPCTerm(BinaryLatentGPCTerm):

    def __init__(
        self,
        apply_proba: bool,
        apply_std: bool,
        apply_bstd: bool,
        apply_entropy: bool,
        shgo_n: int,
        shgo_iters: int,
        **gpc_kwargs
    ):
        super().__init__(
            apply_proba = apply_proba,
            apply_std = apply_std,
            apply_bstd = apply_bstd,
            apply_entropy = apply_entropy,
            positive_class='inlier',
            negative_class='outlier',
            shgo_n=shgo_n,
            shgo_iters=shgo_iters,
            kernel=KERNEL,
            random_state=RANDOM_STATE,
            **gpc_kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Determine class status based on NaN presence in y
        y = np.where(
            np.isnan(y).any(axis=1),
            self.class_to_index[self.negative_class],  # 0
            self.class_to_index[self.positive_class]   # 1
        )
        super().fit(X, y)
