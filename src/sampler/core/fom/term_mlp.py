from typing import List, Tuple, Dict, Union, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from scipy.stats import skew, kurtosis

from .term_base import ModelFOMTerm, RANDOM_STATE
from ..data_processing.scalers import hypercube_linear_tent, hypercube_exponential_tent, find_sigma_for_interval, interest_probability


class MLPTerm(ModelFOMTerm):

    required_args = ['interest_region']
    fit_config = {'X_only': False, 'drop_nan': True}
    scaler_keys = ['linear_tent', 'exponential_tent', 'interest_probability']

    def __init__(
        self,
        # Config kwargs
        score_weights: Dict[str, float],
        scaler_config: Dict[str, bool],
        center_proba: float,  # interest region center score
        # kwargs required from FOM attributes 
        interest_region: Dict[str, Tuple[float, float]],
    ):
        """
        scaler_config: Dict[str, bool]
            The configuration of the scaler used to transform the surrogate
            prediction into a prior interest score in [0, 1].
        center_proba : float
            The expected probability when a sample is at the interest region center.
        """

        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.005,
            batch_size=16,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=50,
            random_state=RANDOM_STATE,
        )
        self.is_trained = False
        
        ModelFOMTerm.__init__(self, score_weights)
        
        self.interest_region = np.array(list(interest_region.values()))
        self.center_proba = center_proba
        self._validate_scaler_config(scaler_config)
        self.scaler_config = scaler_config
        self.interest_std = self.get_interest_std()

    def _validate_scaler_config(self, scaler_config):
        """
        Validate the scaler configuration.
        """
        intro = f"{self.__class__.__name__}: "

        # Check if all required keys are present
        missing_keys = [key for key in self.scaler_keys if key not in scaler_config]
        if missing_keys:
            raise ValueError(intro + f"Scaler config is missing keys: {missing_keys}")

        # Check if boolean keys are indeed boolean
        non_boolean_keys = [
            key for key in self.scaler_keys
            if not isinstance(scaler_config[key], bool)
        ]
        if non_boolean_keys:
            raise ValueError(intro + f"These keys must be boolean: {non_boolean_keys}")

        # Check if exactly one key is True
        true_keys = [key for key in self.scaler_keys if scaler_config[key]]
        if len(true_keys) != 1:
            raise ValueError(intro + f"Exactly one of these must be True: {self.scaler_keys}")

    def get_interest_std(self) -> np.ndarray:
        """
        Search for biggest sigma that makes interest probability equal to 1 when
        normal distribution mean is at the center of interest region.
        """
        if not self.scaler_config['interest_probability']:
            return None
        # Set sigma for each interval
        interest_std = [
            find_sigma_for_interval(interval_width=upper - lower, target_proba=self.center_proba)
            for lower, upper in self.interest_region
        ]
        return np.array(interest_std)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        print(f"{self.__class__.__name__} -> Fitting model...")
        
        # Validate that bounds match the target dimension
        n_dim_y = y.shape[1]
        if len(self.interest_region) != n_dim_y:
            raise ValueError(
                f"Number of bounds ({len(self.interest_region)}) "
                f"does not match the number of targets ({n_dim_y})"
            )

        self.model.fit(X, y)
        self.is_trained = True

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        y = self.model.predict(X)
        y = np.atleast_2d(y.T).T
        if self.scaler_config['linear_tent']:
            scores = hypercube_linear_tent(y, self.interest_region)
        elif self.scaler_config['exponential_tent']:
            scores = hypercube_exponential_tent(y, self.interest_region)
        elif self.scaler_config['interest_probability']:
            y_std = np.ones_like(y)*self.interest_std
            proba = interest_probability(y, y_std, self.interest_region)
            # Normalize to 1 by deviding by the maximal probability
            scores = proba / (self.center_proba ** y.shape[1])
        return np.clip(scores, 0, 1)

    def get_model_params(self):
        """
        Interpretations of loss curve statistics:
        Mean:
            The lower the mean loss value, the better the overall performance of
            the model.

        Standard Deviation:
            A high std doesn't necessarily indicate good progression.
            - Early in training, high std can indicate good progress as the
              model is learning rapidly.
            - Later in training, lower std is generally better, indicating
              stability and convergence.
            - Consistently high std throughout training might suggest
              instability or difficulty in learning.

        Skewness:
            - Positive skew is often good, indicating most losses are below
              average with some higher outliers.
            - However, the ideal skew can depend on the stage of training:
            - Early: Positive skew might be good, showing the model is finding
              better solutions.
            - Late: Near-zero skew might be preferable, indicating consistent
              performance.

        Kurtosis:
            - A kurtosis close to or slightly above 3 (which is the kurtosis of
            a normal distribution) is generally good. It indicates a stable
            learning process with some meaningful improvements.
            - Much higher kurtosis might indicate instability or overfitting.
            - Much lower kurtosis might suggest underfitting or slow learning.

        Additional considerations:
        - These statistics should be interpreted together and in the context of
          the training progress.
        - The trend of these statistics over the course of training can be more
          informative than their absolute values.
        - The nature of the problem and the specific model architecture can
          affect what's considered "good" for these metrics.
        """
        params = {}

        # Add std of prior normal distribution of prediction per target dimension
        if self.scaler_config['interest_probability']:
            for i, optimal_std in enumerate(self.interest_std):
                params[f'interest_std_{i}'] = optimal_std

        if not self.is_trained:
            return params

        loss_curve = np.array(self.model.loss_curve_)
        n_iterations = len(loss_curve)
        mid_point = n_iterations // 2
        window_size = self.model.n_iter_no_change

        early_loss = loss_curve[:mid_point]
        late_loss = loss_curve[mid_point:]

        params = {
            'n_iter': self.model.n_iter_,
            'best_loss': self.model.best_loss_,
            'final_loss': loss_curve[-1] if n_iterations > 0 else None,
            'avg_loss_early': np.mean(early_loss) if n_iterations > 0 else None,
            'avg_loss_late': np.mean(late_loss) if n_iterations > 0 else None,
            'loss_std_early': np.std(early_loss) if n_iterations > 0 else None,
            'loss_std_late': np.std(late_loss) if n_iterations > 0 else None,
            'loss_skewness': skew(loss_curve) if n_iterations > 0 else None,
            'loss_kurtosis': kurtosis(loss_curve) if n_iterations > 0 else None,
        }

        # Add validation score if early stopping was used
        if hasattr(self.model, 'best_validation_score_'):
            params['best_validation_score'] = self.model.best_validation_score_

        # Calculate sliding window statistics with non-overlapping windows
        if len(loss_curve) >= window_size:
            n_windows = len(loss_curve) // window_size
            windowed_losses = loss_curve[:n_windows * window_size].reshape(n_windows, window_size)
            windows_std = np.std(windowed_losses, axis=1)
            params['windows_std_avg'] = np.mean(windows_std)
            params['windows_std_std'] = np.std(windows_std)

        return params

    def get_parameters(self):
        params = {
            'scaler_config': self.scaler_config,
            'center_proba': self.center_proba,
        }
        params.update(self.get_model_params())

        if not self.is_trained:
            return params

        params.update({
            'scaler_config': self.scaler_config,
            'center_proba': self.center_proba,
            'n_layers': len(self.model.hidden_layer_sizes) + 2,  # including input and output layers
            'hidden_layer_sizes': self.model.hidden_layer_sizes,
            'activation': self.model.activation,
            'solver': self.model.solver,
            'alpha': self.model.alpha,
            'batch_size': self.model.batch_size,
            'learning_rate': self.model.learning_rate,
            'max_iter': self.model.max_iter,
            'early_stopping': self.model.early_stopping,
            'n_iter_no_change': self.model.n_iter_no_change,
        })
        return params
