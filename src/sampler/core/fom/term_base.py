from typing import List, Tuple, Dict, Union, Type, Any, Optional, ClassVar, Literal
from abc import ABC, abstractmethod
import time
import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic


class BaseFOMTerm(ABC):
    """
    Base class for FOM (Figure of Merit) terms.

    This abstract base class defines the structure for all FOM terms used in the
    Figure of Merit calculations.

    Class Attributes:
        required_args (List[str]):
            Required arguments to be provided by the FOM class, not from the
            JSON config. For example: ['features', 'targets',
            'interest_region']. Subclasses should override this list as needed.

    Attributes:
        predict_count (int): Number of prediction calls.
        predict_count_log (List[int]): Log of prediction counts.
        predict_cumtime (float): Cumulative time spent in predictions.
        predict_cumtime_log (List[float]): Log of cumulative prediction times.

        score_names (List[str]):
            Active score names. Defaults to ['_default'] for single-score terms.

        score_signs (Dict[str, Literal[-1, 1]]):
            Score signs, defaulting to 1 for each active score.

        score_weights (Union[Dict[str, float], float]):
            Weights for each score or a single weight.

    Notes:
        - Subclasses should override `required_args`, `score_names` and
          `score_signs` as needed.
        - `MultiScoreMixin` should override `score_names` for multi-score
          subclasses.
        - `score_signs` should be overridden for subclasses with negative
          scores.
        - Validation ensures consistency between `score_names`, `score_signs`,
          and `score_weights`.
        - Single-score terms use a numeric weight, while multi-score terms use
          dictionaries for both active scores and weights in the configuration.
        - The `BaseFOMTerm` adapts numeric weights to a single-key dict with the
          default score name.
        - The `FigureOfMerit` class later adjusts score names:
            + For single-score terms: replaces '_default' with the term name. 
            + For multi-score terms: adds the term name as a prefix to score names.
    """

    required_args: ClassVar[List[str]] = []

    @classmethod
    def _set_term_name(cls, name: str):
        """Set term name used in FOM TERM_CLASSES."""
        cls._term_name = name

    def __init__(self,
        score_weights: Union[Dict[str, float], float],
        score_names: List[str] = ['_default']
    ) -> None:
        # Initialize prediction profiling variables
        self.predict_count: int = 0
        self.predict_count_log: List[int] = []
        self.predict_cumtime: float = 0
        self.predict_cumtime_log: List[float] = []

        # Validate score names
        self._validate_names(score_names)
        self.score_names = score_names

        # Validate score signs
        self._validate_signs()

        # Validate score weights and convert to dict if float
        self._validate_weight_types(score_weights)
        self.score_weights = self._adapt_weight_keys(score_weights)

    @property
    def score_signs(self) -> Dict[str, Literal[1, -1]]:
        """
        Signs of the scores produced by subclass. By default, is a a dict of
        ones with score_names as keys.

        Note:
            Subclasses should override this property only if negative scores are
            present to replace adequatly default 1 by -1.
        """
        return {score_name: 1 for score_name in self.score_names}

    def _validate_names(self, score_names: List[str]) -> None:
        # raise error if score_names is not a list of strings
        if (
            not isinstance(score_names, list) or
            not all(isinstance(score_name, str) for score_name in score_names)
        ):
            raise ValueError(
                f"Term {self._term_name}: Invalid score_names {score_names}. "
                f"score_names must be a list"
            )

        # raise error if multi_score subclass but score_names is for single score terms
        if isinstance(self, MultiScoreMixin) and score_names == ['_default']:
            raise ValueError(
                f"Term {self._term_name}: Score names cannot be ['_default'] for "
                f"multi-score terms. Check score_names initialization in {self.__class__.__name__}."
            )

        # raise error if subclass of MultiScoreMixin does not have all_scores class attribute

    def _validate_signs(self) -> None:
        # raise error if score_signs is not a list of 1 or -1
        if not all(score_sign in [1, -1] for score_sign in self.score_signs.values()):
            raise ValueError(
                f"Term {self._term_name}: Invalid score_signs {self.score_signs}. "
                f"score_signs values must be either 1 or -1"
            )
        # raise error if score_signs has not keys as score_names
        if list(self.score_signs) != self.score_names:
            raise ValueError(
                f"Term {self._term_name}: Signs keys do not match score_names. "
                f"Expected: {self.score_names}, Got: {list(self.score_signs.keys())}"
            )

    def _validate_weight_types(self, score_weights: Union[Dict[str, float], float]) -> None:
        """
        Validate that score weights are matching score name.

        Steps:
        1. Check if weights is a dictionary of numbers for multi-score terms,
           or a float for single-default-score terms.
        2. Checks if weights keys match names.
        """
        term_name = self._term_name

        # If sublass term can have multiple scores
        if isinstance(self, MultiScoreMixin):
            if not isinstance(score_weights, dict):
                raise TypeError(f"Term {term_name}: Score weights must be a dictionary for multiple scores")

            if not all(isinstance(v, (int, float)) for v in score_weights.values()):
                raise TypeError(f"Term '{term_name}': all weight values must be a number")

        # If only one score, score_weights must be a number
        else:
            if not isinstance(score_weights, (int, float)):
                raise TypeError(f"Term {self._term_name}: Weights must be a number for single score terms")

    def _adapt_weight_keys(self, score_weights: Union[Dict[str, float], float]) -> Dict[str, float]:
        
        # If sublass term can have multiple scores
        if isinstance(self, MultiScoreMixin):
            # Check if weight keys match score_names order
            if list(score_weights.keys()) != self.all_scores:
                raise ValueError(
                    f"Term {self._term_name}: Weights keys do not match score_names. "
                    f"Expected: {self.score_names}, Got: {list(score_weights.keys())}"
                )

            # Keep only active score weights
            score_weights = {
                score_name: score_weights[score_name] for score_name in self.score_names
            }

        # If score_weights is a number (already checked its type)
        else:
            score_weights = {self.score_names[0]: score_weights}

        return score_weights

    def _validate_score_shape(self,
        X: np.ndarray, 
        predicted_score: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> None:
        """ Validate the shape and type of the predicted score(s). """
        expected_shape = (X.shape[0],)
        score_names = self.score_names
        n_scores = len(score_names)

        if n_scores == 1:
            # Check if an array
            if not isinstance(predicted_score, np.ndarray):
                raise ValueError(
                    f"Term {self._term_name}: "
                    f"Expected np.ndarray for single score, "
                    f"got {type(predicted_score)}"
                )
            # Check if 1D and same size as X
            if predicted_score.shape != expected_shape:
                raise ValueError(
                    f"Term {self._term_name}: "
                    f"Expected shape {expected_shape} for score '{score_names[0]}', "
                    f"got {predicted_score.shape}"
                )
        else:
            # Check if a tuple
            if not isinstance(predicted_score, tuple):
                raise ValueError(
                    f"Term {self._term_name}: "
                    f"Expected tuple of np.ndarray for {n_scores} scores, "
                    f"got {type(predicted_score)}"
                )
            # Check if right number of arrays
            if len(predicted_score) != n_scores:
                raise ValueError(
                    f"Term {self._term_name}: "
                    f"Expected {n_scores} arrays (one for each of {score_names}), "
                    f"got {len(predicted_score)}"
                )
            # Check if arrays
            for score_name, score in zip(score_names, predicted_score):
                if not isinstance(score, np.ndarray):
                    raise ValueError(
                        f"Term {self._term_name}: "
                        f"Expected np.ndarray for score '{score_name}', "
                        f"got {type(score)}"
                    )
                # Check if 1D and same size as X
                if score.shape != expected_shape:
                    raise ValueError(
                        f"Term {self._term_name}: "
                        f"Expected shape {expected_shape} for score '{score_name}', "
                        f"got {score.shape}"
                    )

    def _validate_score_sign(self,
        predicted_score: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> None:
        """ Check if the predicted scores has the correct sign. """
        # Tolerance of 10% exceeding 0 limit
        tol = 0.1

        # Convert single score to tuple (note: scores are tuple of 1D arrays)
        if not isinstance(predicted_score, tuple):
            predicted_score = (predicted_score,)

        for (score_name, score_sign), score_value in zip(self.score_signs.items(), predicted_score):

            # Check if score has right positive sign
            if score_sign == 1:
                if np.any(score_value < 0 - tol):
                    raise ValueError(
                        f"Term {self._term_name}: "
                        f"Expected positive score '{score_name}', "
                        f"but found negative values: \n{score_value}"
                    )

            # Check if score has right negative sign
            else:
                if np.any(score_value > 0 + tol):
                    raise ValueError(
                        f"Term {self._term_name}: "
                        f"Expected negative score '{score_name}', "
                        f"but found positive values: \n{score_value}"
                    )

    @abstractmethod
    def predict_score(self, X: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Calculate and return the score(s) for this FOM term.
        
        Returns:
            Union[np.ndarray, Tuple[np.ndarray, ...]]: 
                - A single np.ndarray of shape (n_samples,) for a single score
                  per sample
                - A tuple of np.ndarray, each of shape (n_samples,), for
                  multiple scores per sample
        
        Note:
            - If multiple scores are returned, they should be in a consistent
              order with self.score_names property.
            - The score must be either positive in [0, 1] or negative in [-1, 0].
            - Score always verifies greater-is-better.
            - A score is negative when it is more natural to think about it as a
              loss, but the metric still follows the "greater-is-better"
              principle. For example, outlier proximity is 0 for good regions
              and -1 for outlier regions. So the only purpose of negativity is
              a more natural interpretation of the score.
        """
        raise NotImplementedError("Subclasses must implement method")

    def predict_score_with_log(self, X: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Wrapper method that calls predict_score and updates cumulative prediction time.
        """
        start_time = time.perf_counter()
        score = self.predict_score(X)
        end_time = time.perf_counter()

        self.predict_count += 1
        self.predict_cumtime += end_time - start_time

        return score

    def reset_predict_profiling(self) -> None:
        """
        Resets the cumulative prediction time to zero and logs the current value.
        """
        # Log predictions count and time
        self.predict_count_log.append(self.predict_count)
        self.predict_cumtime_log.append(self.predict_cumtime)
        # Reset counters
        self.predict_count = 0
        self.predict_cumtime = 0.0

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieve the fitted parameters of this FOM term.

        This method is intentionally named 'get_parameters' instead of 'get_params'
        to avoid conflicts with scikit-learn's API. The 'get_params' method in 
        scikit-learn estimators is used for hyperparameter management and is called 
        with a 'deep' argument. Overriding it can lead to compatibility issues.

        By using a distinct name, we prevent accidental overriding of scikit-learn's 
        'get_params' method in child classes that may inherit from both this base 
        class and scikit-learn estimators. This ensures that scikit-learn's internal 
        mechanisms for parameter management remain intact.
        """
        params = {}
        if isinstance(self, MultiScoreMixin):
            params['score_names'] = self.score_names
            params['weights'] = self.score_weights
        else:
            params['weights'] = next(iter(self.score_weights.values()))
            
        return params


class NonFittableFOMTerm(BaseFOMTerm):
    """
    A class representing non-fittable FOM (Figure of Merit) terms.

    This class extends BaseFOMTerm for terms that do not require fitting.
    It inherits all functionality from BaseFOMTerm without adding any
    additional methods or attributes.

    Non-fittable terms are typically used for static calculations or
    predefined metrics that don't need to consider global data.
    """

    def __init__(self,
        score_weights: Union[Dict[str, float], float],
        score_names: List[str] = ['_default']
    ) -> None:
        super().__init__(score_weights, score_names)


class FittableFOMTerm(BaseFOMTerm):
    """
    An abstract base class for fittable FOM (Figure of Merit) terms.

    This class extends BaseFOMTerm to include fitting functionality.
    Subclasses must implement the fit method and define their own fit_config.

    Class Attributes:
        - fit_config (Dict[str, Optional[bool]]): A dictionary of
        fitting configuration. Subclasses must override this with their specific
        configuration. Default values are set to None and will raise an error if
        not properly defined.
        - dependencies (List[str]): List of term names on which this subclass
        depends. The subclass can access specific attributes from these terms
        during its fitting process. When a dependency is not present in
        fom.terms (not applied), it's up to the subclass to decide whether to
        raise an error or handle the missing dependency gracefully.

    Attributes:
        Inherits all attributes from BaseFOMTerm.
    """
    fit_config: ClassVar[Dict[str, Optional[bool]]] = {'X_only': None, 'drop_nan': None}
    dependencies: ClassVar[List[str]] = []

    def __init__(self,
        score_weights: Union[Dict[str, float], float],
        score_names: List[str] = ['_default']
    ) -> None:
        super().__init__(score_weights, score_names)
        self.fit_time_log: List[float] = []
        self._validate_fit_config()

    @classmethod
    def _validate_fit_config(cls) -> None:
        """
        Validate that fit_config are properly defined in the subclass.
        Raises a ValueError if any parameter is None.
        """
        for param, value in cls.fit_config.items():
            if value is None:
                raise AttributeError(
                    f"FittableFOMTerm subclass '{cls.__name__}' is missing a "
                    f"required parameter '{param}' in fit_config. All "
                    "parameters in fit_config must be explicitly set to "
                    "boolean values."
                )

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Fit the FOM term to the provided data. """
        pass

    def fit_with_log(self, **kwargs) -> None:
        """ Fit term with time log"""
        start_time = time.time()
        self.fit(**kwargs)
        self.fit_time_log.append(time.time() - start_time)


class ModelFOMTerm(FittableFOMTerm):
    """
    An abstract base class for fittable model-based FOM (Figure of Merit) terms.

    This class extends FittableFOMTerm to implement specific terms that actually 
    train a model, unlike FittableFOMTerm where subclasses might use `fit` just 
    to store training data for easier access in `predict_score`. This class 
    introduces a `model` attribute to be fitted and a `get_model_params` method 
    for retrieving fitted parameters. Both of these must be implemented in 
    subclasses.

    Class Attributes:
        Inherits all class attributes from FittableFOMTerm, including
        fit_config.
        model (Any): ML model to fit.

    Methods:
        get_model_params: Abstract method to retrieve fitted parameters.
    """

    def __init__(self,
        score_weights: Union[Dict[str, float], float],
        score_names: List[str] = ['_default']
    ) -> None:
        super().__init__(score_weights, score_names)
        self._validate_model_attrs()
    
        # Subclass must contain following attributes for consistency
        # self.model
        # self.is_trained

    def _validate_model_attrs(self) -> None:
        if not hasattr(self, 'model') or not hasattr(self, 'is_trained'):
            raise AttributeError(
                f"ModelFOMTerm subclass '{self.__class__.__name__}' must "
                "contain a 'model' and 'is_trained' attributes"
            )

    @abstractmethod
    def get_model_params(self) -> Dict[str, float]:
        """ Retrieve the fitted parameters of the model. """
        pass


class MultiScoreMixin():
    """ A mixin class for FOM terms that enables multi-scoring. """

    all_scores: ClassVar[List[str]]

    def __init__(self, score_config: Dict[str, bool]) -> None:
        self._validate_instance()
        self._validate_score_config(score_config)
        self.score_config = score_config

    def _validate_instance(self):
        error_intro = f"Term {self.__class__.__name__}: "

        # Check if all_scores is defined
        if not hasattr(self, 'all_scores'):
            raise ValueError(error_intro + "all_scores must be defined")

        # Check if all_scores is a list of strings
        if (
            not isinstance(self.all_scores, list) or
            not all(isinstance(name, str) for name in self.all_scores)
        ):
            raise ValueError(
                error_intro + "all_scores must be a list of strings, "
                f"but got {self.all_scores}"
            )

    def _validate_score_config(self, score_config: Dict[str, bool]):
        error_intro = f"Term {self.__class__.__name__}: "

        # Check that all expected names are provided
        expected_names = set(self.all_scores)
        provided_names = set(score_config.keys())

        error_msg = []
        if expected_names != provided_names:
            missing = expected_names - provided_names
            extra = provided_names - expected_names
            if missing:
                error_msg.append(f"Missing score names: {', '.join(missing)}")
            if extra:
                error_msg.append(f"Unexpected score names: {', '.join(extra)}")
            raise ValueError(error_intro + ". ".join(error_msg))

        # Check if all_scores and score_config have same order
        if self.all_scores != list(score_config.keys()):
            raise ValueError(error_intro + "all_scores and score_config must have the same order")

        # Check that at least one score is applied
        if not any(score_config.values()):
            raise ValueError(error_intro + "At least one score must be applied")

    def get_active_scores(self) -> List[str]:
        return [name for name, applied in self.score_config.items() if applied]


# Create tuple of possible FOM term classes
FOMTermClasses = (FittableFOMTerm, ModelFOMTerm, NonFittableFOMTerm)

# Create custom type for instances
FOMTermInstance = Union[FittableFOMTerm, ModelFOMTerm, NonFittableFOMTerm]

# Create custom type for classes
FOMTermType = Type[FOMTermInstance]

# Kernel of Gaussian Process model
RANDOM_STATE = 42
BANDWIDTH_BOUNDS = (1e-5, 1.0)
KERNELS = {
    'RBF': RBF(
        length_scale=0.5,
        length_scale_bounds=BANDWIDTH_BOUNDS
    ),
    'RQ': RationalQuadratic(
        length_scale=0.5,
        alpha=1.0,
        length_scale_bounds=BANDWIDTH_BOUNDS,
        alpha_bounds=(0.1, 10.0)
    )
}
KernelInstance = Union[RBF, RationalQuadratic]


"""
For the implementation of the Gaussian Process Regression model utilizing the
RationalQuadratic kernel, parameter boundaries were carefully selected to align
with the normalized feature space [0, 1]. The RationalQuadratic kernel is
defined as:

        k(x_i, x_j) = (1 + d(x_i, x_j)^2 / (2 * alphal**2))**-alpha

where d(x_i, x_j) is the Euclidean distance between two points, l is the length
scale, and alpha is the scale mixture parameter.

Given the normalized feature space, the following parameter bounds were
established:

Length scale (l):
    Lower bound: 0.01
    Upper bound: 2.0

The lower bound represents 1% of the feature range, allowing the model to
capture fine-grained patterns, while the upper bound of twice the feature range
permits the identification of smooth, global trends across the entire input
space.

Scale mixture parameter (alpha):
    Lower bound: 0.1
    Upper bound: 10.0

This range for alpha facilitates a balanced exploration of both large-scale and
small-scale variations in the data, while mitigating the risk of numerical
instability that could arise from extreme values. The initial values for
length_scale and alpha were set to 0.5 and 1.0 respectively, providing a neutral
starting point for optimization within the defined bounds.


Comparative Analysis: RBF vs. Rational Quadratic (RQ) Kernels

1. Functional Form:
   - RBF: k(x, x') = exp(-0.5 * r^2 / l^2), where r^2 = ||x - x'||^2
   - RQ:  k(x, x') = (1 + r^2 / (2αl^2))^(-α), where α > 0 is the scale mixture parameter

2. Theoretical Properties:
   - RBF: Implies infinitely differentiable functions, characterized by a single length scale.
   - RQ:  Equivalent to a scale mixture of RBF kernels with different length scales, 
          allowing for multi-scale modeling.

3. Spectral Density:
   - RBF: S(s) ∝ exp(-2π^2 l^2 s^2), exhibiting rapid decay.
   - RQ:  S(s) ∝ (1 + 2π^2 l^2 s^2 / α)^(-α - d/2), with heavier tails.

4. Hyperparameter Space:
   - RBF: Θ = {l}, one-dimensional optimization problem.
   - RQ:  Θ = {l, α}, two-dimensional optimization, potentially more complex.

5. Computational Complexity:
   - Both have O(n^2) complexity for n data points, but RBF is generally more efficient 
     due to simpler calculations.

6. Model Capacity:
   - RBF: Uniform smoothness across the input space.
   - RQ:  Adaptive smoothness, capable of capturing both short and long-range variations.
"""
