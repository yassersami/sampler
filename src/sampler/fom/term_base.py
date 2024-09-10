from typing import List, Tuple, Dict, Union, Type, Any, Optional, ClassVar, Literal
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import RationalQuadratic


class BaseFOMTerm(ABC):
    """
    Base class for FOM (Figure of Merit) terms.

    This abstract base class defines the structure for all FOM terms used in the
    Figure of Merit calculations.

    Class Attributes:
        - required_args (List[str]): List of required argument names that should
        be provided by the FOM class, not from the JSON config. For example:
        ['features', 'targets', 'interest_region']. Subclasses should override
        this list as needed.
    """

    required_args: ClassVar[List[str]] = []

    @classmethod
    def _set_term_name(cls, name: str):
        """Set term name used in FOM TERM_CLASSES."""
        cls._term_name = name

    @property
    def score_names(self) -> List[str]:
        """
        Return the names of the scores produced by this term.
        
        By default, it returns a list with the term name.
        Override this property in subclasses if you need multiple score_names.
        """
        if self._term_name is None:
            raise AttributeError("This FOM term has not been registered with a name.")
        return [self._term_name]

    @property
    def score_signs(self) -> List[Literal[1, -1]]:
        """
        Return the signs of the scores produced by this term.
        
        By default, it returns a list with ones like the score_names.
        Override this property in subclasses if negative scores are present.
        """
        return [1] * len(self.score_names)

    def _validate_score_signs(self):
        term_score_signs = self.score_signs
        # raise error if score_signs has not the same length as score_names
        if len(term_score_signs) != len(self.score_names):
            raise ValueError(
                f"{self._term_name} term has an invalid score_signs length "
                f"{len(term_score_signs)}"
            )

    def _validate_predicted_score(
        self,
        X: np.ndarray, 
        predicted_score: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> None:
        """Validate the shape and type of the predicted score(s)."""
        expected_shape = (X.shape[0],)
        score_names = self.score_names
        n_scores = len(score_names)

        if n_scores == 1:
            if not isinstance(predicted_score, np.ndarray):
                raise ValueError(
                    f"Expected np.ndarray for single score '{score_names[0]}', "
                    f"got {type(predicted_score)}"
                )
            if predicted_score.shape != expected_shape:
                raise ValueError(
                    f"Expected shape {expected_shape} for score '{score_names[0]}', "
                    f"got {predicted_score.shape}"
                )
        else:
            if not isinstance(predicted_score, tuple):
                raise ValueError(
                    f"Expected tuple of np.ndarray for {n_scores} scores, "
                    f"got {type(predicted_score)}"
                )
            if len(predicted_score) != n_scores:
                raise ValueError(
                    f"Expected {n_scores} arrays (one for each of {score_names}), "
                    f"got {len(predicted_score)}"
                )
            for score_name, score in zip(score_names, predicted_score):
                if not isinstance(score, np.ndarray):
                    raise ValueError(
                        f"Expected np.ndarray for score '{score_name}', "
                        f"got {type(score)}"
                    )
                if score.shape != expected_shape:
                    raise ValueError(
                        f"Expected shape {expected_shape} for score '{score_name}', "
                        f"got {score.shape}"
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
            - The score must be either in [0, 1] or [-1, 0].
            - Score always verifies greater is better.  
        """
        pass


    @abstractmethod
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
        pass


class FittableFOMTerm(BaseFOMTerm):
    """
    An abstract base class for fittable FOM (Figure of Merit) terms.

    This class extends BaseFOMTerm to include fitting functionality.
    Subclasses must implement the fit method and define their own fit_config.

    Class Attributes:
        - fit_config (ClassVar[Dict[str, Optional[bool]]]): A dictionary of
        fitting configuration. Subclasses must override this with their specific
        configuration. Default values are set to None and will raise an error if
        not properly defined.

    Attributes:
        Inherits all attributes from BaseFOMTerm.
    """
    fit_config: ClassVar[Dict[str, Optional[bool]]] = {'X_only': None, 'drop_nan': None}

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


class ModelFOMTerm(FittableFOMTerm):
    """
    An abstract base class for fittable model-based FOM (Figure of Merit) terms.

    This class extends FittableFOMTerm to specific terms that actually train a
    model, unlike FittableFOMTerm where subclasses could use `fit` just to store
    train data for easier access in `predict_score`. This class includes a
    method for retrieving fitted parameters. Subclasses must implement the
    `get_model_params` method in addition to the fit method inherited from
    FittableFOMTerm.

    Class Attributes:
        Inherits all class attributes from FittableFOMTerm, including
        fit_config.

    Methods:
        get_model_params: Abstract method to retrieve fitted parameters.
    """

    @abstractmethod
    def get_model_params(self) -> Dict[str, float]:
        """ Retrieve the fitted parameters of the model. """
        pass


class NonFittableFOMTerm(BaseFOMTerm):
    """
    A class representing non-fittable FOM (Figure of Merit) terms.

    This class extends BaseFOMTerm for terms that do not require fitting.
    It inherits all functionality from BaseFOMTerm without adding any
    additional methods or attributes.

    Non-fittable terms are typically used for static calculations or
    predefined metrics that don't need to be trained on data.
    """


# Create custom type for instances
FOMTermInstance = Union[FittableFOMTerm, ModelFOMTerm, NonFittableFOMTerm]

# Create custom type for classes
FOMTermType = Type[FOMTermInstance]

# Kernel of Gaussian Process model
RANDOM_STATE = 42
KERNEL = RationalQuadratic(
    length_scale=0.5,
    alpha=1.0, 
    length_scale_bounds=(0.01, 2.0),  # (1e-5, 10)
    alpha_bounds=(0.1, 10.0)
)
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
"""
