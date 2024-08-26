from typing import List, Tuple, Dict, Union, Type, Any, Optional, ClassVar
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


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

    Attributes:
        - apply (bool): Whether this term is active in FOM calculations. Set
        during initialization based on the pipeline JSON configuration.

    """

    required_args: ClassVar[List[str]] = []

    def __init__(self, apply: bool):
        self.apply = apply
    
    @classmethod
    def _set_registry_name(cls, name: str):
        """Set the registry name for this FOM term."""
        cls._registry_name = name
    
    @property
    def score_names(self) -> List[str]:
        """
        Return the names of the scores produced by this term.
        
        By default, it returns a list with the registry name.
        Override this property in subclasses if you need multiple score_names.
        """
        if self._registry_name is None:
            raise ValueError("This FOM term has not been registered with a name.")
        return [self._registry_name]

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
            - The score must be in [0, 1], where greater is better.  
        """
        pass


    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """ Retrieve the parameters of this FOM term. """
        pass


class FittableFOMTerm(BaseFOMTerm):
    """
    An abstract base class for fittable FOM (Figure of Merit) terms.

    This class extends BaseFOMTerm to include fitting functionality.
    Subclasses must implement the fit method and define their own fit_params.

    Class Attributes:
        - fit_params (ClassVar[Dict[str, Optional[bool]]]): A dictionary of
        fitting parameters. Subclasses must override this with their specific
        parameters. Default values are set to None and will raise an error if
        not properly defined.

    Attributes:
        Inherits all attributes from BaseFOMTerm.
    """
    fit_params: ClassVar[Dict[str, Optional[bool]]] = {'X_only': None, 'drop_nan': None}
    
    def __init__(self, apply: bool):
        super().__init__(apply=apply)

    @classmethod
    def _validate_fit_params(cls) -> None:
        """
        Validate that fit_params are properly defined in the subclass.
        Raises a ValueError if any parameter is None.
        """
        for param, value in cls.fit_params.items():
            if value is None:
                raise AttributeError(
                    f"FittableFOMTerm subclass '{cls.__name__}' is missing a "
                    f"required parameter '{param}' in fit_params. All "
                    "parameters in fit_params must be explicitly set to "
                    "boolean values."
                )

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Fit the FOM term to the provided data. """
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
    def __init__(self, apply: bool):
        super().__init__(apply=apply)


# Create custom type for classes
FOMTermType = Union[Type[FittableFOMTerm], Type[NonFittableFOMTerm]]

# Create custom type for instances
FOMTermInstance = Union[FittableFOMTerm, NonFittableFOMTerm]


class FOMTermRegistry:
    _registry: Dict[str, FOMTermType] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(term_class: FOMTermType) -> FOMTermType:
            cls._registry[name] = term_class
            term_class._set_registry_name(name)
            return term_class
        return decorator

    @classmethod
    def get_term(cls, name: str) -> FOMTermType:
        return cls._registry.get(name)
