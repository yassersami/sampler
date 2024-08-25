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

    @abstractmethod
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate and return the score for this FOM term.
        The score must be in [0, 1], where greater is better.  
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
            return term_class
        return decorator

    @classmethod
    def get_term(cls, name: str) -> FOMTermType:
        return cls._registry.get(name)


class FOM:
    def __init__(
        self,
        features: List[str],
        targets: List[str],
        interest_region: Dict[str, Tuple[float, float]],
        terms_config: Dict[str, Dict[str, Union[float, str]]]
    ):
        self.features = features
        self.targets = targets
        self.interest_region = interest_region
        
        self.terms_config = terms_config
        self.terms: Dict[str, FOMTermInstance] = {}
        
        # Set terms
        for term_name, term_args in self.terms_config.items():
            TermClass = FOMTermRegistry.get_term(term_name)
            self._validate_term(term_name, term_args, TermClass)
            
            if term_args['apply']:
                # Add required args from self
                term_args.update({
                    arg: getattr(self, arg) for arg in TermClass.required_args
                })

                # Initialize the term instance
                self.terms[term_name] = TermClass(**term_args)

    def _validate_term(
        self, term_name: str, term_args: Dict[str, Union[float, str]],
        TermClass: Union[FOMTermType, None]
    ) -> None:
        """
        Validate the arguments for a FOM term.

        This method checks if:
        1. The 'apply' parameter is present in term_args.
        2. The TermClass is not None if the term is to be used.
        3. All required arguments for the term are available in the FOM instance.
        4. If FittableFOMTerm, check fit_parms are all booleans.
        
        Raises:
            ValueError: If any validation check fails.
        """
        if 'apply' not in term_args:
            raise ValueError(
                f"Term '{term_name}' is missing the required 'apply' parameter"
            )
        if term_args['apply']:
            # Check if TermClass exists
            if TermClass is None:
                raise ValueError(f"Unknown term: {term_name}")

            # Check if required args from FOM are available
            for arg in TermClass.required_args:
                if not hasattr(self, arg):
                    raise ValueError(
                        f"Required argument '{arg}' for term '{term_name}' is "
                        "not available in FOM"
                    )

            # If fittable term, check if fit_params are well defined
            if issubclass(TermClass, FittableFOMTerm):
                TermClass._validate_fit_params()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for term_name, term in self.terms.items():
            if isinstance(term, FittableFOMTerm):
                fit_params = term.fit_params
                
                # Adapt input data for fittable term
                if fit_params['X_only']:
                    term.fit(X)
                elif fit_params['drop_nan']:
                    X_clean, y_clean = self._drop_target_nans(X, y)
                    term.fit(X_clean, y_clean)
                else:
                    term.fit(X, y)

    def _drop_target_nans(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Drop rows with NaN values in the target (y) and corresponding rows in
        features (X). We only check for NaNs in y because the pipeline ensures X
        doesn't contain NaNs.
        """
        # Find rows where y is not NaN
        valid_rows = ~np.isnan(y).any(axis=1)
        
        # Apply the mask to both X and y
        X_clean = X[valid_rows]
        y_clean = y[valid_rows]
        
        return X_clean, y_clean

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        scores = [
            term.predict_score(X) for term in self.terms.values()
            if term.apply
        ]
        return np.sum(scores, axis=0)

    def predict_scores_df(self, X: np.ndarray) -> pd.DataFrame:
        scores = {
            name: term.predict_score(X) for name, term in self.terms.items()
            if term.apply
        }
        return pd.DataFrame(scores)

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {name: term.get_parameters() for name, term in self.terms.items()}
