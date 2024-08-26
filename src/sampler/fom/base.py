from typing import List, Tuple, Dict, Union, Type, Any, Optional, ClassVar
from abc import ABC, abstractmethod
import json
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


class FOM:
    def __init__(
        self,
        interest_region: Dict[str, Tuple[float, float]],
        terms_config: Dict[str, Dict[str, Union[float, str]]]
    ):
        self.interest_region = interest_region
        
        self.terms_config = terms_config
        self.terms: Dict[str, FOMTermInstance] = {}  # Only active terms
        
        self.n_samples = None
        self.n_features = None
        self.n_targets = None
        
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

        # Check if y is 1D or 2D
        if y.ndim == 1:
            n_targets = 1
        elif y.ndim == 2:
            n_targets = y.shape[1]
        else:
            raise ValueError(f"Unexpected shape for y: {y.shape}")

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_targets = n_targets

        for term in self.terms.values():
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

                # Validate scoring process
                dummy_X = X[:5]
                dummy_score = term.predict_score(dummy_X)
                term._validate_predicted_score(dummy_X, dummy_score)

    @staticmethod
    def _drop_target_nans(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Drop rows with NaN values in the target (y) and corresponding rows in
        features (X). We only check for NaNs in y because the pipeline ensures X
        doesn't contain NaNs.
        """
        # Find rows where y is not NaN
        if y.ndim == 1:
            valid_rows = ~np.isnan(y)
        elif y.ndim == 2:
            valid_rows = ~np.isnan(y).any(axis=1)
        else:
            raise ValueError(f"Unexpected shape for y: {y.shape}")
            
        # Apply the mask to both X and y
        return X[valid_rows], y[valid_rows]

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        all_scores = []
        for term in self.terms.values():
            scores = term.predict_score(X)
            if isinstance(scores, tuple):
                all_scores.extend(scores)
            else:
                all_scores.append(scores)
        return np.sum(all_scores, axis=0)

    def predict_scores_df(self, X: np.ndarray) -> pd.DataFrame:
        scores_dict = {}
        for term in self.terms.values():
            scores = term.predict_score(X)
            score_names = term.score_names
            if isinstance(scores, np.ndarray):
                scores = (scores,)  # Convert single array to tuple for consistent handling
            for score, score_name in zip(scores, score_names):
                scores_dict[score_name] = score
        return pd.DataFrame(scores_dict)

    def get_score_names(self) -> List[str]:
        score_names = []
        for term in self.terms.values():
            score_names.extend(term.score_names)
        return score_names

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {name: term.get_parameters() for name, term in self.terms.items()}
    
    def get_parameters_str(self) -> str:
        return json.dumps(
            obj=self.get_parameters(),
            default=lambda o: ' '.join(repr(o).split()),
            indent=4
        )
