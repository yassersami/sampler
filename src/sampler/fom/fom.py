from typing import List, Tuple, Dict, Union, Type, Any, Optional, ClassVar
import numpy as np
import pandas as pd
import json
import inspect

from .term_base import FittableFOMTerm, NonFittableFOMTerm, FOMTermType, FOMTermInstance
from .term_gpr import SurrogateGPRTerm
from .term_gpc import InterestGPCTerm, OutlierGPCTerm
from .term_spatial import OutlierProximityTerm, SigmoidLocalDensityTerm


class FOMTermAccessor:
    """
    This class enables accessing terms in FigureOfMerit.terms.term_name with
    autocompletion and provides methods to access term classes.
    """

    surrogate_gpr: SurrogateGPRTerm
    interest_gpc: InterestGPCTerm
    sigmoid_density: SigmoidLocalDensityTerm
    outlier_proximity: OutlierProximityTerm
    outlier_gpc: OutlierGPCTerm

    def __init__(self, terms: Dict[str, FOMTermInstance]):
        self._terms = terms

    def __getattr__(self, name: str) -> FOMTermInstance:
        if name not in self._terms:
            raise AttributeError(f"Term '{name}' is not active or does not exist.")
        return self._terms[name]

    @classmethod
    def is_valid_term_class(cls, term_class: FOMTermType) -> bool:
        """
        Checks if a given input is a valid FOM term class.
        """
        return (
            isinstance(term_class, type) and
            issubclass(term_class, (FittableFOMTerm, NonFittableFOMTerm))
        )

    @classmethod
    def is_valid_term_name(cls, term_name: str) -> bool:
        """
        Checks if a given term name is defined and is a valid FOM term class.
        """
        if term_name not in cls.__annotations__:
            return False
        term_class = cls.__annotations__[term_name]
        return cls.is_valid_term_class(term_class)

    @classmethod
    def get_term_class(cls, term_name: str) -> FOMTermType:
        """ Returns the class for a given term name. """
        if not cls.is_valid_term_name(term_name):
            raise ValueError(f"'{term_name}' is not a valid FOM term class.")
        return cls.__annotations__[term_name]

    @classmethod
    def get_term_classes(cls) -> Dict[str, FOMTermType]:
        """
        Returns a dictionary of term names and their corresponding classes.
        """
        return {
            term_name: term_class
            for term_name, term_class in cls.__annotations__.items()
            if cls.is_valid_term_class(term_class)
        }


class FigureOfMerit:

    def __init__(
        self,
        interest_region: Dict[str, Tuple[float, float]],
        terms_config: Dict[str, Dict]
    ):
        self.interest_region = interest_region
        self.terms_config = terms_config.copy()
        self._terms: Dict[str, FOMTermInstance] = {}

        self.n_samples = None
        self.n_features = None
        self.n_targets = None
        
        # Set terms
        for term_name, term_args in terms_config.items():
            TermClass = FOMTermAccessor.get_term_class(term_name)
            self._validate_term(term_name, term_args, TermClass)
            apply = term_args.pop('apply')

            if apply:
                # Set term name class attribute
                TermClass._set_term_name(term_name)

                # Add required args from self
                term_args.update({
                    arg: getattr(self, arg) for arg in TermClass.required_args
                })

                try:
                    self._terms[term_name] = TermClass(**term_args)
                except Exception as e:
                    raise type(e)(f"Error instantiating term '{term_name}': {str(e)}") from e

        # Create the accessor
        self.terms = FOMTermAccessor(self._terms)

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

        for term in self._terms.values():
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
        for term in self._terms.values():
            scores = term.predict_score(X)
            if isinstance(scores, tuple):
                all_scores.extend(scores)
            else:
                all_scores.append(scores)
        return np.sum(all_scores, axis=0)

    def predict_loss(self, X: np.ndarray) -> float:
        """
        A custom translated opposite to enable a smaller-is-better objective
        where smallest best value is 0.
        """
        return self.n_positive_scores - self.predict_score(X)

    def predict_scores_df(self, X: np.ndarray) -> pd.DataFrame:
        scores_dict = {}
        for term in self._terms.values():
            scores = term.predict_score(X)
            score_names = term.score_names
            if isinstance(scores, np.ndarray):
                scores = (scores,)  # Convert single array to tuple for consistent handling
            for score, score_name in zip(scores, score_names):
                scores_dict[score_name] = score

        df_scores = pd.DataFrame(scores_dict)
        self._validate_n_negative_scores(df_scores)

        return df_scores
    
    def _validate_n_negative_scores(self, df_scores: pd.DataFrame) -> None:
        """
        Validate if the number of columns with negative or null values matches
        the expected count.

        Logic:
        1- Negative or null columns include negative scores but can also include
           positive scores with only null values.
        2- So number of negative or null columns is greater than negative scores.
        3- But if it is not true then raise error, because number of expected
           negative scores is too high.
        """
        expected_negative_columns = self.n_negative_scores

        # Count columns with only negative or null values
        columns_with_negative_or_null = ((df_scores <= 0).any()).sum()

        if columns_with_negative_or_null < expected_negative_columns:
            raise ValueError(
                f"Mismatch in negative scores count. "
                f"Expected {expected_negative_columns} columns with negative/null values, "
                f"but found {columns_with_negative_or_null}."
            )

    @property
    def score_names(self) -> List[str]:
        score_names = []
        for term in self._terms.values():
            score_names.extend(term.score_names)
        return score_names

    @property
    def n_negative_scores(self) -> int:
        """
        Count number of negative active scores.

        Note:
        - Some terms can return multiple scores.
        - Only active terms (stored in self._terms) are considered.
        - 'outlier_proximity', if present, is treated as a term with negative
        score(s).
        """
        negative_term_names = ['outlier_proximity']
        count_negative_scores = 0

        for term_name in negative_term_names:
            # Check if term is well defined
            if not FOMTermAccessor.is_valid_term_name(term_name):
                raise AttributeError(f"Class attribute TERM_CLASSES is missing '{term_name}' term")

            # Check if term is active and count its negative scores
            if term_name in self._terms.keys() :
                count_negative_scores += len(self._terms[term_name].score_names)

        return count_negative_scores

    @property
    def n_positive_scores(self) -> int:
        # Count number of scores of all active terms
        count_scores = len(self.score_names)

        # Subtract number of negative scores of active terms
        return count_scores - self.n_negative_scores

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {name: term.get_parameters() for name, term in self._terms.items()}
    
    def get_parameters_str(self) -> str:
        return json.dumps(
            obj=self.get_parameters(),
            default=lambda o: ' '.join(repr(o).split()),
            indent=4
        )
