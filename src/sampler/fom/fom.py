from typing import List, Tuple, Dict, Union, Type, Any, Optional, ClassVar
import numpy as np
import pandas as pd
import json

from .base import FittableFOMTerm, FOMTermAccessor, FOMTermType, FOMTermInstance
from .surrogate import SurrogateGPRTerm, OutlierGPCTerm
from .spatial import OutlierProximityTerm, SigmoidLocalDensityTerm


class FigureOfMerit:
    TERM_CLASSES: Dict[str, FOMTermType] = {
        'surrogate_gpr': SurrogateGPRTerm,
        'sigmoid_density': SigmoidLocalDensityTerm,
        'outlier_proximity': OutlierProximityTerm,
        'outlier_gpc': OutlierGPCTerm,
    }

    def __init__(
        self,
        interest_region: Dict[str, Tuple[float, float]],
        terms_config: Dict[str, Dict]
    ):
        self.interest_region = interest_region
        self.terms_config = terms_config
        self._terms: Dict[str, FOMTermInstance] = {}
        
        self.n_samples = None
        self.n_features = None
        self.n_targets = None
        
        # Set terms
        for term_name, term_args in self.terms_config.items():
            TermClass = self.TERM_CLASSES.get(term_name)
            self._validate_term(term_name, term_args, TermClass)

            if term_args.get('apply', False):
                # Set term name class attribute
                TermClass._set_term_name(term_name)

                # Add required args from self
                term_args.update({
                    arg: getattr(self, arg) for arg in TermClass.required_args
                })

                # Initialize the term instance
                self._terms[term_name] = TermClass(**term_args)

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

    def predict_scores_df(self, X: np.ndarray) -> pd.DataFrame:
        scores_dict = {}
        for term in self._terms.values():
            scores = term.predict_score(X)
            score_names = term.score_names
            if isinstance(scores, np.ndarray):
                scores = (scores,)  # Convert single array to tuple for consistent handling
            for score, score_name in zip(scores, score_names):
                scores_dict[score_name] = score
        return pd.DataFrame(scores_dict)

    def get_score_names(self) -> List[str]:
        score_names = []
        for term in self._terms.values():
            score_names.extend(term.score_names)
        return score_names

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {name: term.get_parameters() for name, term in self._terms.items()}
    
    def get_parameters_str(self) -> str:
        return json.dumps(
            obj=self.get_parameters(),
            default=lambda o: ' '.join(repr(o).split()),
            indent=4
        )
