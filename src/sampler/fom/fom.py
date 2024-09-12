from typing import List, Tuple, Dict, Union, Literal, Any, Iterator
import numpy as np
import pandas as pd
import json

from .term_base import FittableFOMTerm, ModelFOMTerm, FOMTermType
from .term_accessor import FOMTermAccessor


class FigureOfMerit:

    def __init__(
        self,
        interest_region: Dict[str, Tuple[float, float]],
        terms_config: Dict[str, Dict]
    ):
        self.interest_region = interest_region
        self.terms_config = terms_config.copy()

        # Set terms
        terms_dict = {}
        for term_name, term_args in terms_config.items():
            TermClass = FOMTermAccessor.get_term_class(term_name)
            self._validate_term(term_name, term_args, TermClass)
            apply = term_args.pop('apply')

            if apply:
                # Set term name class attribute
                TermClass._set_term_name(term_name)

                # Add required args from FOM attributes
                term_args.update({
                    arg: getattr(self, arg) for arg in TermClass.required_args
                })

                # Instantiate term
                term_instance = TermClass(**term_args)

                # Validate term sign
                term_instance._validate_score_signs()

                terms_dict[term_name] = term_instance

        # Create the accessor
        self.terms = FOMTermAccessor(terms_dict)

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
        4. If FittableFOMTerm, check fit_config are all booleans.

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

            # If fittable term, check if fit_config is well defined
            if issubclass(TermClass, FittableFOMTerm):
                TermClass._validate_fit_config()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        # Check if y is 1D or 2D
        if y.ndim > 2:
            raise ValueError(f"Unexpected shape for y: {y.shape}")

        for term in self.terms.values():
            # Reset term prediction profiling
            term.reset_predict_profiling()

            if isinstance(term, FittableFOMTerm):
                fit_config = term.fit_config
                fit_args = {}

                # Prepare arguments based on fit_config
                if fit_config['X_only']:
                    fit_args['X'] = X
                elif fit_config['drop_nan']:
                    X_clean, y_clean = self._drop_target_nans(X, y)
                    fit_args['X'] = X_clean
                    fit_args['y'] = y_clean
                else:
                    fit_args['X'] = X
                    fit_args['y'] = y

                # Add term dependencies if necessary
                if len(term.dependencies) > 0:
                    fit_args.update({dep: self.terms[dep] for dep in term.dependencies})

                # Call the fit method with prepared arguments
                term.fit(**fit_args)

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

    def predict_loss(self, X: np.ndarray) -> float:
        """
        A custom translated opposite to enable a smaller-is-better objective
        where smallest best value is 0.
        """
        return self.score_signs.count(1) - self.predict_score(X)

    def get_scores_df(self, X: np.ndarray) -> pd.DataFrame:
        scores_dict = {}

        for term in self.terms.values():
            # Get scores
            scores = term.predict_score(X)
            score_names = term.score_names
            if isinstance(scores, np.ndarray):
                scores = (scores,)  # Convert single array to tuple for consistent handling
            for score, score_name in zip(scores, score_names):
                scores_dict[score_name] = score

        df_scores = pd.DataFrame(scores_dict)

        self._validate_n_negative_scores(df_scores)

        return df_scores

    @property
    def score_names(self) -> List[str]:
        score_names = []
        for term in self.terms.values():
            score_names.extend(term.score_names)
        return score_names

    @property
    def score_signs(self) -> List[Literal[1, -1]]:
        score_signs = []
        for term in self.terms.values():
            score_signs.extend(term.score_signs)
        return score_signs

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
        expected_negative_columns = self.score_signs.count(-1)

        # Count columns with only negative or null values
        columns_with_negative_or_null = ((df_scores <= 0).any()).sum()

        if columns_with_negative_or_null < expected_negative_columns:
            raise ValueError(
                f"Mismatch in negative scores count. "
                f"Expected {expected_negative_columns} columns with negative/null values, "
                f"but found {columns_with_negative_or_null}."
            )

    def get_model_params(self) -> Dict[str, float]:
        model_params = {}

        for term_name, term in self.terms.items():
            # Get fit parameters if ModelFOMTerm
            if isinstance(term, ModelFOMTerm):
                term_model_params = term.get_model_params()
                for param_name, param_value in term_model_params.items():
                    # Use term name as prefix to avoid conflicts
                    model_params[f"{term_name}_{param_name}"] = param_value

        return model_params

    def get_profile(self) -> Dict[str, float]:
        stats = {}

        tot_cumtime = 0
        tot_calls = 0
        for term_name, term in self.terms.items():
            cumtime = term.predict_cumtime
            calls = term.predict_count
            tot_cumtime += cumtime
            tot_calls += calls
            stats[f'{term_name}_cumtime'] = cumtime
            stats[f'{term_name}_cumtime_percall'] = 0 if calls == 0 else cumtime / calls

        stats['tot_cumtime'] = tot_cumtime
        stats['tot_cumtime_percall'] = 0 if tot_calls == 0 else tot_cumtime / tot_calls

        return stats

    def get_log_profile(self) -> Dict[str, float]:
        stats = {}

        # Get sum of cumulative time over all terms
        tot_cumtime = 0
        tot_calls = 0
        for term_name, term in self.terms.items():
            cumtime = sum(term.predict_cumtime_log)
            calls = sum(term.predict_count_log)
            tot_cumtime += cumtime
            tot_calls += calls
            stats[f'{term_name}_cumtime'] = cumtime
            stats[f'{term_name}_cumtime_percall'] = 0 if calls == 0 else cumtime / calls 

        stats['tot_cumtime'] = tot_cumtime
        stats['tot_cumtime_percall'] = 0 if tot_calls == 0 else tot_cumtime / tot_calls

        return stats

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {name: term.get_parameters() for name, term in self.terms.items()}

    @staticmethod
    def serialize_dict(dic: Dict) -> str:
        return json.dumps(
            obj=dic,
            default=lambda o: ' '.join(repr(o).split()),
            indent=4
        )
