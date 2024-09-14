from typing import List, Tuple, Dict, Union, Literal, Any, Iterator
import numpy as np
import pandas as pd
import json

from .term_base import FittableFOMTerm, ModelFOMTerm, MultiScoreMixin, FOMTermType, FOMTermInstance
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
            self._validate_term_args(term_name, term_args, TermClass)
            apply = term_args.pop('apply')

            if apply:
                # Set term name class attribute
                TermClass._set_term_name(term_name)

                # Add required args from FOM attributes
                term_args.update({
                    arg: getattr(self, arg) for arg in TermClass.required_args
                })

                # Instantiate term
                print(f"FigureOfMerit -> Instantiating term '{term_name}'...")  # To help debug on missing args
                term_instance = TermClass(**term_args)

                terms_dict[term_name] = term_instance

        # Create the accessor
        self.terms = FOMTermAccessor(terms_dict)

        # Compute this sum once for all
        self.positive_score_weights_sum = self.get_positive_score_weights_sum()

    def _validate_term_args(self,
        term_name: str,
        term_args: Dict[str, Union[float, str]],
        TermClass: FOMTermType
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
            # Check if required args from FOM are available
            for arg in TermClass.required_args:
                if not hasattr(self, arg):
                    raise ValueError(
                        f"Required argument '{arg}' for term '{term_name}' is "
                        "not available in FOM"
                    )

    def _validate_term_scoring(self, term: FOMTermInstance, X: np.ndarray) -> None:

        # Select 5 dummy random samples
        rng = np.random.default_rng()
        dummy_X = rng.choice(X, size=5, replace=False, shuffle=False)

        # Validate scoring
        dummy_scores = term.predict_score(dummy_X)
        term._validate_score_shape(dummy_X, dummy_scores)
        term._validate_score_sign(dummy_scores)
        

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

            # Take advantage of having available valid samples
            # Validate scoring process after fitting (a key step)
            self._validate_term_scoring(term, X)

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
        """ Compute a weighted sum of all scores. """
        all_scores = []
        for term in self.terms.values():
            scores = term.predict_score(X)

            # Convert single score to tuple
            scores = (scores,) if isinstance(scores, np.ndarray) else scores

            # Apply weights
            weights = term.score_weights.values()
            weighted_scores = [score * weight for score, weight in zip(scores, weights)]
            all_scores.extend(weighted_scores)

        return np.sum(all_scores, axis=0)

    def predict_loss(self, X: np.ndarray) -> float:
        """
        Compute the weighted sum of losses for all terms.

        This method transforms the scores into a loss function where lower values 
        are better, with 0 being the optimal value. The transformation is as follows:

                loss =   Σ(w_i * (1 - s_i)) for positive scores
                       - Σ(w_j * s_j)       for negative scores

        where w_i and w_j are weights, and s_i and s_j are scores.
        """
        return self.positive_score_weights_sum - self.predict_score(X)

    def get_scores_df(self, X: np.ndarray) -> pd.DataFrame:
        scores_dict = {}

        for term_name, term in self.terms.items():
            # Get scores
            scores = term.predict_score(X)
            
            # Convert single array to tuple for consistent handling
            if isinstance(scores, np.ndarray):
                scores = (scores,)  

            # Add scores with alias key to dictionary
            for score, score_name in zip(scores, term.score_names):
                score_alias = self.get_score_alias(score_name, term_name, term)
                scores_dict[score_alias] = score

        df_scores = pd.DataFrame(scores_dict)
        return df_scores

    def get_score_alias(self, score_name: str, term_name: str, term: FOMTermInstance) -> str:
        if isinstance(term, MultiScoreMixin):
            return f'{term_name}_{score_name}'
        return term_name

    def get_score_names(self) -> List[str]:
        score_names = []
        for term_name, term in self.terms.items():
            term_score_names = [
                self.get_score_alias(score_name, term_name, term)
                for score_name in term.score_names
            ]
            score_names.extend(term_score_names)
        return score_names

    def get_score_weights(self) -> Dict[str, float]:
        score_weights = {}
        for term_name, term in self.terms.items():
            term_score_weights = {
                self.get_score_alias(score_name, term_name, term): score_weight 
                for score_name, score_weight in term.score_weights.items()
            }
            score_weights.update(term_score_weights)
        return score_weights

    def get_score_signs(self) -> Dict[str, Literal[1, -1]]:
        score_signs = {}
        for term_name, term in self.terms.items():
            term_score_signs = {
                self.get_score_alias(score_name, term_name, term): score_sign 
                for score_name, score_sign in term.score_signs.items()
            }
            score_signs.update(term_score_signs)
        return score_signs

    def get_positive_score_weights_sum(self) -> float:
        """ Sum of weights of all positive scores. """
        score_weights = list(self.get_score_weights().values())
        score_signs = list(self.get_score_signs().values())
        return sum([
            score_weight for score_weight, score_sign in zip(score_weights, score_signs)
            if score_sign == 1
        ])

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

    def get_profile(self, use_log: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Profiling since initialization (if use_log=True) or since
        `reset_predict_profiling` (if use_log=False).
        """
        stats = {}

        tot_cumtime = 0
        tot_calls = 0
        for term_name, term in self.terms.items():
            if use_log:
                cumtime = sum(term.predict_cumtime_log)
                calls = sum(term.predict_count_log)
            else:
                cumtime = term.predict_cumtime
                calls = term.predict_count

            tot_cumtime += cumtime
            tot_calls += calls
            stats[term_name] = {
                'cumtime': cumtime,
                'cumtime_percall': 0 if calls == 0 else cumtime / calls
            }

        stats['Total'] = {
            'cumtime': tot_cumtime,
            'cumtime_percall': 0 if tot_calls == 0 else tot_cumtime / tot_calls
        }

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
