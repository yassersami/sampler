from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
from abc import ABC, abstractmethod
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import shgo
from sko.GA import GA


class DiversityMultiModalSelector:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

        # Multimodal progress record attributes
        self._mmm_score_names = ['_id', 'loss', 'diversity', 'extremeness', 'normal_loss', 'normal_diversity', 'mmm_loss']
        self.reset_mmm_scores()

    def reset_mmm_scores(self):
        self._mmm_scores = {name: [] for name in self._mmm_score_names}

    def select_diverse_optima(
        self, X: np.ndarray, loss_values: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Select diversified batch of candidates among explored points during
        minimization.
        """
        X = np.atleast_2d(X)

        # Normalize objective values to [0, 1] range
        normalized_loss = self._normalize(loss_values)

        # Calculate pairwise distances between individuals
        distances = squareform(pdist(X))

        # Edge proximity loss (extremeness to minimize)
        edge_proximity_mask = (np.isclose(X, 0) | np.isclose(X, 1))  # Shape of X
        extremeness = edge_proximity_mask.mean(axis=1)

        # Initialize selected candidates with the best sample
        first_index = np.argmin(loss_values)  # best sample with smallest objective value
        selected_indices = [first_index]

        # Initialize list of not yet selected indices
        n_samples = X.shape[0]
        available_indices = list(range(n_samples))
        available_indices.remove(first_index)

        # Set dict to store selection progress
        self._store_mmm_progress(
            _id             =first_index,
            loss            =loss_values[first_index],
            diversity       =np.nan,
            extremeness     =extremeness[first_index],
            normal_loss     =normalized_loss[first_index],
            normal_diversity=np.nan,
            mmm_loss        =normalized_loss[first_index],
        )

        # Select diverse candidates
        for _ in range(1, min(self.batch_size, n_samples)):

            # Calculate diversity score: mean of distances to already selected candidates
            diversity_scores = distances[:, selected_indices].mean(axis=1)

            # Normalize diversity scores to [0, 1] range
            normalized_diversity = self._normalize(diversity_scores)

            # Combine normalized objective and diversity scores
            # objective is minimized while diversity is maximized
            combined_loss = normalized_loss - normalized_diversity  + 2 * extremeness  # This is the main operation

            # Remove rows of already selected indices
            available_combined_loss = [combined_loss[i] for i in available_indices]

            # Select the individual with the smallest loss
            next_index = available_indices[np.argmin(available_combined_loss)]

            selected_indices.append(next_index)
            available_indices.remove(next_index)

            # Store selection progress
            self._store_mmm_progress(
                _id             =next_index,
                loss            =loss_values[next_index],
                diversity       =diversity_scores[next_index],
                extremeness     =extremeness[next_index],
                normal_loss     =normalized_loss[next_index],
                normal_diversity=normalized_diversity[next_index],
                mmm_loss        =combined_loss[next_index],
            )

        return X[selected_indices], pd.DataFrame(self._mmm_scores)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        """Normalize values to [0, 1] range."""
        min_val, max_val = values.min(), values.max()
        return (values - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(values)

    def _store_mmm_progress(self, **kwargs):
        # Store selection progress
        for key in self._mmm_score_names:
            if key not in kwargs.keys():
                raise KeyError(f"Multimodal selection progess record is missing '{key}'")

            self._mmm_scores[key].append(kwargs[key])


class BoundaryMultiModalSelector():

    @classmethod
    def sort_by_relevance(cls, X_unique: np.ndarray, mask: np.ndarray) -> List:
        n_dim = X_unique.shape[1]
        bounds = [[] for _ in range(n_dim + 1)]
        for x_row, cond in zip(X_unique, mask):
            cond_sum = cond.sum()
            # TODO why not directly bounds[cond_sum].append since cond_sum is in [0, n_dim] anyway ?
            for local_sum in range(n_dim + 1):
                if cond_sum == local_sum:
                    bounds[local_sum].append(x_row)
        return [
            item
            for val in bounds
            for item in val
        ]

    @classmethod
    def select_diverse_optima(cls, X_candidates: np.ndarray, batch_size: int) -> np.ndarray:
        X_unique = np.unique(X_candidates, axis=0)
        if len(X_unique) <= batch_size:
            # if not enough candidates return all available ones
            return X_candidates

        # If a lot of candidates, select best ones
        mask = (np.isclose(X_unique, 0) | np.isclose(X_unique, 1))
        X_unique_sorted = cls.sort_by_relevance(X_unique, mask)
        return np.array(X_unique_sorted[:batch_size])


class MultiModalSelector():
    boundary_mms: BoundaryMultiModalSelector
    diversity_mms: DiversityMultiModalSelector

    def __init__(self, name):
        if name == 'boundary_selector':
            self.selector = BoundaryMultiModalSelector()
        elif name == 'diversity_selector':
            self.selector = DiversityMultiModalSelector()
        else:
            raise AttributeError(f"{name} is unknown multimodal selector")
