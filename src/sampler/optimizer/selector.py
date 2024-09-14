from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from .base import MultiModalSelector, BaseFactory

class SelectorFactory(BaseFactory[MultiModalSelector]):
    @classmethod
    def create_from_config(cls,
        batch_size: int,
        selector_config: Dict[str, Dict]
    ) -> MultiModalSelector:
        return super().create_from_config(
            selector_config,
            item_type="selector",
            batch_size=batch_size
        )


@SelectorFactory.register('diversity')
class DiversitySelector(MultiModalSelector):

    def __init__(self, batch_size: int, tol: float = 1e-5):
        super().__init__(batch_size)
        self.tol = tol

        self._record_columns = [
            '_id', 'loss', 'diversity', 'extremeness',
            'normal_loss', 'normal_diversity', 'mmm_loss'
        ]
        self.reset_records()

    def select_diverse_minima(
        self, X: np.ndarray, loss_values: np.ndarray
    ) -> np.ndarray:
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
        edge_proximity_mask = (0.5 - np.abs(X - 0.5)) < self.tol  # Shape of X
        extremeness = edge_proximity_mask.sum(axis=1)

        # Initialize list of selected indices
        selected_indices = []

        # Initialize list of not yet selected indices
        n_samples = X.shape[0]
        available_indices = list(range(n_samples))

        # Select diverse candidates
        for _ in range(min(self.batch_size, n_samples)):

            if not selected_indices:
                # If no selected indices yet set dummy null diversity value
                diversity_scores = np.zeros(n_samples)
                normalized_diversity = np.zeros(n_samples)
            else:
                # Calculate diversity score: mean distance to already selected candidates
                diversity_scores = distances[:, selected_indices].mean(axis=1)

                # Normalize diversity scores to [0, 1] range
                normalized_diversity = self._normalize(diversity_scores)

            # Combine normalized objective and diversity scores
            # objective is minimized while diversity is maximized
            combined_loss = 2 * extremeness + normalized_loss - normalized_diversity    # This is the main operation

            # Remove rows of already selected indices
            available_combined_loss = [combined_loss[i] for i in available_indices]

            # Select the individual with the smallest loss
            next_index = available_indices[np.argmin(available_combined_loss)]

            selected_indices.append(next_index)
            available_indices.remove(next_index)

            # Store selection progress
            self.save_records(
                _id             =next_index,
                loss            =loss_values[next_index],
                diversity       =diversity_scores[next_index],
                extremeness     =extremeness[next_index],
                normal_loss     =normalized_loss[next_index],
                normal_diversity=normalized_diversity[next_index],
                mmm_loss        =combined_loss[next_index],
            )

        return X[selected_indices]


@SelectorFactory.register('centrism')
class CentrismSelector(MultiModalSelector):

    def __init__(self, batch_size: int, tol: float = 1e-5):
        super().__init__(batch_size)
        self.tol = tol

        self._record_columns = ['_id', 'loss', 'extremeness', 'mmm_loss']
        self.reset_records()

    def select_diverse_minima(
        self, X: np.ndarray, loss_values: np.ndarray
    ) -> np.ndarray:

        X = np.atleast_2d(X)

        # Create edge proximity mask where true if value close to (0 | 1) edges 
        edge_proximity_mask = (0.5 - np.abs(X - 0.5)) < self.tol  # Shape of X

        # Get a loss based on number of close features to an edge per row
        extremeness = edge_proximity_mask.sum(axis=1)

        # Combine extremeness and loss values but consider extremeness first
        combined_loss = 10 * extremeness + loss_values

        # Get indices to sort X where smallest value (less close to an edge) comes first
        # 'stable' sorting maintains the relative order of equal elements
        sort_indices = np.argsort(combined_loss, kind='stable')

        # Set the final selected indices
        selected_indices = sort_indices[:self.batch_size]

        self._records = {
            '_id': selected_indices,
            'loss': loss_values[selected_indices],
            'extremeness': extremeness[selected_indices],
            'mmm_loss': combined_loss[selected_indices],
        }
        return X[selected_indices]


@SelectorFactory.register('elitism')
class ElitismSelector(MultiModalSelector):

    def __init__(self, batch_size: int):
        super().__init__(batch_size)

        self._record_columns = ['_id', 'loss', 'mmm_loss']
        self.reset_records()

    def select_diverse_minima(
        self, X: np.ndarray, loss_values: np.ndarray
    ) -> np.ndarray:

        X = np.atleast_2d(X)

        # Combined loss based exclusively on objective loss value
        combined_loss = loss_values

        # Get indices to sort X where smallest loss value comes first
        sort_indices = np.argsort(combined_loss)

        # Set the final selected indices
        selected_indices = sort_indices[:self.batch_size]

        self._records = {
            '_id': selected_indices,
            'loss': loss_values[selected_indices],
            'mmm_loss': combined_loss[selected_indices]
        }
        return X[selected_indices]
    