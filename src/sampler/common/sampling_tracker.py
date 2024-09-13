from typing import List, Dict, Tuple, Callable, Type, Union, Optional, ClassVar
import numpy as np
import pandas as pd


class SamplingProgressTracker:
    def __init__(self,
        targets: List[str],
        max_inliers: int = 10,
        max_interest: Optional[int] = None,
        stop_on_max_inliers: bool = True,
    ):
        self.targets = targets

        # Stop condition
        self.max_inliers = max_inliers
        self.max_interest = max_interest
        self.stop_on_max_inliers = stop_on_max_inliers
        self._validate_stop_condition()

        # Initialize state
        self.n_total = 0  # all simulations
        self.n_inliers = 0  # only inliers
        self.n_interest = 0  # only interesting inliers
        self.iteration = 1

    def _validate_stop_condition(self):
        if not self.stop_on_max_inliers and self.max_interest is None:
            raise ValueError("max_interest must be provided when stop_on_max_inliers is False")

    @property
    def pbar_total(self):
        return self.max_inliers if self.stop_on_max_inliers else self.max_interest

    def get_state(self):
        return {
            'n_total': self.n_total,
            'n_inliers': self.n_inliers,
            'n_interest': self.n_interest,
            'iteration': self.iteration,
        }

    def update_state(self, new_df: pd.DataFrame):
        # Count
        n_new_samples = new_df.shape[0]
        n_new_inliers = new_df.dropna(subset=self.targets).shape[0]
        n_new_interest = new_df[new_df['quality'] == 'interest'].shape[0]

        # Update state
        self.n_total += n_new_samples
        self.n_inliers += n_new_inliers
        self.n_interest += n_new_interest
        self.iteration += 1

        # Compute pbar progress (second term avoids exceeding maximum advancement)
        if self.stop_on_max_inliers:
            return n_new_inliers - max(0, n_new_inliers - self.max_inliers)
        else:
            return n_new_interest - max(0, n_new_interest - self.max_interest)

    def print_iteration_report(self):
        print(
            f"Round {self.iteration - 1:03} (end) - Report count: "
            f"Total: {self.n_total}, "
            f"Inliers: {self.n_inliers}, "
            f"Interest: {self.n_interest}"
        )

    def should_continue(self):
        if self.stop_on_max_inliers:
            return self.n_inliers < self.max_inliers
        else:
            return self.n_interest < self.max_interest


def get_first_iteration_index(df: pd.DataFrame) -> int:
    """
    Finds the index of first sample that was generated through adaptive sampling
    pipeline using either 'iteration' or 'datetime' column.
    """
    # Check if df is empty
    if df.empty:
        return 0

    # Check if 'iteration' or 'datetime' column exists
    if 'iteration' in df.columns:
        column = 'iteration'
    elif 'datetime' in df.columns:
        column = 'datetime'
    else:
        raise ValueError("DataFrame does not contain 'iteration' or 'datetime' column.")

    # Find the index of the first non-empty cell in the chosen column
    first_non_empty_index = df[column].first_valid_index()

    # If all cells are empty, return last index
    if first_non_empty_index is None:
        return df.index[-1]

    # Return the first non-empty index
    return first_non_empty_index


def get_max_interest_index(df: pd.DataFrame, stop_condition: Dict) -> int:
    """
    Finds the index in the DataFrame where the number of 'interest' samples
    reaches the specified max_interest value.

    This function starts counting from the first sample generated through
    adaptive sampling. It then counts 'interest' samples until reaching the
    max_interest value specified in the stop_condition.
    """
    # Get index of first sample that was generated through adaptive sampling
    index = get_first_iteration_index(df)

    # Initialize interest samples count
    interest_count = 0

    # Loop until number of interest samples matches max_interest
    while (
        interest_count < stop_condition['max_interest'] and 
        index < len(df)
    ):
        if df.iloc[index]['quality'] == 'interest':
            interest_count += 1
        index += 1

    # truncate increased_data to respect stop_condition
    if interest_count != stop_condition['max_interest']:
        raise ValueError(
            "Not enough interest samples in the dataset. "
            f"Expected: {stop_condition['max_interest']}, Got: {interest_count}."
        )
    return index
