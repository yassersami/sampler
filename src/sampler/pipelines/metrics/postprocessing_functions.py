import os
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from sampler.common.data_treatment import DataTreatment


def aggregate_csv_files(directory_path: str) -> pd.DataFrame:
    """
    Combine data from all CSV files in a given directory into a single DataFrame.

    Args:
        directory_path (str): Path to the directory containing CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame from all CSV files in the directory.
    """
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    dataframes = []

    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def prepare_new_data(
        df: pd.DataFrame,
        treatment: DataTreatment,
        features: List[str],
        targets: List[str],
) -> pd.DataFrame:
    """
    Prepare the new data by scaling it back to physical units and classifying the
    quality of the points.

    Returns:
        pd.DataFrame: Prepared DataFrame with the quality of the points classified.
    """
    res = df.copy()

    # Scale back to physical units
    scaler_cols = features + targets
    res[scaler_cols] = pd.DataFrame(
        treatment.scaler.inverse_transform(df[scaler_cols].values),
        columns=scaler_cols
    )

    # Classify quality
    res = treatment.classify_quality_interest(res, data_is_scaled=False)
    # Classify outliers
    res = treatment.classify_quality_error(res, data_is_scaled=False)
    return res


def subset_by_quality(
    df: pd.DataFrame, name: str, color: str
) -> Dict[str, Union[str, pd.DataFrame]]:
    return {
        'name': name,
        'color': color,
        'interest': df[(df.quality == 'interest')],
        'no_interest': df[(df.quality == 'no_interest')],
        'inliers': df[(df.quality == 'interest') | (df.quality == 'no_interest')],
        'outliers': df[(df.quality != 'interest') & (df.quality != 'no_interest')],
        'df': df
    }


def get_first_iteration_index(df: pd.DataFrame) -> int:
    """
    Get index of first sample that was generated through adaptive sampling
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