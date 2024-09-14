import os
from typing import Dict, List, Union
import warnings
import numpy as np
import pandas as pd

from sampler.common.data_treatment import DataTreatment
from sampler.common.scalers import MixedMinMaxScaler


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


def scale_back_to_SI_units(
    df: pd.DataFrame,
    features: List[str],
    targets: List[str],
    scaler: MixedMinMaxScaler,
) -> pd.DataFrame:
    """ Scale the data back to physical SI units. """
    scaler_cols = features + targets
    df[scaler_cols] = pd.DataFrame(
        scaler.inverse_transform(df[scaler_cols].values),
        columns=scaler_cols
    )
    return df


def add_quality_columns(
    df: pd.DataFrame,
    df_name: str,
    treatment: DataTreatment,
) -> pd.DataFrame:
    """
    Add quality classification columns to the DataFrame based on the base variables configuration.

    This function classifies data points as 'interest' or 'no_interest', and further
    categorizes outliers among 'no_interest' points into specific error types.

    Note:
        Quality classification is based on the base variables configuration in
        the treatment object. If an experiment uses a different scaler that
        transforms feature or target values outside the base ranges, those
        points will be classified as 'out_of_bounds' outliers.

        Generally, out-of-bounds outliers are very few (< 5). If their number is
        abnormally high, it may indicate that the base ranges are too narrow.
    """

    # Quality specifies either 'interest', 'no_interest' or which outlier type
    df = treatment.classify_quality(df, data_is_scaled=False)

    # Check for out-of-bounds outliers
    out_of_bounds_feat = df[df['quality'] == 'out_of_bounds_feat']
    out_of_bounds_tar = df[df['quality'] == 'out_of_bounds_tar']

    # Prepare and print the report only if out-of-bounds outliers are present
    if not out_of_bounds_feat.empty or not out_of_bounds_tar.empty:
        report = 'add_quality_columns -> \n'
        report += f"Out-of-bounds outliers report for experiment '{df_name}':\n"
        if not out_of_bounds_feat.empty:
            report += f"  - Feature out-of-bounds: {len(out_of_bounds_feat)}\n"
        if not out_of_bounds_tar.empty:
            report += f"  - Target out-of-bounds: {len(out_of_bounds_tar)}\n"
        print(report)

    # Raise a warning if out-of-bounds feature outliers are present
    if not out_of_bounds_feat.empty:
        warnings.warn(
            f"{len(out_of_bounds_feat)} feature out-of-bounds outliers detected in '{df_name}' data. "
            "These points will not appear in plots and may affect analysis.",
            UserWarning
        )

    return df


def subset_by_quality(
    df: pd.DataFrame, exp_config: Dict[str, str],
) -> Dict[str, Union[str, pd.DataFrame]]:
    return {
        **exp_config,
        'interest': df[(df.quality == 'interest')],
        'no_interest': df[(df.quality == 'no_interest')],  # or (df.quality != 'interest') ?
        'inliers': df[(df.quality == 'interest') | (df.quality == 'no_interest')],
        'outliers': df[(df.quality != 'interest') & (df.quality != 'no_interest')],
        'df': df
    }
