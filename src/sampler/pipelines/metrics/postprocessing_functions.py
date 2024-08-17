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


def categorize_df_by_quality(
    df: pd.DataFrame, name: str, color: str
) -> Dict[str, Union[str, pd.DataFrame]]:
    return {
        'name': name,
        'color': color,
        'interest': df[(df.quality == 'interest')],
        'not_interesting': df[(df.quality == 'not_interesting')],
        'inliers': df[(df.quality == 'interest') | (df.quality == 'not_interesting')],
        'outliers': df[(df.quality != 'interest') & (df.quality != 'not_interesting')],
        'df': df
    }


def prepare_new_data(
        df: pd.DataFrame, treatment: DataTreatment,
        f: List[str], t: List[str], t_c: List[str]
) -> pd.DataFrame:
    res = df.copy()
    res[f+t] = pd.DataFrame(treatment.scaler.inverse_transform(df[f+t].values), columns=f+t)

    # TODO: Temporal fix to allow plotting data without prediction columns
    if t_c[0] not in df.columns:
        df[t_c[0]] = df[t[0]]
        df[t_c[1]] = df[t[1]]

    res[f+t_c] = pd.DataFrame(treatment.scaler.inverse_transform(df[f+t_c].values), columns=f+t_c)
    # res = treatment.define_quality_of_data(data=res, specify_errors=True)
    res = treatment.classify_quality_interest(res, data_is_scaled=False)
    res = treatment.classify_quality_error(res, data_is_scaled=False)
    return res


def extract_percentage(initial_size, tot_size, n_slice, vals):
    res_io = pd.DataFrame(columns=['interest', 'others', 'in%', 'o%'])
    for lim in np.append([initial_size], np.arange(n_slice, tot_size + 1, n_slice)):
        if lim <= vals['df'].shape[0]:
            res_io.loc[lim, 'interest'] = vals['interest'].loc[vals['interest'].index < lim].shape[0]
            res_io.loc[lim, 'in%'] = res_io.loc[lim, 'interest'] / lim
            res_io.loc[lim, 'others'] = lim - res_io.loc[lim, 'interest']
            res_io.loc[lim, 'o%'] = 1 - res_io.loc[lim, 'interest'] / lim
    return res_io
