import glob
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from sampler.common.data_treatment import DataTreatment


def get_result(path: str):
    if 'history' in path:
        path = path.replace('/db-t/', '/') # TODO: fix this with the actual path being used in the first place
        path = path.replace('/db-p/', '/') # TODO: fix this with the actual path being used in the first place
        path = path.replace('/db-tp/', '/') # TODO: fix this with the actual path being used in the first place
        files = glob.glob(f"{path}/*.csv")
        files.sort()
        dfs = [pd.read_csv(f) for f in files]
        history = pd.concat(dfs, ignore_index=True)
        return history
    else:
        return pd.read_csv(path)


def prepare_benchmark(df: pd.DataFrame, f: List[str], t: List[str], treatment: DataTreatment,) -> pd.DataFrame:
    res = treatment.define_quality_of_data(data=df, specify_errors=False)
    return res


def create_dict(df: pd.DataFrame, name: str, color: str) -> Dict[str, Union[str, pd.DataFrame]]:
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
    final_res = treatment.define_quality_of_data(data=res, specify_errors=False)
    return final_res


def extract_percentage(initial_size, tot_size, n_slice, vals):
    res_io = pd.DataFrame(columns=['interest', 'others', 'in%', 'o%'])
    for n_row, lim in enumerate(np.append([initial_size], np.arange(n_slice, tot_size + 1, n_slice))):
        if lim <= vals['df'].shape[0]:
            res_io.loc[lim, 'interest'] = vals['interest'].loc[vals['interest'].index < lim].shape[0]
            res_io.loc[lim, 'in%'] = res_io.loc[lim, 'interest'] / lim
            res_io.loc[lim, 'others'] = lim - res_io.loc[lim, 'interest']
            res_io.loc[lim, 'o%'] = 1 - res_io.loc[lim, 'interest'] / lim
    return res_io
