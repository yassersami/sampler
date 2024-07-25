from typing import Dict, List

import numpy as np
import pandas as pd

from sampler.common.scalers import MixedMinMaxScaler
from sampler.common.data_treatment import DataTreatment


def prepare_initial_data(
        initial_data: pd.DataFrame, features: List[str], targets: List[str], additional_values: List[str],
        interest_region: Dict, variables_ranges: Dict, outliers_filling: Dict, sim_time_cutoff: int,
        scaler: MixedMinMaxScaler,
) -> Dict:
    """
    Define the values that will be considered of interest or Inliers/Outliers. Return all these rules in treatment.
    Take the initial data, scale it, classify it, remove unwanted values. Return as treated data.
    """
    treatment = DataTreatment(
        features=features, targets=targets, scaler=scaler, interest_region=interest_region,
        variables_ranges=variables_ranges, sim_time_cutoff=sim_time_cutoff, outliers_filling=outliers_filling,
    )
    res, _ = treatment.treat_data(df_real=initial_data, scale=True)
    res[additional_values] = initial_data[additional_values]
    res = res[(res[features].isna().sum(axis=1) == 0)]
    return dict(
        treated_data=res[features + targets + additional_values],
        treatment=treatment,
    )


def get_scaler(log_scale: bool, features: List[str], targets: List[str], variables_ranges: Dict) -> MixedMinMaxScaler:
    """ Construct scaler to make all values go from 0 to 1 """
    bounds = np.array([value['bounds'] for key, value in variables_ranges.items()]).T
    scale = [value['scale'] for key, value in variables_ranges.items()] if log_scale else None
    scaler = MixedMinMaxScaler(scale=scale, features=features, targets=targets)
    scaler.fit(bounds)
    return scaler
