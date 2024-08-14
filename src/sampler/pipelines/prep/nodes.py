from typing import Dict, List

import numpy as np
import pandas as pd

from sampler.common.scalers import MixedMinMaxScaler
from sampler.common.data_treatment import DataTreatment


def prepare_initial_data(
    initial_data: pd.DataFrame, features: List[str], targets: List[str],
    additional_values: List[str], interest_region: Dict, variables_ranges: Dict,
    outliers_filling: Dict, sim_time_cutoff: int, scaler: MixedMinMaxScaler,
) -> Dict:
    """
    Define the values that will be considered of interest or Inliers/Outliers.
    Return all these rules in treatment. Take the initial data, scale it,
    classify it, remove unwanted values. Return as treated data.
    """
    treatment = DataTreatment(
        features=features, targets=targets, scaler=scaler,
        interest_region=interest_region, variables_ranges=variables_ranges,
        sim_time_cutoff=sim_time_cutoff, outliers_filling=outliers_filling,
    )
    res = initial_data[(initial_data[features].isna().sum(axis=1) == 0)]
    res = treatment.select_in_bounds_feat(res)
    res = treatment.treat_real_data(res)
    return dict(
        treated_data=res[features + targets + additional_values],
        treatment=treatment,
    )


def get_scaler(
    log_scale: bool, features: List[str], targets: List[str],
    variables_ranges: Dict
) -> MixedMinMaxScaler:
    """ Construct scaler to make all values go from 0 to 1 """
    bounds = np.array([v['bounds'] for v in variables_ranges.values()]).T
    scale = [v['scale'] for v in variables_ranges.values()] if log_scale else None
    scaler = MixedMinMaxScaler(scale=scale, features=features, targets=targets)
    scaler.fit(bounds)
    return scaler
