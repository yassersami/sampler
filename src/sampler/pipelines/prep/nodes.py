from typing import Dict, List
import numpy as np
import pandas as pd
from sampler.common.scalers import MixedMinMaxScaler
from sampler.common.data_treatment import DataTreatment


def prepare_treatment(
    features: List[str], targets: List[str], variables_ranges: Dict,
    interest_region: Dict, simulator_env: Dict
) -> DataTreatment:
    """
    Prepare the DataTreatment class with the necessary configuration.
    """
    # Construct scaler to scale all values from 0 to 1
    bounds = np.array([v['bounds'] for v in variables_ranges.values()]).T
    scale = [v['scale'] for v in variables_ranges.values()]
    scaler = MixedMinMaxScaler(scale=scale, features=features, targets=targets)
    scaler.fit(bounds)

    # Initialize and return data treatment
    treatment = DataTreatment(
        features=features, targets=targets, scaler=scaler,
        variables_ranges=variables_ranges, interest_region=interest_region,
        sim_time_cutoff=simulator_env["sim_time_cutoff"],
    )
    return treatment


def prepare_data(
    initial_data: pd.DataFrame, additional_values: List[str],
    treatment: DataTreatment
) -> Dict:
    """
    Prepare and scale initial data using the provided DataTreatment instance.
    """
    features, targets = treatment.features, treatment.targets
    
    # Filter and treat data
    res = initial_data[(initial_data[features].isna().sum(axis=1) == 0)]
    res = treatment.select_in_bounds_feat(res)
    res = treatment.treat_real_data(res)

    return res[features + targets + additional_values]
