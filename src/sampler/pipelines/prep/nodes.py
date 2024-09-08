from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sampler.common.scalers import MixedMinMaxScaler
from sampler.common.data_treatment import DataTreatment


def prepare_treatment(
    features: List[str],
    targets: List[str],
    variables_ranges: Dict[str, Dict],
    interest_region: Dict[str, Tuple[float, float]],
    simulator_config: Dict,
) -> DataTreatment:
    """
    Prepare the DataTreatment class with the necessary configuration.
    """
    # Prepare input-output space description
    all_vars = features + targets
    bounds_dict = {var: variables_ranges[var]['bounds'] for var in all_vars}
    scale = {var: variables_ranges[var]['scale'] for var in all_vars}
    interest_region = {var: interest_region[var] for var in targets}

    # Fit scaler to scale all values from 0 to 1
    scaler = MixedMinMaxScaler(features=features, targets=targets, scale=scale)
    bounds = np.array(list(bounds_dict.values())).T  # dict.values() keeps order
    scaler.fit(bounds)

    # Initialize and return data treatment
    treatment = DataTreatment(
        features=features, targets=targets, scaler=scaler,
        bounds=bounds_dict, interest_region=interest_region,
        max_simu_time=simulator_config['max_simu_time'],
    )
    return treatment


def prepare_data(
    initial_data: pd.DataFrame,
    additional_values: List[str],
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

    # Select available columns from additional_values
    available_values = [col for col in additional_values if col in res]

    return res[features + targets + available_values]
