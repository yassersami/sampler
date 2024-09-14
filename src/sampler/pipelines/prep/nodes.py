from typing import Dict, List, Tuple, Union
import yaml
import numpy as np
import pandas as pd
from sampler.core.data_processing.scalers import MixedMinMaxScaler, set_scaler
from sampler.core.data_processing.data_treatment import DataTreatment


def prepare_treatment(
    features: List[str],
    targets: List[str],
    variables_ranges: Dict[str, Dict[str, Dict]],
    interest_region: Dict[str, Tuple[float, float]],
    simulator_config: Dict,
) -> DataTreatment:
    """
    Prepare the DataTreatment class with the base configuration.
    """
    # Use the 'base' dict as the foundation
    base_ranges = variables_ranges['base']

    # Get base input-output space bounds and interest region
    bounds_dict = {var_name: base_ranges[var_name]['bounds'] for var_name in features + targets}
    interest_region  = {var_name: interest_region[var_name] for var_name in targets}

    # Initialize and return data treatment defined by base_ranges
    treatment = DataTreatment(
        features=features,
        targets=targets,
        scaler=set_scaler(features, targets, base_ranges),
        bounds=bounds_dict,
        interest_region=interest_region,
        max_sim_time=simulator_config['max_sim_time'],
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
