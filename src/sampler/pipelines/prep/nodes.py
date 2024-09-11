from typing import Dict, List, Tuple, Union
import yaml
import numpy as np
import pandas as pd
from sampler.common.scalers import MixedMinMaxScaler
from sampler.common.data_treatment import DataTreatment


def set_scaler(
    features: List[str],
    targets: List[str],
    ranges: Dict[str, Dict[str, Union[float, str]]],
) -> MixedMinMaxScaler:
    scaler_variables = features + targets

    # Prepare input-output space description
    bounds_dict = {var_name: ranges[var_name]['bounds'] for var_name in scaler_variables}
    scale = {var_name: ranges[var_name]['scale'] for var_name in scaler_variables}

    # Fit scaler to scale all values from 0 to 1
    scaler = MixedMinMaxScaler(features=features, targets=targets, scale=scale)
    bounds = np.array(list(bounds_dict.values())).T  # dict.values() keeps order
    scaler.fit(bounds)

    return scaler


def prepare_treatment(
    features: List[str],
    targets: List[str],
    variables_ranges: Dict[str, Dict[str, Dict]],
    interest_region: Dict[str, Tuple[float, float]],
    simulator_config: Dict,
) -> DataTreatment:
    """
    Prepare the DataTreatment class with the necessary configuration.
    """
    # Use the 'base' dict as the foundation
    base_ranges = variables_ranges['base']

    # Complete other dicts with base dict
    other_config_names = [name for name in variables_ranges if name != 'base']
    for config_name in other_config_names:
        # Check for invalid keys
        for var_name in variables_ranges[config_name]:
            if var_name not in base_ranges:
                raise KeyError(
                    f"Variable '{var_name}' in '{config_name}' configuration "
                    "is not present in the base configuration."
                )

        # Complete with base
        variables_ranges[config_name] = {**base_ranges, **variables_ranges[config_name]}

    # Set a scaler for each variables configuration
    scalers = {
        config_name: set_scaler(features, targets, ranges)
        for config_name, ranges in variables_ranges.items()
    }

    # Get base input-output space bounds and interest region
    bounds_dict = {var_name: base_ranges[var_name]['bounds'] for var_name in features + targets}
    interest_region  = {var_name: interest_region[var_name]       for var_name in targets}

    # Initialize and return data treatment defined by base_ranges
    treatment = DataTreatment(
        features=features,
        targets=targets,
        scaler=set_scaler(features, targets, base_ranges),
        bounds=bounds_dict,
        interest_region=interest_region,
        max_simu_time=simulator_config['max_simu_time'],
    )

    # Prepare base configuration with interest_region as YAML data
    yaml_data = {
        'base_ranges': base_ranges,
        'interest_region': interest_region
    }

    return {
        'treatment': treatment,
        'scalers': scalers,
        'base_config': yaml_data,
    }


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
