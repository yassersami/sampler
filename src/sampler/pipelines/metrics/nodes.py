"""
This is a boilerplate pipeline 'metrics'
generated using Kedro 0.18.5
"""
from typing import List, Dict, Tuple, Union
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from .postprocessing_functions import aggregate_csv_files, scale_back_to_SI_units, add_quality_columns, subset_by_quality
from .volume import covered_space_bound
from .asvd import ASVD
from .asvd_plot import plot_stars_volumes_distribution, plot_multiple_asvd_distributions
from .voronoi import get_volume_voronoi
from .voronoi_plot import dist_volume_voronoi
from .graphics_metrics import plot_initial_data, plot_feature_pairs, plot_violin_distribution, targets_kde, plot_feat_tar
from sampler.core.data_processing.data_treatment import DataTreatment
from sampler.core.data_processing.scalers import MixedMinMaxScaler, set_scaler
from sampler.core.data_processing.sampling_tracker import get_first_iteration_index


def set_data_scalers(
    features: List[str],
    targets: List[str],
    variables_ranges: Dict[str, Dict[str, Dict]],
) -> Dict[str, MixedMinMaxScaler]:
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

    return scalers


def read_and_prepare_data(
    experiments: Dict[str, Dict[str, str]],
    features: List[str],
    targets: List[str],
    treatment: DataTreatment,
    scalers: Dict[str, MixedMinMaxScaler],
) -> Dict:
    """
    Prepare data metrics by processing experiment data based on the path type.
    """
    data = {}

    for exp_key, exp_config in experiments.items():
        # Check if experiment configuration has valid scaler
        if 'scaler' not in exp_config:
            raise ValueError(f"Experiment '{exp_key}' is missing scaler key.")
        if exp_config['scaler'] not in scalers:
            raise ValueError(
                f"Experiment '{exp_key}' has invalid scaler '{exp_config['scaler']}'. "
                f"Valid scalers: {list(scalers)}."
            )

        # Import data, etiher from a folder or a csv file
        file_path = exp_config['path']

        if os.path.isdir(file_path):
            # Combine all csv files in given directory
            df = aggregate_csv_files(file_path)
        elif os.path.isfile(file_path) and file_path.endswith('.csv'):
            # Read csv file
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Path '{file_path}' is neither a valid directory nor a CSV file.")

        # Scale back and classify quality
        df = scale_back_to_SI_units(df, features, targets, scalers[exp_config['scaler']])
        df = add_quality_columns(df, exp_key, treatment)

        # Categorize quality
        data[exp_key] = subset_by_quality(df, exp_config)

    return data


def compute_metrics(
    data: Dict[str, Union[str, pd.DataFrame]],
    features: List[str],
    targets: List[str],
    treatment: DataTreatment,
    params_volume: Dict,
    params_asvd: Dict,
    params_voronoi: Dict
) -> Dict:

    n_interest = {} # for each experiment, number of interest points (dict of int)
    volume = {} # for each experiment, volume space covered (dict of float)
    asvd = {}  # for each experiment, ASVD class
    volume_voronoi = {} # for each experiment, volumes of the Voronoi regions (clipped by the unit hypercube)
    
    variable_names = features + targets
    
    for exp_key, exp_dic in data.items():
        first_iter_index = get_first_iteration_index(exp_dic['df'])
        # Scale all data
        XY = exp_dic['df'][variable_names].values
        scaled_data = pd.DataFrame(
            treatment.scaler.transform(XY),
            columns=variable_names
        )
        # Scale interest data
        XY_interest = exp_dic['interest'][variable_names].values
        scaled_data_interest = pd.DataFrame(
            treatment.scaler.transform(XY_interest),
            columns=variable_names
        )
        scaled_x_interest = scaled_data_interest[features]
        scaled_y_interest = scaled_data_interest[targets]

        # Get volume of interesting samples
        if exp_key in params_volume['default']:
            # Use default values
            volume[exp_key] = params_volume['default'][exp_key]
        elif params_volume['compute_volume']:
            # Compute volume
            radius = 0.025 # Sphere radius
            volume[exp_key] = covered_space_bound(
                scaled_x_interest, radius, params_volume, len(features)
            )
        else:
            volume[exp_key] = np.array([0,0])

        if params_asvd['compute_asvd']:
            irbs_scaled_data = scaled_data[first_iter_index:]
            # Get all data distribution using ASVD
            asvd[exp_key] = ASVD(irbs_scaled_data, features, targets)

        # Get Voronoi volume
        if any(params_voronoi['compute_voronoi'].values()):
            n_interest = len(exp_dic['interest'])
            volume_voronoi[exp_key] = {
                'features': np.array([0]*n_interest),
                'targets': np.array([0]*n_interest)
            }
            if params_voronoi['compute_voronoi']['features']:
                volume_voronoi[exp_key]['features'] = get_volume_voronoi(
                    scaled_x_interest,
                    dim=len(features),
                    tol=params_voronoi['tol'],
                    isFilter=params_voronoi['isFilter']
                )
            if params_voronoi['compute_voronoi']['targets']:
                volume_voronoi[exp_key]['targets'] = get_volume_voronoi(
                    scaled_y_interest,
                    dim=len(variable_names),
                    tol=params_voronoi['tol'],
                    isFilter=params_voronoi['isFilter']
                )
        else:
            volume_voronoi = {}

    return dict(
        volume=volume,
        asvd=asvd,
        volume_voronoi=volume_voronoi
    )


def get_variables_for_plot(
    features: List[str],
    targets: List[str],
    plot_variables: Dict[str, Dict[str, Union[str, float]]],
):
    return { # Get aliases for features and targets
        'feature_aliases': [plot_variables[name]['alias'] for name in features],
        'target_aliases': [plot_variables[name]['alias'] for name in targets],
        'latex_mapper': {dic['alias']: dic['latex'] for dic in plot_variables.values()},
        'alias_scales': {dic['alias']: dic['scale'] for dic in plot_variables.values()},
    }


def scale_variables_for_plot(
    data: Dict[str, Union[str, pd.DataFrame]],
    features: List[str],
    targets: List[str],
    feature_aliases: List[str],
    target_aliases: List[str],
    alias_scales: Dict[str, float],
    variables_ranges: Dict[str, Dict[str, Dict]],
    interest_region: Dict[str, Tuple[float, float]]
):
    """
    Scales features and targets to match physical unit in aliases. For example,
    if alias unit is MPa, divide by 1e6 or if alias unit is micron divide by
    1e-6.
    """
    variable_names = features + targets
    variable_aliases = feature_aliases + target_aliases

    # Scale features and targets
    alias_mapper = { # New names for data columns
        name: alias for name, alias in 
        zip(variable_names, variable_aliases)
    }
    variable_scales = [alias_scales[alias] for alias in variable_aliases]  # Scale of each variable
    for exp_dic in data.values():
        for df in exp_dic.values():
            if isinstance(df, pd.DataFrame):
                # Rename features and targets, use inplace to update data dict 
                df.rename(columns=alias_mapper, inplace=True) 
                # Scale features and targets
                df[variable_aliases] /= variable_scales 

    # Set variable bounds
    scaled_plot_ranges = {}
    margin_ratio = 0.1
    for name, alias in zip(variable_names, variable_aliases):
        bounds = variables_ranges['base'][name]['bounds']
        scaled_bounds = [v/alias_scales[alias] for v in bounds]
        margin = (scaled_bounds[1] - scaled_bounds[0]) * margin_ratio
        scaled_plot_ranges[alias] = [
            scaled_bounds[0] - margin,
            scaled_bounds[1] + margin
        ]

    # Scale interest region
    scaled_interest_region = {}
    for name, alias in zip(targets, target_aliases):
        interval = interest_region[name]
        scaled_interest_region[alias] = [v/alias_scales[alias] for v in interval]

    return dict(
        scaled_data=data,
        scaled_plot_ranges=scaled_plot_ranges,
        scaled_interest_region=scaled_interest_region
    )


def plot_metrics(
    data: Dict[str, Union[str, pd.DataFrame]],
    feature_aliases: List[str],
    target_aliases: List[str],
    latex_mapper: Dict[str, str],
    plot_ranges: Dict[str, Tuple[float, float]],
    interest_region: Dict[str, Tuple[float, float]],
    volume: Dict[str, float],
    asvd: Dict[str, ASVD],
    volume_voronoi: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
):
    exp_config = {exp_key: {key: data[exp_key][key] for key in ['name', 'color']} for exp_key in data}

    # Initial data distribution
    initial_data_plot = plot_initial_data(data, feature_aliases, target_aliases, latex_mapper, plot_ranges, only_first_exp=True)

    # plot features pairs
    feature_pairs_plot_dict = {}
    for feat_1, feat_2 in combinations(feature_aliases, 2):
        idx_1, idx_2 = feature_aliases.index(feat_1) + 1, feature_aliases.index(feat_2) + 1
        feature_pairs_plot_dict.update({
            f'X_{idx_1}_{idx_2}'         : plot_feature_pairs(data, (feat_1, feat_2), latex_mapper, plot_ranges, only_new=False),
            f'X_{idx_1}_{idx_2}_only_new': plot_feature_pairs(data, (feat_1, feat_2), latex_mapper, plot_ranges, only_new=True)
        })

    # targets distribution
    targets_plot_dict = {
        'y_violin': plot_violin_distribution(data, target_aliases, latex_mapper, interest_region, plot_ranges),
        'y_kde': targets_kde(data, asvd, target_aliases, latex_mapper, interest_region, plot_ranges),
    }

    # Distribution analysis
    distribution_plots_dict = {}
    if asvd:
        distribution_plots_dict['ASVD_all_exp'] = plot_multiple_asvd_distributions(asvd, exp_config)
        for i, exp_key in enumerate(data.keys()):
            distribution_plots_dict[f'ASVD_exp{i+1}'] = plot_stars_volumes_distribution(asvd[exp_key], exp_config[exp_key]['name'])
    if volume_voronoi:
        distribution_plots_dict['Voronoi'] = dist_volume_voronoi(data, volume_voronoi)
    if volume:
        print(f"Design space volumes: {volume}")

    # Detailed features versus targets plots
    feat_tar_plots_dict = {}
    for i, exp_key in enumerate(data.keys()):
        feat_tar_plots_dict[f'X_y_exp{i+1}'] = plot_feat_tar(data[exp_key], feature_aliases, target_aliases, latex_mapper, plot_ranges)

    # Aggregate plots
    all_plots = [
        {'initial_data': initial_data_plot},
        feature_pairs_plot_dict,
        targets_plot_dict,
        distribution_plots_dict,
        feat_tar_plots_dict
    ]
    plots_dict = {}
    for i, dic in enumerate(all_plots):
        # Index plots by groups
        dic = {f'{i+1:02d}_{k}': v for k, v in dic.items()}
        plots_dict.update(dic)

    # Save plots
    plots_paths_png = {}
    plots_paths_svg = {}

    output_dir_png = f"{output_dir}/png_outputs/"
    output_dir_svg = f"{output_dir}/svg_outputs/"
    os.makedirs(output_dir_png, exist_ok=True)
    os.makedirs(output_dir_svg, exist_ok=True)
    
    for plot_name, plot in plots_dict.items():
        png_path = os.path.join(output_dir_png, f'{plot_name}.png')
        svg_path = os.path.join(output_dir_svg, f'{plot_name}.svg')

        plot.savefig(png_path, format='png')
        plot.savefig(svg_path, format='svg')
        
        plots_paths_png[plot_name] = png_path
        plots_paths_svg[plot_name] = svg_path
        
        plt.close(plot)
    
    return None