"""
This is a boilerplate pipeline 'metrics'
generated using Kedro 0.18.5
"""
from typing import List, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

from .postprocessing_functions import (
    categorize_df_by_quality, prepare_new_data, aggregate_csv_files
)
from .volume import covered_space_bound
from .asvd import ASVD
from .voronoi import get_volume_voronoi
import sampler.pipelines.metrics.graphics_metrics as gm
from sampler.common.data_treatment import DataTreatment


def prepare_data_metrics(
    experiments: Dict,
    variable_aliases: Dict,
    features: List[str],
    targets: List[str],
    treatment: DataTreatment
) -> Dict:
    """
    Prepare data metrics by processing experiment data based on the path type.
    """
    data = {}
    # New names for data columns
    feature_aliases = variable_aliases['features']['str']
    target_aliases = variable_aliases['targets']['str']
    column_renaming = {
        orig: alias for orig, alias in 
        zip(features + targets, feature_aliases + target_aliases)
    }

    for exp_id, exp_config in experiments.items():
        file_path = exp_config['path']
        
        # Import data, etiher from a folder or a csv file
        if os.path.isdir(file_path):
            imported_df = aggregate_csv_files(file_path)
        elif os.path.isfile(file_path) and file_path.endswith('.csv'):
            imported_df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Path '{file_path}' is neither a valid directory "
                             "nor a CSV file. Exiting program.")

        df = prepare_new_data(
            df=imported_df, treatment=treatment, f=features, t=targets
        )
        df = df.rename(columns=column_renaming)

        data[exp_id] = categorize_df_by_quality(
            df=df, name=exp_config['name'], color=exp_config['color']
        )

    return {
        'exp_data': data,
        'features': feature_aliases,
        'targets': target_aliases,
    }


def get_metrics(
        data: Dict,
        features: List[str],
        targets: List[str],
        treatment: DataTreatment,
        params_volume: Dict,
        params_asvd: Dict,
        params_voronoi: Dict
) -> Dict:
    radius = 0.025 # Sphere radius

    n_interest = {} # for each experiment, number of interest points (dict of int)
    volume = {} # for each experiment, volume space covered (dict of float)
    total_asvd_scores = {}
    interest_asvd_scores = {}
    volume_voronoi = {} # for each experiment, volumes of the Voronoi regions (clipped by the unit hypercube)
    
    for exp_key, value in data.items():
        # Scale all data
        XY = value['df'][features+targets].values
        scaled_data = pd.DataFrame(
            treatment.scaler.transform(XY),
            columns=features+targets
        )
        # Scale interest data
        XY_interest = value['interest'][features+targets].values
        scaled_data_interest = pd.DataFrame(
            treatment.scaler.transform(XY_interest),
            columns=features+targets
        )
        scaled_x_interest = scaled_data_interest[features]
        scaled_y_interest = scaled_data_interest[targets]

        # Get volume of interesting samples
        if exp_key in params_volume['default']:
            # Use default values
            volume[exp_key] = params_volume['default'][exp_key]
        elif params_volume['compute_volume']:
            # Compute volume
            volume[exp_key] = covered_space_bound(
                scaled_x_interest, radius, params_volume, len(features)
            )
        else:
            volume[exp_key] = np.array([0,0])

        if params_asvd['compute_asvd']:
            # Get all data distribution using ASVD
            total_asvd = ASVD(scaled_data, features, targets)
            total_asvd_scores[exp_key] = total_asvd.compute_scores()

            # Get only interest data distribution using ASVD
            interest_asvd = ASVD(scaled_data_interest, features, targets)
            interest_asvd_scores[exp_key] = interest_asvd.compute_scores()
        else:
            total_asvd_scores = {}
            interest_asvd_scores = {}

        # Get Voronoi volume
        if any(params_voronoi['compute_voronoi'].values()):
            n_interest = len(value['interest'])
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
                    dim=len(features+targets),
                    tol=params_voronoi['tol'],
                    isFilter=params_voronoi['isFilter']
                )
        else:
            volume_voronoi = {}

    return dict(
        volume=volume,
        total_asvd_scores=total_asvd_scores,
        interest_asvd_scores=interest_asvd_scores,
        volume_voronoi=volume_voronoi
    )


def scale_data_for_plots(
    data: Dict,
    features: List[str],
    targets: List[str],
    scales: Dict,
    interest_region: Dict
):
    """Scales data in place for visualization purposes."""
    df_names = ['interest', 'no_interest', 'inliers', 'outliers', 'df']
    for v in data.values():
        for name in df_names:
            v[name][features] /= scales['features']
            v[name][targets] /= scales['targets']

    scaled_interest_region = {}
    for region, target, target_scale in zip(interest_region.values(), targets, scales['targets']):
        scaled_interest_region[target] = [v / target_scale for v in region]

    return dict(
        scaled_data=data,
        scaled_region=scaled_interest_region
    )


def plot_metrics(
    data: Dict,
    variable_aliases: Dict,
    region: Dict,
    volume: Dict,
    total_asvd_scores: Dict[str, Dict[str, float]],
    interest_asvd_scores: Dict[str, Dict[str, float]],
    volume_voronoi: Dict,
    output_dir: str,
):
    features_dic = variable_aliases['features']
    features = features_dic['str']
    targets = variable_aliases['targets']['str']
    asvd_metrics_to_plot = ['sum_augm', 'rsd_x', 'rsd_xy', 'rsd_augm', 'riqr_x', 'riqr_xy']

    # Initial data distribution
    initial_data_plot = gm.plot_initial_data(data, features, targets, only_first_exp=True)

    # plot features pairs
    feature_pairs_plot_dict = {}
    for feat_1, feat_2 in combinations(features, 2):
        idx_1, idx_2 = features.index(feat_1) + 1, features.index(feat_2) + 1
        feature_pairs_plot_dict.update({
            f'X_{idx_1}_{idx_2}'         : gm.plot_feature_pairs(data, features_dic, (feat_1, feat_2), only_new=False),
            f'X_{idx_1}_{idx_2}_only_new': gm.plot_feature_pairs(data, features_dic, (feat_1, feat_2), only_new=True)
        })

    # targets distribution
    targets_plot_dict = {
        'y_violin': gm.plot_violin_distribution(data, targets, region),
        'y_kde': gm.targets_kde(data, targets, region),
    }

    # Distribution analysis
    print(f"Design space volumes: {volume}")
    distribution_plots_dict = {}
    if total_asvd_scores:
        distribution_plots_dict['ASVD'] = gm.plot_asvd_scores(data, total_asvd_scores, asvd_metrics_to_plot)
    if interest_asvd_scores:
        distribution_plots_dict['ASVD_only_interest'] = gm.plot_asvd_scores(data, interest_asvd_scores, asvd_metrics_to_plot)
    if volume_voronoi:
        distribution_plots_dict['Voronoi'] = gm.dist_volume_voronoi(data, volume_voronoi)

    # Detailed features versus targets plots
    feat_tar_plots_dict = {}
    feat_tar_plots_dict['X_y'] = gm.plot_feat_tar(data, features, targets, only_interest=False)
    feat_tar_plots_dict['X_y_only_interest'] = gm.plot_feat_tar(data, features, targets, only_interest=True, title_extension='(only interest)')
    for i, exp_key in enumerate(data.keys()):
        feat_tar_plots_dict[f'X_y_exp{i+1}'] = gm.plot_feat_tar({exp_key: data[exp_key]}, features, targets, only_interest=False)

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