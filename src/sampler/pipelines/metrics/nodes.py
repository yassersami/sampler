"""
This is a boilerplate pipeline 'metrics'
generated using Kedro 0.18.5
"""
from typing import List, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .postprocessing_functions import (
    categorize_df_by_quality, prepare_new_data, aggregate_csv_files
)
from .volume import covered_space_bound
from .asvd import ASVD
from .voronoi import get_volume_voronoi
import sampler.pipelines.metrics.graphics_metrics as gm
from sampler.common.data_treatment import DataTreatment


def prepare_data_metrics(
    experiments: Dict, names: Dict, features: List[str], targets: List[str],
    additional_values: List[str], treatment: DataTreatment
) -> Dict:
    """
    Prepare data metrics by processing experiment data based on the path type.
    """
    data = {}
    # New names for data columns
    feature_aliases = names['features']['str']
    target_aliases = names['targets']['str']
    column_renaming = {
        orig: alias for orig, alias in 
        zip(features + targets, feature_aliases + target_aliases)
    }
    # Name fot target prediction columns
    targets_pred = [f'{target}_hat' for target in targets]

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
            df=imported_df, treatment=treatment, f=features, t=targets,
            t_c=targets_pred
        )
        df = df.rename(columns=column_renaming)

        data[exp_id] = categorize_df_by_quality(
            df=df, name=exp_config['name'], color=exp_config['color']
        )

    return {
        'exp_data': data,
        'features': feature_aliases,
        'targets': target_aliases,
        'targets_prediction': targets_pred
    }


def get_metrics(
        data: Dict, features: List[str], targets: List[str],
        treatment: DataTreatment,
        params_volume: Dict, params_voronoi: Dict
) -> Dict:
    radius = 0.025 # Sphere radius

    n_interest = {} # for each experiment, number of interest points (dict of int)
    volume = {} # for each experiment, volume space covered (dict of float)
    total_asvd_scores = {}
    interest_asvd_scores = {}
    volume_voronoi = {} # for each experiment, volumes of the Voronoi regions (clipped by the unit hypercube)
    
    for key, value in data.items():
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

        # Get number of interesting samples
        n_interest[key] = len(value['interest'])
        
        # Get volume of interesting samples
        if key in params_volume['default']:
            volume[key] = params_volume['default'][key]
        elif params_volume['compute_volume']:
            volume[key] = covered_space_bound(
                scaled_x_interest, radius, params_volume, len(features)
            )
        else:
            volume[key] = np.array([0,0])
        
        # Get all data distribution using ASVD
        total_asvd = ASVD(scaled_data, features, targets)
        total_asvd_scores[key] = total_asvd.compute_scores()
        
        # Get only interest data distribution using ASVD
        interest_asvd = ASVD(scaled_data_interest, features, targets)
        interest_asvd_scores[key] = interest_asvd.compute_scores()

        # Get Voronoi volume
        volume_voronoi[key] = {
            'features': np.array([0]*n_interest[key]),
            'targets': np.array([0]*n_interest[key])
        }
        if params_voronoi['compute_voronoi']['features']:
            volume_voronoi[key]['features'] = get_volume_voronoi(
                scaled_x_interest,
                dim=len(features),
                tol=params_voronoi['tol'],
                isFilter=params_voronoi['isFilter']
            )
        if params_voronoi['compute_voronoi']['targets']:
            volume_voronoi[key]['targets'] = get_volume_voronoi(
                scaled_y_interest,
                dim=len(features+targets),
                tol=params_voronoi['tol'],
                isFilter=params_voronoi['isFilter']
            )

    return dict(
        n_interest=n_interest,
        volume=volume,
        total_asvd_scores=total_asvd_scores,
        interest_asvd_scores=interest_asvd_scores,
        volume_voronoi=volume_voronoi
    )


def scale_data_for_plots(
    data: Dict, features: List[str], targets: List[str],
    targets_prediction: List[str], scales: Dict, interest_region: Dict
):
    """Scales data in place for visualization purposes."""
    df_names = ['interest', 'no_interest', 'inliers', 'outliers', 'df']
    for v in data.values():
        for name in df_names:
            v[name][features] /= scales['features']
            v[name][targets] /= scales['targets']
            # Check if every element of targets prediction is in v[name].columns
            if all([t in v[name].columns for t in targets_prediction]):
                v[name][targets_prediction] /= scales['targets']
            
    scaled_interest_region = {}
    for region, target, target_scale in zip(interest_region.values(), targets, scales['targets']):
        scaled_interest_region[target] = [v / target_scale for v in region]

    return dict(
        scaled_data=data,
        scaled_region=scaled_interest_region
    )


def plot_metrics(
    env_name: str,
    data: Dict,
    names: Dict,
    region: Dict,
    volume: Dict,
    total_asvd_scores: Dict[str, Dict[str, float]],
    interest_asvd_scores: Dict[str, Dict[str, float]],
    volume_voronoi: Dict
):
    features_dic = names['features']
    features = features_dic['str']
    targets = names['targets']['str']
    asvd_metrics_to_plot = ['sum_augm', 'rsd_x', 'rsd_xy', 'rsd_augm', 'riqr_x', 'riqr_xy']

    targets_volume = None  # TODO yasser: compute covered area on targets space
    # targets_volume = {k: 10000 for k in data.columns}

    # Features and targets space viz
    features_2d = gm.plot_2d(data, features_dic, volume)
    violin_plot = gm.plot_violin_distribution(data, targets, region, targets_volume)
    kde_plot = gm.targets_kde(data, targets, region)

    # Distribution analysis
    total_asvd_plot = gm.plot_asvd_scores(data, total_asvd_scores, asvd_metrics_to_plot)
    interest_asvd_plot = gm.plot_asvd_scores(data, interest_asvd_scores, asvd_metrics_to_plot)
    voronoi_plot = gm.dist_volume_voronoi(data, volume_voronoi)

    # Detailed features versus targets plots
    # pair_plot = gm.pair_grid_for_all_variables(data, features, targets)
    feat_tar_dict = {}
    feat_tar_dict['all'] = gm.plot_feat_tar(data, features, targets, only_interest=False)
    feat_tar_dict['all_int'] = gm.plot_feat_tar(data, features, targets, only_interest=True, title_extension='(only interest)')
    for k in data.keys():
        feat_tar_dict[k] = gm.plot_feat_tar({k: data[k]}, features, targets, only_interest=False)

    plots_dict = {
        'features_2d': features_2d,
        'violin_plot': violin_plot,
        'targets_kde': kde_plot,
        # 'pair_plot': pair_plot,
        'ASVD_all': total_asvd_plot,
        'ASVD_interest': interest_asvd_plot,
        'volume_voronoi': voronoi_plot,
        **{f'features_targets_{k}': v for k, v in feat_tar_dict.items()},
    }
    plots_dict = {f'{i+1:02d}_{k}': v for i, (k, v) in enumerate(plots_dict.items())}

    plots_paths_png = {}
    plots_paths_svg = {}

    output_dir_png = f"data/08_reporting/{env_name}/png_outputs/"
    output_dir_svg = f"data/08_reporting/{env_name}/svg_outputs/"
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