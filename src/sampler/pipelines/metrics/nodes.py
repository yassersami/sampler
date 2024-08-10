"""
This is a boilerplate pipeline 'metrics'
generated using Kedro 0.18.5
"""
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import norm

from sampler.common.data_treatment import DataTreatment
from sampler.pipelines.metrics.asvd import ASVD
from sampler.pipelines.metrics.postprocessing_functions import create_dict, prepare_new_data, prepare_benchmark, get_result
from sampler.pipelines.metrics.volume import covered_space_bound
from sampler.pipelines.metrics.voronoi import get_volume_voronoi
import sampler.pipelines.metrics.graphics_metrics as gm


def prepare_data_metrics(
        experiments: Dict, names: Dict, features: List[str], targets: List[str],
        additional_values: List[str], treatment: DataTreatment
) -> Dict:
    data = {}
    f_r = names["features"]["str"]
    t_r = names["targets"]["str"]
    targets_prediction = [f'{t}_hat' for t in targets]
    renaming_cols = {v1: v2 for v1, v2 in zip(features + targets, f_r + t_r)}
    # * TODO: Don't use magic number 1e6, find a way to generalize
    # region = {
    #     t_r[0]: [v/1e6 for v in interest_region[targets[0]]],
    #     t_r[1]: interest_region[targets[1]]
    # }
    for key, value in experiments.items():
        if value["scale"] == "classify":
            # * TODO: Solve this:
            #  ParserWarning: Falling back to the 'python' engine because the 'c' engine                
            #  does not support regex separators (separators > 1 char and different from                
            #  '\s+' are interpreted as regex); you can avoid this warning by specifying                
            #  engine='python'.
            # TODO yasser: prepare_benchmark is now replaced by case of value["scale"] == "read"
            df = prepare_benchmark(
                df=pd.read_csv(
                    value["path"], sep='[; ,]',
                    usecols=features + targets + additional_values
                ),
                f=features, t=targets, treatment=treatment
            ).rename(columns=renaming_cols)
        elif value["scale"] == "real-inverse":
            history = get_result(value["path"])
            df = prepare_new_data(
                df=history, treatment=treatment, f=features, t=targets, t_c=targets_prediction
            ).rename(columns=renaming_cols)
        elif value["scale"] == "read": # TODO : Rewrite properly this part and "classify"
            df_read = pd.read_csv(value["path"], sep='[; ,]', usecols=features+targets+additional_values+["quality"])
            df = prepare_new_data(
                df=df_read, treatment=treatment, f=features, t=targets, t_c=targets_prediction
            ).rename(columns=renaming_cols)

        else:
            print(f'{value["scale"]} is not a valid scaler for the data. Exiting program')
            sys.exit(1)

        data[key] = create_dict(df=df, name=value["name"], color=value["color"])
    return dict(
        exp_data=data,
        features=f_r,
        targets=t_r,
        targets_prediction=targets_prediction
    )


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
    volume_voronoi = {} # for each experiment, volumes of the Voronoi regions (clipped by the unit hypercube) : in feature space and feature+target space
    
    for key, value in data.items():
        scaled_data_interest_f = treatment.scaler.transform_features(value["interest"][features].values)
        scaled_data_interest_t = treatment.scaler.transform_targets(value["interest"][targets].values)

        # Get number of interesting samples
        n_interest[key] = len(value["interest"])
        
        # Get volume of interesting samples
        if key in params_volume["default"]:
            volume[key] = params_volume["default"][key]
        elif params_volume["compute_volume"]:
            volume[key] = covered_space_bound(scaled_data_interest_f, radius, params_volume, len(features))
        else:
            volume[key] = np.array([0,0])
        
        # Get all data distribution using ASVD
        XY = value["df"][features+targets].values
        scaled_data = pd.DataFrame(treatment.scaler.transform(XY), columns=features+targets)
        total_asvd = ASVD(scaled_data, features, targets)
        total_asvd_scores[key] = total_asvd.compute_scores()
        
        # Get only interest data distribution using ASVD
        XY = value["interest"][features+targets].values
        scaled_data = pd.DataFrame(treatment.scaler.transform(XY), columns=features+targets)
        interest_asvd = ASVD(scaled_data, features, targets)
        interest_asvd_scores[key] = interest_asvd.compute_scores()

        # Get Voronoi volume
        # volume_voronoi[key] = {
        #     "features": np.array([0]*n_interest[key]),
        #     "features_targets": np.array([0]*n_interest[key])
        # }
        # if params_voronoi["compute_voronoi"]["features"]:
        #     volume_voronoi[key]["features"] = get_volume_voronoi(
        #         scaled_data_interest_f,
        #         len(features),tol=params_voronoi["tol"], isFilter=params_voronoi["isFilter"]
        #     )
        # if params_voronoi["compute_voronoi"]["features_targets"]:
        #     volume_voronoi[key]["features_targets"] = get_volume_voronoi(
        #         np.hstack([scaled_data_interest_f, scaled_data_interest_t]),
        #         len(features+targets),tol=params_voronoi["tol"], isFilter=params_voronoi["isFilter"]
        #     )

    return dict(
        n_interest=n_interest,
        volume=volume,
        total_asvd_scores=total_asvd_scores,
        interest_asvd_scores=interest_asvd_scores,
        volume_voronoi=volume_voronoi
    )


def scale_data_for_plots(data: Dict, features: List[str], targets: List[str], targets_prediction: List[str], scales: Dict, interest_region: Dict):
    """Scales data in place for visualization purposes."""
    df_names = ["interest", "not_interesting", "inliers", "outliers", "df"]
    for v in data.values():
        for name in df_names:
            v[name][features] /= scales["features"]
            v[name][targets] /= scales["targets"]
            # Check if every element of targets prediction is in v[name].columns
            if all([t in v[name].columns for t in targets_prediction]):
                v[name][targets_prediction] /= scales["targets"]
            
    scaled_interest_region = {}
    for region, target, target_scale in zip(interest_region.values(), targets, scales["targets"]):
        scaled_interest_region[target] = [v / target_scale for v in region]

    return dict(
        scaled_data=data,
        scaled_region=scaled_interest_region
    )


def plot_metrics(
    data: Dict, names: Dict, region: Dict,
    ignition_points: Dict, volume: Dict,
    total_asvd_scores: Dict[str, Dict[str, float]],
    interest_asvd_scores: Dict[str, Dict[str, float]],
    volume_voronoi: Dict

    # r2: Dict, crps: Dict
):
    features_dic = names["features"]
    features = features_dic["str"]
    targets = names["targets"]["str"]
    asvd_metrics_to_plot = ["sum_augm", "rsd_x", "rsd_xy", "rsd_augm", "riqr_x", "riqr_xy"]

    targets_volume = None  # TODO yasser: compute covered area on targets space
    # targets_volume = {k: 10000 for k in data.columns}
    # Space distribution
    features_2d = gm.plot_2d(data, features_dic, ignition_points, volume)
    violin_plot = gm.plot_violin_distribution(data, targets, region, targets_volume)
    kde_plot = gm.targets_kde(data, targets, region)
    # pair_plot = gm.pair_grid_for_all_variables(data, features, targets)
    feat_tar_dict = {}
    feat_tar_dict["all"] = gm.plot_feat_tar(data, features, targets, only_interest=False)
    feat_tar_dict["all_int"] = gm.plot_feat_tar(data, features, targets, only_interest=True, title_extension='(only interest)')
    for k in data.keys():
        feat_tar_dict[k] = gm.plot_feat_tar({k: data[k]}, features, targets, only_interest=False)
    total_asvd_plot = gm.plot_asvd_scores(data, total_asvd_scores, asvd_metrics_to_plot)
    interest_asvd_plot = gm.plot_asvd_scores(data, interest_asvd_scores, asvd_metrics_to_plot)
    # voronoi_plot = gm.dist_volume_voronoi(data, volume_voronoi)

    # Saving dictionary of plots
    plots_dict = {
        "features_2d.png": features_2d,
        "violin_plot.png": violin_plot,
        "targets_kde.png": kde_plot,
        # "pair_plot.png": pair_plot,
        **{f'features_targets_{k}.png': v for k, v in feat_tar_dict.items()},
        "ASVD_all.png": total_asvd_plot,
        "ASVD_interest.png": interest_asvd_plot,
        # "volume_voronoi.png": voronoi_plot
    }
    plots_dict = {f'{i+1:02d}_{k}': v for i, (k, v) in enumerate(plots_dict.items())}
    return plots_dict