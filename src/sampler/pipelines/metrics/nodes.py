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
from sampler.models.fom import GPSampler
from sampler.pipelines.metrics.postprocessing_functions import create_dict, prepare_new_data, prepare_benchmark, get_result
from sampler.pipelines.metrics.volume import covered_space_upper
import sampler.pipelines.metrics.graphics_metrics as gm

def crps_norm(mu, sigma, y):
    z = (y-mu)/sigma
    crps = sigma*(z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))
    return np.mean(crps)


def prepare_data_metrics( # the name is the same as analysis. Interpreter will not be able to distinguish between them (idk why)
        experiments: Dict, names: Dict, features: List[str], targets: List[str],
        additional_values: List[str], treatment: DataTreatment
) -> Dict:
    data = {}
    f_r = names['features']
    t_r = names['targets']
    targets_prediction = [f'{t}_hat' for t in targets]
    renaming_cols = {v1: v2 for v1, v2 in zip(features + targets, f_r + t_r)}
    # region = {
    #     # TODO: Don't use magic number 1e6, find a way to generalize
    #     t_r[0]: [v/1e6 for v in interest_region[targets[0]]],
    #     t_r[1]: interest_region[targets[1]]
    # }
    for key, value in experiments.items():
        if value['scale'] == 'classify':
            df = prepare_benchmark(
                df=pd.read_csv(value["path"],
                               sep='[; ,]', # TODO: Solve this:
                                            #  ParserWarning: Falling back to the 'python' engine because the 'c' engine                
                                            #  does not support regex separators (separators > 1 char and different from                
                                            #  '\s+' are interpreted as regex); you can avoid this warning by specifying                
                                            #  engine='python'.
                               usecols=features+targets+additional_values),
                f=features, t=targets, treatment=treatment
            ).rename(columns=renaming_cols)
        elif value['scale'] == 'real-inverse':
            history = get_result(value['path'])
            df = prepare_new_data(
                df=history, treatment=treatment, f=features, t=targets, t_c=targets_prediction
            ).rename(columns=renaming_cols)
        else:
            print(f"{value['scale']} is not a valid scaler for the data. Exiting program")
            sys.exit(1)

        data[key] = create_dict(df=df, name=value['name'], color=value['color'])
    return dict(exp_data=data, features=f_r, targets=t_r, targets_prediction=targets_prediction)

def get_metrics(
        data: Dict, features: List[str], targets: List[str],
        treatment: DataTreatment,
        params_volume: Dict, initial_size: int
) -> Dict:
    radius = 0.025 # Sphere radius

    n_interest = {} # for each experiment, number of interest points (dict of int)
    volume = {} # for each experiment, volume space covered (dict of float)
    # for each experiment, r2 score according to train_set (data_init+80% new points) and test_set (20% last new points) 
    r2 = {} # r2 score : r2_target1, r2_target2, ..., r_2_target_norm (dict of array (shape=(1, len(targets)+1)) of float)
    crps = {}

    for key, value in data.items():
        # Get number of interesting samples
        n_interest[key] = len(value['interest'])
        # Get volume of interesting samples
        if key in params_volume["default"]:
            volume[key] = params_volume["default"][key]
        elif params_volume["compute_volume"]:
            scaled_data_interest = treatment.scaler.transform_features(value['interest'][features].values)
            volume[key] = covered_space_upper(scaled_data_interest, radius, params_volume)
        else:
            volume[key] = 0

        # Train surrogate model
        len_train = initial_size + int( 0.8*(len(value['df']) - initial_size) )  # so length of test_set : 20% of the new points 
        train_set = value['df'][:len_train]
        test_set = value['df'][len_train:]
        X_train, Y_train = treatment.scaler.transform_features(train_set[features]),  treatment.scaler.transform_targets(train_set[targets])
        X_test, Y_test =  treatment.scaler.transform_features(test_set[features]),  treatment.scaler.transform_targets(test_set[targets])
        
        gp = GPSampler(features=features, targets=targets)
        gp.fit(X_train, Y_train)

        # Get scores on targets prediction
        Y_pred, sigma = gp.predict(X_test, return_std=True)

        r2[key] = {}
        for i, name_target in enumerate(targets):
            r2[key][name_target] = r2_score(Y_test[:,i], Y_pred[:,i])
        r2[key]['all_targets'] = r2_score(Y_test, Y_pred)

        crps[key] = {}
        for i, name_target in enumerate(targets):
            crps[key][name_target] = crps_norm(Y_pred[:,i], sigma[:,i], Y_test[:,i])
        # crps[key]['all_targets'] = np.mean([crps[key][e] for e in targets]) # Mean it's not the real CRPS for all targets

    return dict(n_interest=n_interest, volume=volume, r2=r2, crps=crps)

def scale_data_for_plots(data: Dict, features: List[str], targets: List[str], targets_prediction: List[str], scales: Dict, interest_region: Dict):
    """Scales data in place for visualization purposes."""
    df_names = ['interest', 'not_interesting', 'inliers', 'outliers', 'df']
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

    return dict(scaled_data=data, scaled_region=scaled_interest_region)


def plot_metrics(
    data: Dict, features: List[str], targets: List[str], region: Dict,
    ignition_points: Dict, volume: Dict,
    # r2: Dict, crps: Dict
):
    area_targets = None  # TODO yasser: compute covered area on targets space
    # area_targets = {k: 10000 for k in data}
    # Space distribution
    features_2d = gm.plot_2d(data, features, ignition_points, volume)
    violin_plot = gm.plot_violin_distribution(data, targets, region, area_targets)
    kde_plot = gm.targets_kde(data, targets, region)
    # pair_plot = gm.pair_grid_for_all_variables(data, features, targets)
    feat_tar_dict = {}
    feat_tar_dict["all"] = gm.plot_feat_tar(data, features, targets, only_interest=False)
    feat_tar_dict["all_int"] = gm.plot_feat_tar(data, features, targets, only_interest=True, title_extension='(only interest)')
    for k in data.keys():
        feat_tar_dict[k] = gm.plot_feat_tar({k: data[k]}, features, targets, only_interest=False)

    # Surrogate performance
    # r2_plot = gm.r2_bar_plot(data, targets, r2, 'R2 Scores',  all_targets=True)
    # r2_plot_crps = gm.r2_bar_plot(data, targets, crps, 'CRPS Scores', all_targets=False)

    # Saving dictionary of plots
    plots_dict = {
        "features_2d.png": features_2d,
        "violin_plot.png": violin_plot,
        "targets_kde.png": kde_plot,
        # "pair_plot.png": pair_plot,
        **{f'features_targets_{k}.png': v for k, v in feat_tar_dict.items()},
        # "bar_plot.png": r2_plot,
        # "bar_plot_crps.png": r2_plot_crps,
    }
    plots_dict = {f'{i+1:02d}_{k}': v for i, (k, v) in enumerate(plots_dict.items())}
    return plots_dict