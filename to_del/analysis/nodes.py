"""
This is a boilerplate pipeline 'analysis'
generated using Kedro 0.18.5
"""
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
from itertools import combinations, combinations_with_replacement
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime

from sklearn.cluster import DBSCAN
from vedo import Sphere, Cube

from sampler.common.data_treatment import DataTreatment
from sampler.pipelines.analysis.graphics import plot_scatter_for_nn, \
    plot_histogram_for_nn, plot_scatter_y, plot_y_histogram, pair_grid_for_all_variables, \
    plot_violin_distribution, plot_2d, targets_kde, plot_scatter_y_1d, plot_y_histogram_1d, plot_feat_tar

from sampler.pipelines.analysis.postprocessing_functions import get_result, prepare_benchmark, create_dict, \
    prepare_new_data


def plot_targets(
        data: Dict, targets: List[str], region: Dict
):
    return dict(
        violin_plot=plot_violin_distribution(data, targets, region),
        kde_plot=targets_kde(data, targets, region)
    )


def plot_multi_analysis(data: Dict, features: List[str], targets: List[str]):
    return pair_grid_for_all_variables(data, features, targets)


def evaluate_surrogate(data: Dict, targets: List[str], prediction: List[str]):
    data = {key: val for key, val in data.items() if (key != 'lhs')}
    if len(targets) == 1:
        scatter = plot_scatter_y_1d(data, prediction, targets)
        hist = plot_y_histogram_1d(data, prediction, targets)
    else:
        scatter = plot_scatter_y(data, prediction, targets)
        hist = plot_y_histogram(data, prediction, targets)
    return dict(y_scatter_plot=scatter, y_histogram_plot=hist)


def analyze_nn(data: Dict, features: List[str], features_nn: List[str]):
    data_dic = {key: val for key, val in data.items() if (key != 'lhs') & ~ (val['name'].endswith('pf'))}
    scatter = plot_scatter_for_nn(data_dic, features, features_nn)
    hist = plot_histogram_for_nn(data_dic, features, features_nn)
    return dict(x_scatter_plot=scatter, x_histogram_plot=hist)


def analyze_ignition_points(data: Dict, features: List[str], targets: List[str], points: Dict):
    #colors = {'interest': "#E02401", 'not_interesting': "#A9B388", 'outliers': "black"}
    features_2d = plot_2d(data=data, features=features, points=points)
    feat_tar_interest = plot_feat_tar(data=data, features=features, targets=targets, only_interest=True)
    feat_tar_all = plot_feat_tar(data=data, features=features, targets=targets, only_interest=False)
    return dict(features_2d=features_2d, feat_tar_interest=feat_tar_interest, feat_tar_all=feat_tar_all)


def plot_constraints(data: Dict, initial_size: int, coefficients: Dict):
    data = {k: v for k, v in data.items() if k != 'lhs'}
    n_cols = len(data)
    fig, axs = plt.subplots(1, n_cols, figsize=(n_cols*5, 5), sharey='row')
    for idx, val in enumerate(data.values()):
        constraints = set(coefficients.keys()).intersection(val['df'].columns)
        val["df"].loc[initial_size:, constraints].plot(ax=axs[idx])
        axs[idx].set_xlabel('Guided steps')
        axs[idx].set_ylabel('Value of constraint')
        axs[idx].set_title(val['name'])
    return fig


def covered_space_euclidean(points):
    """
    Compute the total sum of all edges in a fully connected graph formed by a set of 3D points.

    Parameters:
    points (numpy.ndarray): An array of shape (N, 3) containing N 3D points.

    Returns:
    float: The total sum of all edges in the fully connected graph.
    """
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    # Compute the distances of unique pairs and sum them up
    total_edge_length = np.sum([distance(p1, p2) for p1, p2 in combinations_with_replacement(points, 2)])

    # Take square root of distance
    total_edge_length = np.sqrt(total_edge_length)

    # Change to scientific notation
    #total_edge_length = f"{total_edge_length:.2e}"

    return total_edge_length

def near_cube_border(point, radius):
    # Check distance to each face of the cube
    distances = np.array([
        point[0], 1 - point[0],  # Distance to the left / right face (x = 0 or x = 1)
        point[1], 1 - point[1],  # Distance to the back / front face (y = 0 or y = 1)
        point[2], 1 - point[2]   # Distance to the bottom / top face (z = 0 or z = 1)
    ])
    
    # Check if any distance is less than the radius
    near_faces = distances < radius

    return near_faces 

def compute_two_sphere_intersection_volume(center1, center2, radius):
    d = np.linalg.norm(center1 - center2)

    h = 2 * radius - d
    volume = np.pi * h ** 2 * (radius - h / 3)
    return volume

def compute_simple_case_volume(cluster_points, radius, out_of_cube):
    sphere_volume = (4 / 3) * np.pi * radius ** 3
    acum_volume = 0
    if len(cluster_points) == 1:
        # One sphere case
        acum_volume += sphere_volume
    elif len(cluster_points) == 2:
        # There is a simple equation for the volume of intersection of two spheres
        intersection = compute_two_sphere_intersection_volume(cluster_points[0], cluster_points[1], radius)
        volume = 2 * sphere_volume - intersection
        acum_volume += volume

    return acum_volume
   

def remove_volume_outside_cube(sphere, center, radius, point_borders):
    # Relevant: From near_cube_border()
    #     distances = np.array([
    #         point[0], 1 - point[0],  # Distance to the left / right face (x = 0 or x = 1)
    #         point[1], 1 - point[1],  # Distance to the back / front face (y = 0 or y = 1)
    #         point[2], 1 - point[2]   # Distance to the bottom / top face (z = 0 or z = 1)
    #         ])

    # For each side outside the cube, remove the volume of the sphere outside it
    for i, is_near in enumerate(point_borders):
        if is_near:
            # Remove volume outside the cube
            cube_center = center.copy()
            axis = i // 2 # 0 for x, 1 for y, 2 for z
            # Complicated way to set the cube center to the correct position. Read the "Relevant" comment above
            cube_center[axis] = -radius if i % 2 == 0 else 1 + radius
            outside_cube = Cube(pos=cube_center, side=2*radius)#.subdivide(1 method=1)
            sphere = sphere.boolean("minus", outside_cube)
    
    return sphere
            
def generate_valid_spheres(cluster_points, cluster_borders, radius):
    spheres = []
    # TODO: Add this to kedro parameters
    apply_cube_volume_removal = False
    for center, point_borders in zip(cluster_points, cluster_borders):
        sphere = Sphere(pos=center, r=radius, res=12)  # TODO add res param to kedro ? res=24
        if apply_cube_volume_removal and any(point_borders):
            # At least one side of the sphere is going to be outside the [0,1]^3 cube
            sphere = remove_volume_outside_cube(sphere, center, radius, point_borders)

        spheres.append(sphere)
    return spheres

def remove_very_close_points(cluster_points, radius):
    # TODO: Add this to kedro parameters
    same_sphere_tol = 1e-3
    
    if len(cluster_points) == 1:
        return cluster_points
    
    remaining_points = [cluster_points[0]]

    # Remove points closer than same_sphere_tol
    for i, point in enumerate(cluster_points[1:]):
        distances = np.linalg.norm(remaining_points - point, axis=1)
        if np.all(distances > same_sphere_tol):
            remaining_points.append(point)

    return np.array(remaining_points)

# Function to compute volume of intersection within a cluster
def compute_cluster_volume(cluster_points, radius):
    
    cluster_points = remove_very_close_points(cluster_points, radius)

    # Check for points near [0,1]^3 cube borders
    cluster_borders = [near_cube_border(point, radius) for point in cluster_points]
    out_of_cube = any([any(point_borders) for point_borders in cluster_borders])
    
    if len(cluster_points) <= 2 and not out_of_cube:
        return compute_simple_case_volume(cluster_points, radius, out_of_cube)

    spheres = generate_valid_spheres(cluster_points, cluster_borders, radius)
    
    cluster_union = spheres[0]
    if len(spheres) > 1:
        # if len(spheres) > 5:
        #     print(f'analysis.nodes.compute_cluster_volume -> spheres: {spheres}')
        #     print(f'analysis.nodes.compute_cluster_volume -> len(spheres): {len(spheres)}')
        for other_sphere in spheres[1:]:
            print(f'analysis.nodes.compute_cluster_volume -> datetime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            cluster_union = cluster_union.boolean("plus", other_sphere)  # TODO: can be very long when len(spheres) > 5
            print(f'analysis.nodes.compute_cluster_volume -> cluster_union: {other_sphere}')

    print(f'analysis.nodes.compute_cluster_volume -> end datetime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    return cluster_union.volume()

# Function to compute total volume of intersected spheres
def covered_space(points, radius, digits=7):
    tol = 10**(-digits)

    if points.shape[0] == 0:
        return 0

    points = points.round(digits)

    # Remove duplicated points (they occupy the same volume)
    points = np.unique(points, axis=0)

    # Clustering of touching spheres
    max_distance = 2 * radius - tol
    dbscan = DBSCAN(eps=max_distance, min_samples=1)
    clusters = dbscan.fit_predict(points)
    
    acum_volume = 0
    for cluster_id in tqdm(np.unique(clusters)):
        cluster_points = points[clusters == cluster_id]
        acum_volume += compute_cluster_volume(cluster_points, radius)  # Can be very long because of spheres union

    return acum_volume


def make_resume(data: Dict, initial_size: int, n_slice: int, tot_size: int, features: List, treatment: DataTreatment, digits: int):
    scaler = treatment.scaler
    test = {}

    # Sphere radius
    radius = 0.025
    
    for k, v in data.items():
        mini_test = {}
        print(f"Computing volumes for {k}: '{v['name']}'")
        for lim in np.arange(initial_size, tot_size + 1, n_slice):
            print(f"> From row {lim - n_slice} to {lim} of {tot_size}")
            num_interest = len(v['interest'].loc[v['interest'].index <= lim])
            interest_points = v['interest'].loc[v['interest'].index <= lim][features].values
            scaled_points = scaler.transform_features(interest_points)
            vol_I = covered_space(scaled_points, radius, digits)

            all_points = v['df'].loc[v['df'].index <= lim][features].values
            scaled_points = scaler.transform_features(all_points)
            vol_T = covered_space(scaled_points, radius, digits)

            mini_test[lim - initial_size] = {'I #': num_interest,
                                             'I %': np.round(100 * num_interest / lim, 1),
                                             'Vol I': f"{vol_I/lim:.2e}",
                                             'Vol T': f"{vol_T/lim:.2e}"
                                             }
        test[v['name']] = pd.DataFrame.from_dict(mini_test, orient='index')
    df = pd.concat([test[v['name']] for v in data.values()], axis=1, keys=[v['name'] for v in data.values()])
    return df.reset_index()


def prepare_data(
        experiments: Dict, names: Dict, features: List[str], targets: List[str],
        additional_values: List[str], treatment: DataTreatment,
        n_slice: int, tot_size: int
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