import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.neighbors import NearestNeighbors

from typing import List, Dict, Tuple
import warnings

from sampler.common.data_treatment import DataTreatment


def get_values_from_nearest_neighbour(x, big_data, features, n_proc):
    """
    Instead of using the physics simulator to compute y = physics.compute(x),
    this function computes the nearest neighbour in feature space to get the closest results
    to physics.compute(x) using a big dataset of previously computed results.
    """

    # Find nearest neighbour point to x in data[features]
    knn = NearestNeighbors(n_neighbors=1, n_jobs=n_proc)
    knn.fit(big_data[features].values)
    dist, index = knn.kneighbors(x)

    if any(dist == 0):
        warnings.warn(f"Warning! Nearest neighbour is the same point as x ! x: \n{x}")

    res = big_data.iloc[index.reshape(-1)].copy()
    res = res.drop_duplicates()

    # Remove contents of res from big_data
    big_data.drop(index.reshape(-1), inplace=True)
    big_data.reset_index(drop=True, inplace=True)

    return res

# ----------------- Unused functions for replacing simulator during test

def get_big_data(treatment: DataTreatment, features: List[str], targets: List[str], additional_values: List[str]):
    '''
    Use as follow with get_values_from_nearest_neighbour:
        big_data = get_big_data(**args)
        # Then in the loop:
            new_df = get_values_from_nearest_neighbour(
                    new_x=new_x, big_data=big_data, features=features, n_proc=batch_size
                )
    '''
    # Load recorded data
    # TODO: Add path to kedro paths insteado of directly here
    path = '/home/jack/sampler/data/01_raw/all_data_p3_new.csv'
    big_data = pd.read_csv(path)[features + targets + additional_values]
    big_data, _ = treatment.treat_data(big_data)
    # Remove rows with 0 in any target column
    big_data = big_data[(big_data[targets] != 0).all(axis=1)]
    big_data["sim_time"] = big_data["sim_time"].fillna(0.0)
    big_data = big_data.reset_index(drop=True)
    return big_data


def get_lhs_points(treatment, features, seed=42):
    '''
    Use to directly sample new points without using the fom as follow:
        next_points = get_lhs_points(treatment, features)
        # Then in the loop : 
            new_x = next_points[size:size+batch_size]
            if len(new_x) == 0:
                warnings.warn("No new points available from LHS! Finishing here for size = {}".format(size))
                break
            constraints_array = np.zeros((batch_size, len(cons_names)))
    '''

    # Number of samples
    num_samples = 300

    # Get bounds from treatment
    bounds = np.array([treatment.variables_ranges[f]["bounds"] for f in features])
    n_dimensions = len(bounds)
    
    # Create the Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)
    
    # Generate Latin Hypercube Samples
    samples_0_1 = sampler.random(n=num_samples)
    
    # Scale the samples to the specified bounds
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    samples_L_U = qmc.scale(samples_0_1, lower_bounds, upper_bounds)

    warnings.warn(f"Using LHS for sampling! Sampling points of dim {samples_L_U.shape}")


    # Scale samples down again using the treatment scaler (some variables may be log-scaled)
    scaled_samples = treatment.scaler.transform_features(samples_L_U)
    
    return scaled_samples
