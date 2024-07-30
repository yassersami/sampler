import sys
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import norm
from smt.sampling_methods import LHS

from sampler.common.data_treatment import DataTreatment
from sampler.common.scalers import MixedMinMaxScaler

from sampler.models.wrapper_for_0d import run_simulation

RANDOM_STATE = 42

def generate_LHS_inputs(
        n_input: int, features: List[str], variables_ranges: Dict,
    )-> pd.DataFrame:

    bounds = np.array([ variables_ranges[f]['bounds'] for f in features]) # features boundaries 
    sampling = LHS(xlimits=bounds, random_state=RANDOM_STATE)
    samples = sampling(n_input)

    
    # ? If you want scale data but simulator doesn't accept scaled data
    # scale = [variables_ranges[f]['scale'] for f in features]
    # scaler = MixedMinMaxScaler(features=features, targets=[], scale=scale) 
    # scaler.fit(bounds.T) # fit the scaler to the boundaries
    # samples = scaler.transform_features(samples) # normalize the samples

    df_lhs = pd.DataFrame(samples, columns=features)


    # Handle discretes values 
    for f in variables_ranges:
        discrete_steps = variables_ranges[f]['discete_steps']
        if discrete_steps: # not 0
            df_lhs[f] = df_lhs[f].apply(lambda x: round(x/discrete_steps)*discrete_steps)

    return df_lhs
    

def evaluate_LHS(df_lhs: pd.DataFrame, features: List['str'], targets: List['str'], additional_values: List['str'], n_proc: int) -> pd.DataFrame:
    assert all(f in df_lhs.columns for f in features), f"Error! Features {features} not in df_lhs columns!"
    assert len(df_lhs) > 0, "Error! df_lhs is empty!"

    if "r_ext_pMeO" not in features:
        df_lhs['r_ext_pMeO'] = df_lhs['r_ext_pAl']

    df_result = run_simulation(x=df_lhs, n_proc=n_proc, size=0) 
    df_evaluated = pd.concat([df_lhs, df_result], axis=1)
    
    return df_evaluated

