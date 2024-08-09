from typing import Dict, List
import os
import pandas as pd


def parse_results(results_df: pd.DataFrame, n_new_samples: int) -> Dict[str, pd.DataFrame]:
    """ Create an Incremental DataSet named according to the index range. """
    last_idx = results_df.shape[0]
    start_idx = last_idx - n_new_samples
    return {f'[{start_idx}-{last_idx}]': results_df[start_idx: last_idx]}


def join_history(history: Dict[str, pd.DataFrame], run_condition: Dict, initial_size: int) -> pd.DataFrame:
    """ Joins all checkpoints of a run into a single file """
    df_out = pd.DataFrame()
    for df in history.values():
        df_out = pd.concat([df_out, df], ignore_index=True)

    if run_condition['run_until_max_size']:
        # df_out = df_out.iloc[-(initial_size+run_condition['max_size']):]
        # yasser: you can't truncate, this means that you remove data what you can do is
        # df.dropna(targets) for example to keep only inliers and then you will have
        # size-initial_size = max_size.
        return df_out

    # truncate increased_data to respect run_condition
    interest_count = 0
    index = initial_size
    while interest_count<run_condition['n_interest_max'] and index<len(df_out):
        if df_out.iloc[index]['quality']=='interest':
            interest_count += 1
        index+=1

    df_out = df_out.iloc[:index]
    assert interest_count==run_condition['n_interest_max'], "Not enough 'interest' rows in the dataset."
    return df_out


def set_history_folder(history_path: str, should_rename: bool = True):
    '''
    Set folder where to store explored samples during optimization.
    '''
    folder_exists = os.path.isdir(history_path)
    if not folder_exists:
        os.makedirs(history_path)  # create new folder
    elif folder_exists and should_rename:
        rename_folder(history_path)
        os.makedirs(history_path)  # create new folder


def rename_folder(old_path: str):
    '''
    Rename folder and avoid duplicate.
    If folder already exists, add _i suffix.
    '''
    i = 1
    new_path = old_path + f'_{i}'
    folder_exists = os.path.isdir(new_path)
    while folder_exists:
        i += 1
        new_path = old_path + f'_{i}'
        folder_exists = os.path.isdir(new_path)
    os.rename(old_path, new_path)
