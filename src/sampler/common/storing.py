from typing import Dict, List
import os
import pandas as pd


def parse_results(df: pd.DataFrame, current_history_size: int) -> Dict[str, pd.DataFrame]:
    """ Create an Incremental DataSet named according to the index range. """
    # Calculate the starting and ending indices for the new samples
    n_new_samples = df.shape[0]
    start_idx = current_history_size  # By default dataset size is (last_idx + 1)
    end_idx = current_history_size + n_new_samples - 1

    # Create a dictionary with the index range as the key
    return {f'[{start_idx}-{end_idx}]': df}


def join_history(history: Dict[str, pd.DataFrame], run_condition: Dict, initial_size: int) -> pd.DataFrame:
    """ Joins all checkpoints of a run into a single file """
    df_history = pd.DataFrame()
    for df_batch in history.values():
        df_history = pd.concat([df_history, df_batch], ignore_index=True)

    if run_condition['run_until_max_size']:
        # df_history = df_history.iloc[-(initial_size+run_condition['max_size']):]
        # * yasser: you can't truncate, this means that you remove data what you can do is
        # * df.dropna(targets) for example to keep only inliers and then you will have
        # * size-initial_size = max_size.
        return df_history

    # truncate increased_data to respect run_condition
    interest_count = 0
    index = initial_size
    while interest_count<run_condition['n_interest_max'] and index<len(df_history):
        if df_history.iloc[index]['quality']=='interest':
            interest_count += 1
        index+=1

    df_history = df_history.iloc[:index]
    assert interest_count==run_condition['n_interest_max'], "Not enough 'interest' rows in the dataset."
    return df_history


def set_history_folder(history_path: str, should_rename: bool = True):
    '''
    Set history folder where explored samples are incrementally stored.
    '''
    # Check if the folder already exists
    folder_exists = os.path.isdir(history_path)
    
    if not folder_exists:
        # If the folder does not exist, create it
        os.makedirs(history_path)
    elif should_rename:
        # If the folder exists and should_rename is True, rename existing folder
        rename_folder(history_path)
        # Create a new folder after renaming the old one
        os.makedirs(history_path)


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
