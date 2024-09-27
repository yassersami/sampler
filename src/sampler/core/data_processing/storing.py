from typing import Dict, List
import os
import pandas as pd

from .sampling_tracker import get_max_interest_index


def parse_results(df: pd.DataFrame, current_history_size: int) -> Dict[str, pd.DataFrame]:
    """ Create an Incremental DataSet named according to the index range. """
    # Calculate the starting and ending indices for the new samples
    n_new_samples = df.shape[0]
    start_idx = current_history_size  # By default dataset size is (last_idx + 1)
    end_idx = current_history_size + n_new_samples - 1

    # Create a dictionary with the index range as the key
    return {f'[{start_idx:03}-{end_idx:03}]': df}


def parse_logs(logs: Dict[str, Dict[str, float]], iteration: int) -> Dict[str, Dict]:
    return {f'log_{iteration:02}': logs}


def join_history(
    history: Dict[str, pd.DataFrame], stop_condition: Dict
) -> pd.DataFrame:
    """ Joins all checkpoints of a run into a single file """
    # Concatenate all batch checkpoints
    df_history = pd.DataFrame()
    for df_batch in history.values():
        df_history = pd.concat([df_history, df_batch], ignore_index=True)

    if stop_condition['stop_on_max_inliers']:
        return df_history
    else:
        # Truncate increased_data to respect stop_condition
        max_interest_index = get_max_interest_index(df_history)
        df_history = df_history.iloc[:max_interest_index]
        return df_history


def join_logs(logs: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Joins all log checkpoints into a single DataFrame.

    Args:
    logs (Dict[str, Dict[str, Dict[str, float]]]): A dictionary where each key
        is a checkpoint and each value is a dictionary of log entries.

    Returns:
    pd.DataFrame: A DataFrame containing all log entries, with columns for each
        log key and an additional 'checkpoint' column.
    """
    # Initialize an empty list to store all log entries
    all_logs = []

    # Iterate through each checkpoint
    for checkpoint, log_entries in logs.items():
        # For each log entry in the checkpoint
        for term_name, log_values in log_entries.items():
            # For each log name and value
            for log_name, value in log_values.items():
                # Create a dictionary for this log entry
                entry = {
                    'term_name': term_name,
                    'log_name': log_name,
                    'checkpoint': checkpoint,
                    'value': value
                }
                all_logs.append(entry)

    # Convert the list of dictionaries to a DataFrame
    df_logs = pd.DataFrame(all_logs)

    # Pivot the DataFrame to have checkpoints as columns
    df_pivoted = df_logs.pivot(index=['term_name', 'log_name'], columns='checkpoint', values='value')

    # Reset index to make term_name and log_name regular columns
    df_final = df_pivoted.reset_index()
    
    # Create a mask for duplicates in the term_name column
    mask = df_final.duplicated('term_name')
    
    # Replace duplicates with an empty string
    df_final.loc[mask, 'term_name'] = ''

    return df_final


def create_history_folder(history_path: str, should_rename: bool = True):
    """
    Create history folder where explored samples are incrementally stored.
    """
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
    """
    Rename folder and avoid duplicate.
    If folder already exists, add _i suffix.
    """
    i = 1
    new_path = old_path + f'_{i}'
    folder_exists = os.path.isdir(new_path)
    while folder_exists:
        i += 1
        new_path = old_path + f'_{i}'
        folder_exists = os.path.isdir(new_path)
    os.rename(old_path, new_path)


def store_df(df: pd.DataFrame, history_path: str, file_name: str) -> None:
    
    # Check if file name ends with tile extension
    if not file_name.endswith('.csv'):
        file_name += '.csv'

    # Save file
    file_path = os.path.abspath(os.path.join(history_path, file_name))
    df.to_csv(file_path, index=False)
