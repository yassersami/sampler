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

    # truncate increased_data to respect run_condition
    if run_condition['run_until_max_size']:
        df_out = df_out.iloc[-(initial_size+run_condition['max_size']):]
    else:
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


def join_history_outside_pipeline():
    """ 
    Auxiliary function in case of pipeline interrumption.
    Joins all csv pieces of results into a single csv file (used by Analysis pipeline)
    or use kedro run parameters like --to-nodes/--from-nodes/--node to explicitly define
    what needs to be run.
    """
    import os
    import re
    import csv
    from kedro.config import ConfigLoader

    conf_loader = ConfigLoader(conf_source="conf")
    conf_catalog = conf_loader.get("catalog.yml")

    history_path = conf_catalog["fom_history"]["path"]
    output_file_path = conf_catalog["fom_increased_data"]["filepath"]

    # Get list of CSV files in the directory
    csv_files = [f for f in os.listdir(history_path) if f.endswith('.csv')]

    # Sort the list of files based on their numeric prefix
    csv_files.sort(key=lambda x: int(re.search(r'\[(\d+)-', x).group(1)))

    # Write contents of all CSV files into the output file
    with open(output_file_path, 'w', newline='') as output_csv:
        writer = csv.writer(output_csv)
        for i_file, csv_file in enumerate(csv_files):
            with open(os.path.join(history_path, csv_file), 'r') as input_csv:
                reader = csv.reader(input_csv)
                for i_row, row in enumerate(reader):
                    if i_file == 0 or i_row > 0:
                        writer.writerow(row)

