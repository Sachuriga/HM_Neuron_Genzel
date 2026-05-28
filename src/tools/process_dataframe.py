import numpy as np
import pandas as pd
from pathlib import Path
from pathnames import find_paths, find_session_folders
from process_log import process_log

# Column with index 'col_index' (default = 1) of dataframe 'df_2' is inserted in dataframe 'df_1'
def merge_df(df_1, df_2, col_label = None, col_index = 1, insert_index = 1):
    df = df_1.copy()
    df.insert(insert_index, col_label, df_2.iloc[:, col_index])
    return df

# Trim to active trials
# Trim dataframe to filter all rows for which column 'label' contains NaN
def filter_nan_in_col(df, label):
    df = df[df[label].notna()]
    return df

# Convert all elements from each column to a list in one cell, converting all rows to a single row
def nest_df(df):
    return pd.DataFrame({col: [df[col].tolist()] for col in df.columns})

# Split such that every trial is a row
# df is a dataframe with rows for all frames of active trials
# trial_col has to be the name of the column containing the trial numbers
def split_trials(df, trial_col='Trial_Num'):
    grouped = df.groupby(trial_col)
    df = pd.DataFrame({
        col: grouped[col].apply(lambda x: np.array(x))
        #col: grouped[col].apply(list) # if you want to use list instead of np array
        for col in df.columns if col != trial_col
    })
    df.set_index(df.index.astype(int), inplace=True)
    return df

def extract_col_by_name(df, session_nr, trial_nr, col_name):
    return df.loc[(session_nr, trial_nr)][col_name]

def process_df(df_coordinates, df_framewise_seconds):
    # Merge stitched framewise into coordinates
    df_merged = merge_df(df_coordinates, df_framewise_seconds, 'Time (seconds)', 1)

    # Trim merged dataframe to only contain frames with active trials
    df_trial = filter_nan_in_col(df_merged, 'Trial_Num')

    df_split_trials = split_trials(df_trial, 'Trial_Num')

    return df_split_trials


if __name__ == "__main__":
    from pathnames import find_paths, find_session_folders
    root = r"C:\Users\Jacob\OneDrive\Documenten\University\Masters Internship\Scripts\Data_Structure/"
    input_folder= r"nwb_data/"
    session_folders = find_session_folders(Path(root + input_folder))
    session_paths = []
    for session_dir in session_folders:
        paths = find_paths(session_dir)
        session_paths.append(paths)
        print(f"Session {session_dir.name}:")
        print(paths, '\n')

    def safe_read(key, paths, ext):
        ext = ext.lower()
        
        if ext == "log":
            function = process_log
        #elif ext == "txt"
        #    function = None
        elif ext == "csv":
            function = pd.read_csv
        else:
            raise Exception("Invalid .ext!")
        try:
            result = function(paths.get(key.lower()))
            print(f"Loaded {key} [Shape: {result.shape}]")
            return result
        except:
            print(f"{key}.{ext} not loaded.")
            return None

    paths = session_paths[0]
    df_coordinates = safe_read('Coordinates_Full', paths.csv_paths, 'csv')
    df_coordinates_with_frames = safe_read('Coordinates_Full_with_frames', paths.csv_paths, 'csv')
    df_framewise_ts = safe_read('framewise_ts', paths.csv_paths, 'csv')
    df_framewise_seconds = safe_read('stitched_framewise_seconds', paths.csv_paths, 'csv')
    df_log = safe_read(next(iter(paths.log_paths)), paths.log_paths, 'log')
    # Merge stitched framewise into coordinates
    df_merged = merge_df(df_coordinates.copy(), df_framewise_seconds, 'Time (seconds)', 1)

    # Trim merged dataframe to only contain frames with active trials
    df_trial = filter_nan_in_col(df_merged, 'Trial_Num')

    # Nest trials to form array of arrays per trial
    df_nested = nest_df(df_trial)

    df_split_trials = split_trials(df_trial, 'Trial_Num')