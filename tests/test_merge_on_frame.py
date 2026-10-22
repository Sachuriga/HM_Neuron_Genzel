import numpy as np
import pandas as pd

import process_dataframe as pdf


def test_rows_are_matched_by_frame_not_by_position():
    seconds = pd.DataFrame({"Frame Number": np.arange(10),
                            "Seconds From Creation": np.arange(10) / 30.0})
    # tracker skipped frames 3 and 4, and starts at frame 1
    coords = pd.DataFrame({"Frame_Index": [1, 2, 5, 6, 7], "Rat_X": [10, 11, 12, 13, 14]})
    merged = pdf.merge_on_frame(coords, seconds, "Time (seconds)")
    assert np.allclose(merged["Time (seconds)"], coords.Frame_Index / 30.0)
    # the old positional insert pairs row r with frame r: wrong after the skip
    old = pdf.merge_df(coords, seconds, "Time (seconds)")
    assert not np.allclose(old["Time (seconds)"], coords.Frame_Index / 30.0)


def test_missing_frame_gets_nan():
    seconds = pd.DataFrame({"Frame Number": [0, 1, 2], "Seconds From Creation": [0.0, 0.1, 0.2]})
    coords = pd.DataFrame({"Frame_Index": [0, 2, 3]})
    merged = pdf.merge_on_frame(coords, seconds, "Time (seconds)")
    assert np.isnan(merged["Time (seconds)"].iloc[2])
    assert merged["Time (seconds)"].iloc[1] == 0.2
