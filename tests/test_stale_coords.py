"""A Coordinates_Full_with_frames.csv left by an earlier tracker run is deleted,
not read: a session re-tracked into 26 trials was plotted with the old 17."""
import numpy as np
import pandas as pd

import stale_coords


def _tracking(trials):
    """One row per frame; `trials` gives the Trial_Num of each tracked frame."""
    n = len(trials)
    return pd.DataFrame({
        "Frame_Index": np.arange(n),
        "Timestamp": np.arange(n) / 30.0 - 72.09419484380086,
        "Trial_Num": trials,
        "Rat_X": np.where(np.isnan(trials), np.nan, np.linspace(101.123456789, 2000.5, n)),
        "Rat_Y": np.where(np.isnan(trials), np.nan, np.linspace(55.987654321, 900.25, n)),
    })


def _write(folder, full, copy_of):
    """The tracker's CSV, and step d's copy of `copy_of` (two columns more)."""
    full_p = folder / "20260728_Rat6_Coordinates_Full.csv"
    copy_p = folder / "20260728_Rat6_Coordinates_Full_with_frames.csv"
    full.to_csv(full_p, index=False)
    copy_of.to_csv(copy_p, index=False)
    df = pd.read_csv(copy_p)                 # step d reads the CSV and writes it again
    df.assign(region_id=3.0, extracted_frame_idx=df["Frame_Index"]).to_csv(copy_p, index=False)
    return full_p, copy_p


OLD = np.array([np.nan, 1, 1, 1, np.nan, 2, 2, 2, 2, np.nan])     # 2 trials
NEW = np.array([np.nan, 1, 1, 1, np.nan, 2, 2, 3, 3, np.nan])     # re-tracked: 3 trials


def test_copy_of_an_earlier_tracker_run_is_deleted(tmp_path, capsys):
    full_p, copy_p = _write(tmp_path, _tracking(NEW), _tracking(OLD))
    assert stale_coords.drop_stale_with_frames(tmp_path) == [copy_p]
    assert not copy_p.exists() and full_p.exists()
    assert "Deleted 20260728_Rat6_Coordinates_Full_with_frames.csv" in capsys.readouterr().out


def test_copy_of_the_current_run_is_kept(tmp_path):
    cur = _tracking(NEW)
    full_p, copy_p = _write(tmp_path, cur, cur)
    # the clock repair rewrites Timestamp in place; the tracking is unchanged
    df = pd.read_csv(copy_p)
    df["Timestamp"] += 0.25
    df.to_csv(copy_p, index=False)
    assert stale_coords.drop_stale_with_frames(tmp_path) == []
    assert copy_p.exists()


def test_same_trials_but_other_positions_or_length_is_stale(tmp_path):
    moved = _tracking(NEW)
    moved.loc[2, "Rat_X"] += 5.0
    _, copy_p = _write(tmp_path, _tracking(NEW), moved)
    assert stale_coords.drop_stale_with_frames(tmp_path) == [copy_p]

    _, copy_p = _write(tmp_path, _tracking(NEW), _tracking(NEW[:-2]))
    assert stale_coords.drop_stale_with_frames(tmp_path) == [copy_p]


def test_nothing_to_do_without_both_files(tmp_path):
    assert stale_coords.drop_stale_with_frames(tmp_path / "missing") == []
    _tracking(OLD).to_csv(tmp_path / "20260728_Rat6_Coordinates_Full_with_frames.csv", index=False)
    assert stale_coords.drop_stale_with_frames(tmp_path) == []      # no tracker CSV to compare with
    assert (tmp_path / "20260728_Rat6_Coordinates_Full_with_frames.csv").exists()


def test_copy_that_cannot_be_deleted_is_still_reported(tmp_path, monkeypatch, capsys):
    _, copy_p = _write(tmp_path, _tracking(NEW), _tracking(OLD))

    def locked(self, *a, **k):
        raise PermissionError("in use")
    monkeypatch.setattr(type(copy_p), "unlink", locked)
    assert stale_coords.drop_stale_with_frames(tmp_path) == [copy_p]
    assert copy_p.exists()
    assert "could not be deleted" in capsys.readouterr().out


def test_plot_trials_reads_the_tracker_csv_when_the_copy_is_stale(tmp_path):
    import plot_trials
    _, copy_p = _write(tmp_path, _tracking(NEW), _tracking(OLD))
    df, stem = plot_trials.build_df_from_coords(tmp_path, tmp_path)
    assert not copy_p.exists()
    assert stem == "20260728_Rat6_Coordinates_Full"
    assert sorted(df["trial_id"].unique()) == [1, 2, 3]
