"""Step u = the former step r (recompute the curated metrics) + writing the units
into the NWB. These tests fake the two heavy parts (the spikeinterface recompute
and the waveform extraction) and check only the order and the all-or-nothing rule."""
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pynwb")                     # add_units imports pynwb at the top
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "nwb"))
import add_units  # noqa: E402


def make_op(tmp_path, with_nwb):
    """An op folder with one curated phy folder, and the session NWB if asked."""
    op = tmp_path / "op1"
    phy = op / "Rat1_rec_sorting_output" / "phy_export"
    phy.mkdir(parents=True)
    (phy / "params.py").write_text("sample_rate = 30000.0\n")
    if with_nwb:
        (op / "Rat1_20260101.nwb").write_text("not read: _write_units is faked")
    return op, phy


@pytest.fixture
def events(monkeypatch):
    """Fake the recompute, the unit reading and the NWB writing; record the calls."""
    calls = []
    fake = types.SimpleNamespace(files_written=["curated_quality_metrics.csv",
                                                "curated_template_metrics.csv"])

    def recompute_curated_metrics(phy, n_jobs=4, refresh_auto_labels=True):
        calls.append(("recompute", Path(phy).name))
        fake.refresh_auto_labels = refresh_auto_labels
        return list(fake.files_written)

    def collect_units(phy, n_jobs=4):
        calls.append(("collect", Path(phy).name))
        return [{"phy_cluster_id": 1}], ["snr"], []

    def write_units(nwb_path, all_units, metric_cols):
        calls.append(("write", Path(nwb_path).name))

    # step u imports sorter_common inside recompute_metrics(): a fake module in
    # sys.modules is what that import then finds
    sorter_common = types.ModuleType("sorter_common")
    sorter_common.recompute_curated_metrics = recompute_curated_metrics
    monkeypatch.setitem(sys.modules, "sorter_common", sorter_common)
    monkeypatch.setattr(add_units, "_collect_units", collect_units)
    monkeypatch.setattr(add_units, "_write_units", write_units)
    fake.calls = calls
    return fake


def test_metrics_are_recomputed_before_the_units_are_written(tmp_path, events):
    op, _phy = make_op(tmp_path, with_nwb=True)
    assert add_units.add_units_to_nwb(op) is True
    assert events.calls == [("recompute", "phy_export"), ("collect", "phy_export"),
                            ("write", "Rat1_20260101.nwb")]


def test_step_u_does_not_change_the_labels(tmp_path, events):
    # re-running u must leave every good/mua/noise label as it is: the manual
    # cluster_group.tsv (never touched anyway) and the automated ones
    op, phy = make_op(tmp_path, with_nwb=True)
    add_units.add_units_to_nwb(op)
    assert events.refresh_auto_labels is False


def test_without_an_nwb_only_the_metrics_are_recomputed(tmp_path, events):
    # e.g. a session being prepared for labelling in Phy
    op, _phy = make_op(tmp_path, with_nwb=False)
    assert add_units.add_units_to_nwb(op) is True
    assert events.calls == [("recompute", "phy_export")]


def test_stale_quality_metrics_write_nothing(tmp_path, events):
    # the recompute ran but could not refresh curated_quality_metrics.csv: the
    # old file would put pre-curation units into the NWB, so nothing is written
    op, _phy = make_op(tmp_path, with_nwb=True)
    events.files_written = ["curated_template_metrics.csv"]
    assert add_units.add_units_to_nwb(op) is False
    assert events.calls == [("recompute", "phy_export")]


def test_stale_template_metrics_only_warn(tmp_path, events, capsys):
    op, _phy = make_op(tmp_path, with_nwb=True)
    events.files_written = ["curated_quality_metrics.csv"]
    assert add_units.add_units_to_nwb(op) is True
    assert ("write", "Rat1_20260101.nwb") in events.calls
    assert "template metrics could not be recomputed" in capsys.readouterr().out
