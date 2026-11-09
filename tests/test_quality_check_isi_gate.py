"""Automated quality check: the ISI gate passes a unit when EITHER the Hill
isi_violations_ratio or the firing-rate-free isi_violations_fraction is under
its cutoff, so a sparse CA1 pyramidal cell with a few violations is not
labelled mua."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

si = pytest.importorskip("spikeinterface.full")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sorter"))
import sorter_common  # noqa: E402

try:
    from spikeinterface.curation import bombcell_label_units  # noqa: F401
    HAVE_BOMBCELL = True
except ImportError:      # spikeinterface 0.103: no labeling at all
    HAVE_BOMBCELL = False
needs_bombcell = pytest.mark.skipif(
    not HAVE_BOMBCELL, reason="this spikeinterface has no bombcell_label_units")


class _Analyzer:
    """Just enough of a SortingAnalyzer for label_units_quality_check."""

    def __init__(self, qm):
        self._qm = qm

    def get_extension(self, name):
        if name != "quality_metrics":
            return None
        qm = self._qm

        class _Ext:
            def get_data(self):
                return qm
        return _Ext()

    def get_metrics_extension_data(self):
        return self._qm


def _qm(**units):
    """Quality metrics, one keyword per unit: name=(ratio, count, num_spikes)."""
    rows = {
        name: {"snr": 5.0, "num_spikes": n, "isolation_distance": 20.0,
               "isi_violations_ratio": ratio, "isi_violations_count": count}
        for name, (ratio, count, n) in units.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def test_fraction_column():
    qm = sorter_common.add_isi_violations_fraction(_qm(a=(2.0, 4, 1668), b=(np.nan, 0, 1)))
    cols = list(qm.columns)
    assert cols[cols.index("isi_violations_count") + 1] == "isi_violations_fraction"
    assert np.isclose(qm.loc["a", "isi_violations_fraction"], 4 / 1667)
    assert np.isnan(qm.loc["b", "isi_violations_fraction"])      # 1 spike: no ISI


def test_fraction_needs_both_columns():
    qm = _qm(a=(2.0, 4, 1668)).drop(columns="isi_violations_count")
    assert sorter_common.add_isi_violations_fraction(qm) is qm


@needs_bombcell
def test_isi_gate_either_metric_passes():
    qm = _qm(
        sparse_pyr=(2.0, 4, 1668),      # ratio 2 but 0.24% of ISIs -> good
        fast_clean=(0.1, 300, 30000),   # ratio under 0.2 although 1% -> good
        contaminated=(2.0, 100, 1668),  # 6% of ISIs, ratio 2 -> mua
        at_cutoff=(0.2, 50, 2001),      # both exactly at the cutoff -> good
    )
    qm.loc["noisy"] = qm.loc["contaminated"]
    qm.loc["noisy", "snr"] = 1.0        # fails the noise gate first -> stays noise

    labels = sorter_common.label_units_quality_check(_Analyzer(qm))["quality_label"]

    assert labels.to_dict() == {
        "sparse_pyr": "good", "fast_clean": "good", "contaminated": "mua",
        "at_cutoff": "good", "noisy": "noise",
    }


@needs_bombcell
def test_isi_gate_and_isolation_gate_both_apply():
    qm = _qm(sparse_pyr=(2.0, 4, 1668))
    qm.loc["sparse_pyr", "isolation_distance"] = 5.0     # fails isolation -> mua
    labels = sorter_common.label_units_quality_check(_Analyzer(qm))
    assert labels.loc["sparse_pyr", "quality_label"] == "mua"


@needs_bombcell
def test_isi_gate_nan_does_not_fail():
    # As in bombcell's MUA gate, a metric that could not be computed (NaN) does
    # not fail the unit; one NaN value falls back to the other metric.
    qm = _qm(both_nan=(np.nan, 0, 1668), ratio_nan=(np.nan, 100, 1668))
    qm.loc["both_nan", "isi_violations_count"] = np.nan
    labels = sorter_common.label_units_quality_check(_Analyzer(qm))["quality_label"]
    assert labels.to_dict() == {"both_nan": "good", "ratio_nan": "mua"}


def test_isi_violation_is_requested():
    assert "isi_violation" in sorter_common._quality_check_request_names()


def test_fraction_from_spikeinterface_metrics():
    # SI's own isi_violations_count and num_spikes give the expected fraction.
    fs = 30000.0
    recording = si.generate_recording(num_channels=4, sampling_frequency=fs,
                                      durations=[60.0], seed=0)
    train = np.arange(1000, 1_700_000, 3000)                 # 10 Hz, 567 spikes
    train = np.sort(np.r_[train, train[:5] + 30])            # + 5 ISIs of 1 ms
    sorting = si.NumpySorting.from_samples_and_labels(
        [train], [np.zeros(len(train), dtype=int)], fs)
    analyzer = si.create_sorting_analyzer(sorting, recording, format="memory",
                                          sparse=False)
    analyzer.compute("quality_metrics", metric_names=["num_spikes", "isi_violation"])

    qm = sorter_common.add_isi_violations_fraction(
        analyzer.get_extension("quality_metrics").get_data())
    assert qm["isi_violations_count"].iloc[0] == 5
    assert np.isclose(qm["isi_violations_fraction"].iloc[0], 5 / (len(train) - 1))
