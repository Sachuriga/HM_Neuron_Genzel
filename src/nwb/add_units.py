"""
Runner step [u]: add the curated spike-sorting Units table to the session NWB.

Runs AFTER step 4 (which writes the behaviour NWB via create_nwb.py). For each op folder it:
  - finds the session NWB (step 4 / step 8):    <op>/Rat*_*.nwb
  - finds the curated phy folder(s):            <op>/*_sorting_output/phy_export
  - reads, per curated unit:
      * spike times (seconds)      from spike_times.npy + spike_clusters.npy
      * quality metrics            from curated_quality_metrics.csv
      * waveform/template metrics  from curated_template_metrics.csv
      * good/mua/noise label       from cluster_group.tsv (Phy's MANUAL group —
                                   the human good/noise truth) -> quality_label;
                                   quality_check_labels.csv (automated) is also
                                   kept as auto_quality_label for reference
      * waveforms                  ALWAYS extracted here by rebuilding the analyzer
                                   from the phy recording (processed_binary): the mean
                                   template, plus the individual waveforms of up to
                                   MAX_SPIKE_WAVEFORMS spikes per unit (seeded random
                                   sample) on the unit's own tetrode. A stale
                                   curated_templates.npy is never trusted.
  - recomputes, from the fresh templates + spikes, authoritative waveform metrics
    (trough-to-peak, peak/trough half-width), the CellExplorer ACG tau_rise and the
    putative cell_type (see spike_metrics.py) — the NWB template-metric CSVs were
    unreliable.
  - writes an NWB Units table with spike_times, waveform_mean, one column per
    quality/template metric, the recomputed metrics + cell_type, plus quality_label
    (manual cluster_group.tsv), auto_quality_label, phy_cluster_id, sorting_group,
    and the per-spike waveforms:
      waveforms              units['waveforms'][i] -> (n_spikes, 4, n_samples) int16,
                             same units as the phy recording (ADC counts)
      waveforms_spike_index  which of the unit's spike_times each row belongs to
      waveforms_channel_ids  the tetrode channels, in the order of axis 1

Waveforms are ALWAYS extracted and stored. Every run REPLACES an existing Units
table, so re-curating in Phy and re-running u updates the NWB. If the waveforms
cannot be extracted for any phy folder, nothing is written (the old Units table
stays) and the step exits with an error — a Units table without waveform_mean
is never written.

Phy's own templates.npy is NOT used: it is keyed by the pre-curation cluster ids
and goes stale after merges/splits.

Usage:
    python add_units.py --output_folder <op_folder> [--n_jobs 4]
"""

import sys
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

# shared metric functions (also used by step v) — same code computes what we store
# here and what visualize_nwb.py shows, so they can't drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spike_metrics import waveform_metrics, acg_tau_rise, classify_cell_type  # noqa: E402

# NOTE: the spikeinterface-based phy loaders are imported lazily inside
# _extract_waveforms, so importing this module does not need the sorting stack.

MAX_SPIKE_WAVEFORMS = 1000   # individual waveforms stored per unit (random sample)
WAVEFORM_SEED = 0            # fixed, so a re-run stores the same spikes


def _read_phy_params(phy_folder):
    """Parse phy's params.py (`key = value` lines) into a dict. Local copy so the
    fast path doesn't import sorter_common (and thus spikeinterface)."""
    ns = {}
    exec((Path(phy_folder) / "params.py").read_text(), {}, ns)
    return {k: v for k, v in ns.items() if not k.startswith("__")}


def find_phy_folders(output_folder):
    """Find curated phy_export folders under the op folder (same logic as
    recompute_metrics.py so the two steps agree on what to process)."""
    op = Path(output_folder)
    if (op / "params.py").exists():                 # pointed straight at a phy folder
        return [op]
    phys = sorted(op.glob("*_sorting_output/phy_export"))
    if not phys:
        phys = sorted(op.glob("*/phy_export"))
    if not phys and (op / "phy_export").exists():
        phys = [op / "phy_export"]
    return phys


def find_nwb_file(output_folder):
    """The session NWB (step 4 / step 8) lives directly in the op folder as Rat*_*.nwb."""
    op = Path(output_folder)
    cands = [p for p in sorted(op.glob("*.nwb")) if not p.name.endswith(".tmp.nwb")]
    if cands:
        return cands[0]
    cands = [p for p in sorted(op.glob("**/*.nwb")) if not p.name.endswith(".tmp.nwb")]
    return cands[0] if cands else None


def _read_manual_group(phy):
    """Phy's MANUALLY-curated good/mua/noise group from cluster_group.tsv — the
    human source of truth for which units are good vs noise (it is edited by hand
    in Phy). Keyed by curated cluster id."""
    f = phy / "cluster_group.tsv"
    if f.exists():
        df = pd.read_csv(f, sep="\t")
        if {"cluster_id", "group"} <= set(df.columns):
            return {int(r.cluster_id): str(r.group) for r in df.itertuples()}
    return {}


def _read_auto_labels(phy):
    """The AUTOMATED good/mua/noise labels from quality_check_labels.csv. Kept for
    reference; these can differ from the manual cluster_group.tsv when the user
    relabels units in Phy. Keyed by curated cluster id."""
    f = phy / "quality_check_labels.csv"
    if f.exists():
        df = pd.read_csv(f, index_col=0)
        col = "quality_label" if "quality_label" in df.columns else df.columns[-1]
        return {int(k): str(v) for k, v in df[col].items()}
    return {}


def _tetrode_sparsity(sorting, recording, phy, n_jobs):
    """ChannelSparsity giving every unit the channels of its own tetrode (the phy
    channel group of its best channel), or None when the phy folder has no
    channel_groups.npy (the analyzer then falls back to radius sparsity)."""
    from spikeinterface.core import ChannelSparsity, estimate_sparsity
    groups_file = phy / "channel_groups.npy"
    if not groups_file.exists():
        return None
    ch_groups = np.load(groups_file).astype(int).ravel()
    ch_map = (np.load(phy / "channel_map.npy").astype(int).ravel()
              if (phy / "channel_map.npy").exists() else np.arange(len(ch_groups)))
    by_col = np.full(recording.get_num_channels(), -1)
    by_col[ch_map] = ch_groups                      # dat column -> tetrode
    best = estimate_sparsity(sorting, recording, method="best_channels", num_channels=1,
                             n_jobs=n_jobs, chunk_duration="1s", progress_bar=False)
    mask = np.zeros((len(sorting.unit_ids), recording.get_num_channels()), dtype=bool)
    for i, u in enumerate(sorting.unit_ids):
        g = by_col[best.unit_id_to_channel_indices[u][0]]
        if g < 0:
            return None
        mask[i] = by_col == g
    return ChannelSparsity(mask, sorting.unit_ids, recording.channel_ids)


def _extract_waveforms(phy, n_jobs=4):
    """Return {unit_id: {...}} with, per unit:
        template     (n_samples, n_channels) mean waveform, all dat columns (zero
                     off the unit's tetrode) — stored as waveform_mean
        spikes       (n_spikes, n_tetrode_channels, n_samples) individual waveforms
                     of up to MAX_SPIKE_WAVEFORMS random spikes
        spike_index  index of each of those spikes in the unit's sorted spike train
        channel_ids  tetrode channel ids, in the order of spikes' axis 1

    ALWAYS rebuilds the analyzer from the phy recording — a cached
    curated_templates.npy is never trusted (it can be stale from an earlier
    curation); the fresh templates are saved over it. Returns {} on failure (the
    caller then refuses to write the session)."""
    tpl_file = phy / "curated_templates.npy"
    ids_file = phy / "curated_template_unit_ids.npy"
    print("  Extracting waveforms from the phy recording (rebuilding analyzer; "
          "this reads the whole recording)...")
    try:
        import spikeinterface.full as si
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sorter"))
        from sorter_common import (  # noqa: E402
            _load_curated_sorting_from_phy, _load_recording_from_phy)
        try:
            sorting = _load_curated_sorting_from_phy(phy)
        except Exception as e:
            print(f"  Direct load failed ({e}); falling back to si.read_phy.")
            sorting = si.read_phy(phy, load_all_cluster_properties=True)
        recording = _load_recording_from_phy(phy)
        jk = {"n_jobs": n_jobs, "chunk_duration": "1s", "progress_bar": True}

        sparsity = _tetrode_sparsity(sorting, recording, phy, n_jobs)
        if sparsity is None:
            print("  No phy channel_groups.npy — per-spike waveforms use radius "
                  "sparsity instead of the unit's tetrode.")
            analyzer = si.create_sorting_analyzer(sorting, recording, format="memory",
                                                  sparse=True)
        else:
            analyzer = si.create_sorting_analyzer(sorting, recording, format="memory",
                                                  sparsity=sparsity)
        analyzer.compute("random_spikes", method="uniform",
                         max_spikes_per_unit=MAX_SPIKE_WAVEFORMS, seed=WAVEFORM_SEED)
        analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, **jk)
        analyzer.compute("templates", **jk)
        templates = np.asarray(analyzer.get_extension("templates").get_data())
        unit_ids = np.asarray(analyzer.unit_ids)

        # Which spikes the random sample took, in the order the waveforms
        # extension returns them for each unit (spike-vector order).
        wf_ext = analyzer.get_extension("waveforms")
        picked = analyzer.sorting.to_spike_vector()[
            analyzer.get_extension("random_spikes").get_data()]
        chan_ids = np.asarray(recording.channel_ids)
        si_ids_file = phy / "channel_map_si.npy"   # original (pre-phy) channel ids
        si_ids = np.load(si_ids_file).ravel() if si_ids_file.exists() else None

        by_unit = {}
        for i, u in enumerate(unit_ids):
            wf = np.asarray(wf_ext.get_waveforms_one_unit(u, force_dense=False))
            samples = picked["sample_index"][picked["unit_index"] == i]
            train = np.asarray(analyzer.sorting.get_unit_spike_train(u))  # sorted
            cols = analyzer.sparsity.unit_id_to_channel_indices[u] \
                if analyzer.sparsity is not None else np.arange(len(chan_ids))
            # (spikes, samples, channels) -> (spikes, channels, samples): pynwb's
            # waveforms layout. The recording is int16 with unit gain, so the values
            # are whole numbers and int16 is lossless.
            spikes = np.transpose(wf, (0, 2, 1))
            if np.all(spikes == np.round(spikes)):
                spikes = spikes.astype(np.int16)
            by_unit[int(u)] = {
                "template": templates[i],
                "spikes": spikes,
                "spike_index": np.searchsorted(train, samples).astype(np.int64),
                "channel_ids": (si_ids[cols] if si_ids is not None
                                else chan_ids[cols]).astype(np.int64),
            }
        try:
            np.save(tpl_file, templates)
            np.save(ids_file, unit_ids)
            np.save(phy / "curated_template_channel_ids.npy", np.asarray(analyzer.channel_ids))
        except Exception:
            pass
        return by_unit
    except Exception as e:
        print(f"  Could not extract waveforms ({e}).")
        traceback.print_exc()
        return {}


def _spike_times_by_unit(phy, unit_ids):
    """Per-unit spike times in SECONDS, from phy's spike_times/spike_clusters."""
    p = _read_phy_params(phy)
    fs = float(p["sample_rate"])
    samples = np.load(phy / "spike_times.npy").astype("int64").flatten()
    clusters = np.load(phy / "spike_clusters.npy").astype("int64").flatten()
    wanted = set(int(u) for u in unit_ids)
    # Sort by cluster once, then slice each unit's contiguous block — fast even
    # for millions of spikes.
    order = np.argsort(clusters, kind="stable")
    c_sorted = clusters[order]
    s_sorted = samples[order]
    uniq, starts = np.unique(c_sorted, return_index=True)
    starts = np.append(starts, len(c_sorted))
    pos = {int(u): i for i, u in enumerate(uniq)}
    out = {}
    for u in wanted:
        if u in pos:
            i = pos[u]
            seg = s_sorted[starts[i]:starts[i + 1]].astype(np.float64) / fs
            out[u] = np.sort(seg)
        else:
            out[u] = np.array([], dtype=np.float64)
    return out


def _collect_units(phy, n_jobs=4):
    """Gather every curated unit in one phy folder into a list of dicts."""
    phy = Path(phy)
    qm_file = phy / "curated_quality_metrics.csv"
    tm_file = phy / "curated_template_metrics.csv"

    quality = pd.read_csv(qm_file, index_col=0) if qm_file.exists() else pd.DataFrame()
    template = pd.read_csv(tm_file, index_col=0) if tm_file.exists() else pd.DataFrame()
    manual = _read_manual_group(phy)   # cluster_group.tsv — human good/mua/noise truth
    auto = _read_auto_labels(phy)      # quality_check_labels.csv — automated labels
    if not manual:
        print(f"  WARNING: no cluster_group.tsv in {phy}; quality_label will fall "
              f"back to the automated labels.")

    templates_by_unit = _extract_waveforms(phy, n_jobs=n_jobs)
    if not templates_by_unit:
        raise RuntimeError(f"no waveforms could be extracted from {phy}")

    # Canonical unit set: the metric rows (the analyzable curated units), kept
    # only where a template exists so waveform_mean is present for every written
    # unit (NWB optional columns are all-or-none).
    if not quality.empty:
        unit_ids = [int(u) for u in quality.index]
    elif templates_by_unit:
        unit_ids = sorted(templates_by_unit.keys())
    else:
        # last resort: whatever clusters exist in the phy sorting
        clusters = np.load(phy / "spike_clusters.npy").astype("int64").flatten()
        unit_ids = [int(u) for u in np.unique(clusters)]

    missing = [u for u in unit_ids if u not in templates_by_unit]
    if missing:
        print(f"  WARNING: {len(missing)} unit(s) have metrics but no template and are "
              f"left out: {missing}")
    unit_ids = [u for u in unit_ids if u in templates_by_unit]

    spikes = _spike_times_by_unit(phy, unit_ids)

    q_cols = list(quality.columns)
    t_cols = list(template.columns)
    group_name = phy.parent.name  # the *_sorting_output folder

    # sampling rate (for waveform metrics) and session duration (for firing rate)
    try:
        fs = float(_read_phy_params(phy)["sample_rate"])
    except Exception:
        fs = 30000.0
    duration = max((s.max() for s in spikes.values() if s.size), default=0.0) or 1.0

    units = []
    for u in unit_ids:
        st = spikes.get(u, np.array([], dtype=np.float64))
        rec = {
            "phy_cluster_id": int(u),
            "sorting_group": group_name,
            # quality_label = Phy's MANUAL cluster_group.tsv (the good/noise truth),
            # falling back to the automated label only if a unit is missing there.
            "quality_label": manual.get(u, auto.get(u, "unsorted")),
            "auto_quality_label": auto.get(u, "unsorted"),
            "spike_times": st,
            "_metrics": {},
        }
        wf = templates_by_unit[u]
        rec["waveform_mean"] = np.asarray(wf["template"], dtype=np.float64)
        rec["waveforms"] = wf["spikes"]
        rec["waveforms_spike_index"] = wf["spike_index"]
        rec["waveforms_channel_ids"] = wf["channel_ids"]

        # --- recomputed waveform metrics + ACG tau_rise + putative cell type ---
        # (the spikeinterface template-metrics CSVs were unreliable; these are the
        #  authoritative, self-consistent values, computed the same way as step v.)
        wm = waveform_metrics(rec["waveform_mean"], fs)
        t2p = wm.get("peak_to_trough_s", np.nan)
        tau = acg_tau_rise(st)
        fr = float(len(st)) / duration
        rec["firing_rate_hz"] = fr
        rec["trough_to_peak_s"] = t2p
        rec["peak_half_width_s"] = wm.get("peak_half_width_s", np.nan)
        rec["trough_half_width_s"] = wm.get("trough_half_width_s", np.nan)
        rec["acg_tau_rise_ms"] = tau
        rec["cell_type"] = classify_cell_type(fr, t2p, tau)

        for col in q_cols:
            rec["_metrics"][col] = _num(quality.at[u, col]) if u in quality.index else np.nan
        for col in t_cols:
            rec["_metrics"][col] = _num(template.at[u, col]) if u in template.index else np.nan
        units.append(rec)

    return units, q_cols, t_cols


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def add_units_to_nwb(output_folder, n_jobs=4):
    """Write the curated Units table(s) for one op folder into its NWB, replacing
    any existing one. Returns False when the waveforms could not be extracted
    (nothing is written then)."""
    nwb_path = find_nwb_file(output_folder)
    if nwb_path is None:
        print(f"No .nwb file found under '{output_folder}' (run step 4 or 8 first). Skipping.")
        return True
    phys = find_phy_folders(output_folder)
    if not phys:
        print(f"No 'phy_export' folder found under '{output_folder}'. Nothing to add.")
        return True

    print(f"NWB target: {nwb_path}")
    print(f"Found {len(phys)} phy folder(s).")

    all_units = []
    metric_cols = []
    failed = []
    for phy in phys:
        print(f"\n--- Reading units from {phy} ---")
        try:
            units, q_cols, t_cols = _collect_units(phy, n_jobs=n_jobs)
        except Exception as e:
            print(f"  Failed to read units from {phy}: {e}")
            traceback.print_exc()
            failed.append(phy)
            continue
        print(f"  {len(units)} unit(s) with waveforms.")
        all_units.extend(units)
        for c in q_cols + t_cols:
            if c not in metric_cols:
                metric_cols.append(c)

    if failed:
        # Writing only the folders that worked would silently drop units, so the
        # whole session is left as it is until every folder extracts.
        print("\n" + "*" * 70)
        print(f"ERROR: waveforms could not be extracted for {len(failed)} phy folder(s):")
        for phy in failed:
            print(f"  {phy}")
        print("Nothing was written; the NWB keeps its previous Units table (if any).")
        print("Check that the phy recording (params.py dat_path -> processed_binary)")
        print("exists and spikeinterface is installed, then re-run step u.")
        print("*" * 70)
        return False

    if not all_units:
        print("No units collected; nothing to write.")
        return True

    _write_units(nwb_path, all_units, metric_cols)
    return True


def _drop_units(nwb_path):
    """Remove an existing Units table so it can be rewritten (pynwb cannot
    replace one in place). Nothing else in the NWB references it. HDF5 does not
    give the freed space back (h5repack does). Returns True if one was removed."""
    import h5py
    with h5py.File(str(nwb_path), "r+") as f:
        if "units" in f:
            del f["units"]
            return True
    return False


def _write_units(nwb_path, all_units, metric_cols):
    """Write the Units table into an existing NWB file (in-place, mode='r+'),
    replacing the one a previous run wrote."""
    if _drop_units(nwb_path):
        print("Replacing the existing Units table.")
    io = NWBHDF5IO(str(nwb_path), mode="r+")
    try:
        nwbfile = io.read()

        nwbfile.add_unit_column(name="phy_cluster_id",
                                description="Original phy cluster id of this unit.")
        nwbfile.add_unit_column(name="sorting_group",
                                description="Source *_sorting_output folder.")
        nwbfile.add_unit_column(name="quality_label",
                                description="Manual good/mua/noise group from Phy's "
                                            "cluster_group.tsv (the human curation truth).")
        nwbfile.add_unit_column(name="auto_quality_label",
                                description="Automated good/mua/noise quality-check label "
                                            "from quality_check_labels.csv (for reference).")
        # recomputed, authoritative unit metrics + putative cell type (step u)
        extra_cols = {
            "firing_rate_hz": "Mean firing rate (n_spikes / session duration), Hz.",
            "trough_to_peak_s": "Trough-to-peak duration on the peak channel, seconds "
                                "(recomputed from the mean waveform).",
            "peak_half_width_s": "Peak half-width, seconds (recomputed from the mean waveform).",
            "trough_half_width_s": "Trough half-width, seconds (recomputed from the mean waveform).",
            "acg_tau_rise_ms": "ACG rise time constant, ms (CellExplorer triple-exponential "
                               "fit to the 0-50ms autocorrelogram).",
            "cell_type": "Putative cell type (CellExplorer criteria + FR gate): interneuron "
                         "if FR>10Hz, or trough-to-peak<=0.425ms, or (>0.425ms & tau_rise>6ms); "
                         "else pyramidal.",
        }
        for col in metric_cols:
            nwbfile.add_unit_column(name=col, description=f"Curated metric '{col}'.")
        for col, desc in extra_cols.items():
            nwbfile.add_unit_column(name=col, description=desc)
        nwbfile.add_unit_column(
            name="waveforms_spike_index", index=True,
            description="For each row of this unit's `waveforms`, the index of that spike "
                        "in the unit's spike_times (a seeded random sample of up to "
                        f"{MAX_SPIKE_WAVEFORMS} spikes per unit).")
        nwbfile.add_unit_column(
            name="waveforms_channel_ids", index=True,
            description="Channel ids (original sorting ids) of the unit's tetrode, in the "
                        "order of the channel axis of `waveforms`.")

        for rec in all_units:
            kwargs = dict(
                spike_times=rec["spike_times"],
                phy_cluster_id=rec["phy_cluster_id"],
                sorting_group=rec["sorting_group"],
                quality_label=rec["quality_label"],
                auto_quality_label=rec["auto_quality_label"],
            )
            for col in metric_cols:
                kwargs[col] = rec["_metrics"].get(col, np.nan)
            for col in extra_cols:
                kwargs[col] = rec[col]
            kwargs["waveform_mean"] = rec["waveform_mean"]
            kwargs["waveforms"] = rec["waveforms"]
            kwargs["waveforms_spike_index"] = rec["waveforms_spike_index"]
            kwargs["waveforms_channel_ids"] = rec["waveforms_channel_ids"]
            nwbfile.add_unit(**kwargs)

        # Individual waveforms are the bulk of the table: gzip them. Per unit they
        # read back as units['waveforms'][i] -> (n_spikes, n_channels, n_samples).
        from hdmf.backends.hdf5 import H5DataIO
        nwbfile.units["waveforms"].target.target.set_data_io(
            H5DataIO, dict(compression="gzip", compression_opts=4, chunks=True))

        io.write(nwbfile)
        print(f"\n[units] Wrote {len(all_units)} unit(s) to {nwb_path} "
              f"({len(metric_cols)} metric column(s), waveform_mean + "
              f"{sum(len(r['waveforms']) for r in all_units)} spike waveforms).")
    finally:
        io.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add curated Units (metrics + waveform templates) to the session NWB.")
    parser.add_argument("--output_folder", required=True,
                        help="op folder containing the NWB and *_sorting_output/phy_export.")
    parser.add_argument("--config", required=False, default=None,
                        help="Accepted for runner consistency (unused).")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="Parallel jobs for the waveform extraction.")
    args = parser.parse_args()

    try:
        ok = add_units_to_nwb(args.output_folder, n_jobs=args.n_jobs)
    except Exception as e:
        print(f"[units] Failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    if not ok:
        sys.exit(1)
