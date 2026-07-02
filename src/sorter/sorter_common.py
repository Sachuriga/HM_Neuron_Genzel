"""
Shared post-sorting helpers for sorting.py and continue_sorting.py.

Keep the code that BOTH the full sorting pipeline and the continue-after-sorting
script use in one place, so a change to the post-sorting analysis (metrics,
quality-check labels, Phy export) only has to be made here.

Contents:
- QUALITY_CHECK_THRESHOLDS   : the tetrode quality-check cutoffs.
- label_units_quality_check  : good/mua/noise labeling from the quality metrics.
- analyze_and_export         : analyzer -> metrics -> labels -> Phy -> cleanup.
"""

import shutil
from pathlib import Path

import numpy as np
import spikeinterface.full as si


# ──────────────────────────────────────────────────────────────────────────────
# AUTOMATED QUALITY CHECK — classic tetrode isolation trio
# ──────────────────────────────────────────────────────────────────────────────
# Two geometry-free stages label each unit good / mua / noise (no waveform-shape
# criteria, which were designed for high-density probes). The "noise" gate is
# applied first, then the "mua" isolation gate, so the outcome is:
#   fails NOISE gate -> "noise"; else fails the isolation trio -> "mua"; else "good".
#
# NOISE (not a real unit):
#   snr        > 2.5 : below this there is no real spike amplitude above noise.
#   num_spikes > 100 : too few spikes to be a unit.
#
# MUA (real spikes but poorly isolated) — the classic tetrode isolation trio:
#   rp_contamination   < 0.01 : clear refractory gap — <1% estimated refractory
#                               contamination (the 'rp_violation' metric OUTPUTS
#                               the 'rp_contamination' column).
#   l_ratio            < 0.1  : L-ratio (Schmitzer-Torbert et al., 2005) — how
#                               much other spikes intrude near the cluster; lower
#                               is better. Common cutoffs ~0.05 (strict)–0.2.
#   isolation_distance > 15   : Isolation Distance (Harris et al., 2001) —
#                               Mahalanobis distance out to which the cluster
#                               stays purer than its surroundings; higher is
#                               better, >~15–20 a common floor. (NaN when a
#                               cluster holds >half the region's spikes -> mua.)
#
# l_ratio + isolation_distance are the OUTPUT columns of the 'mahalanobis'
# quality metric (needs the principal_components extension).
# NOTE: snr / num_spikes (noise) and the three isolation cutoffs (mua) are the
# per-rig knobs to tune. A gate is "greater"/"less" = the value it must satisfy
# to PASS; failing it flags the unit into that category.
QUALITY_CHECK_THRESHOLDS = {
    "noise": {
        "snr": {"greater": 2.5, "less": None},
        "num_spikes": {"greater": 100, "less": None},
    },
    "mua": {
        "rp_contamination": {"greater": None, "less": 0.01},
        "l_ratio": {"greater": None, "less": 0.1},
        "isolation_distance": {"greater": 15, "less": None},
    },
}


def label_units_quality_check(analyzer):
    """Label units good / mua / noise via the classic tetrode quality metrics
    (see QUALITY_CHECK_THRESHOLDS). Uses spikeinterface's bombcell_label_units
    purely as the thresholding engine.

    Best-effort: returns a DataFrame (unit-id index, 'quality_label' column) or
    None if unavailable or anything fails. It NEVER raises into the sorting
    pipeline, and it only thresholds metrics that were actually computed (a
    missing metric column would otherwise KeyError inside the engine).

    Requires the quality_metrics extension (snr, num_spikes, rp_contamination,
    l_ratio, isolation_distance) to be computed first.
    """
    try:
        from spikeinterface.curation import bombcell_label_units
    except Exception as e:
        print(f"[QC] Thresholding engine unavailable in this spikeinterface "
              f"version ({e}); skipping quality-check labeling.")
        return None

    # Collect the metric columns that actually exist, so we can drop thresholds
    # for anything that wasn't computed.
    available = set()
    try:
        ext = analyzer.get_extension("quality_metrics")
        if ext is not None:
            available.update(ext.get_data().columns)
    except Exception:
        pass

    thresholds = {}
    dropped = []
    for cat, rules in QUALITY_CHECK_THRESHOLDS.items():
        kept = {m: r for m, r in rules.items() if m in available}
        dropped += [f"{cat}/{m}" for m in rules if m not in available]
        if kept:
            thresholds[cat] = kept
    if dropped:
        print(f"[QC] Metrics not computed, thresholds skipped: {dropped}")
    if not thresholds:
        print("[QC] No usable metrics available; skipping quality-check labeling.")
        return None

    try:
        labels_df = bombcell_label_units(
            sorting_analyzer=analyzer,
            thresholds=thresholds,
            label_non_somatic=False,  # isolation-trio only; no waveform/non-somatic step
        )
        # Re-label the engine's output column to our quality-check naming.
        labels_df = labels_df.rename(columns={"bombcell_label": "quality_label"})
        counts = labels_df["quality_label"].value_counts().to_dict()
        print(f"[QC] Unit labels ({sum(counts.values())} units): {counts}")
        return labels_df
    except Exception as e:
        print(f"[QC] Quality-check labeling failed ({e}); skipping.")
        return None


def analyze_and_export(sorting, recording, output_dir, n_jobs=4, file_stem="", cleanup=True):
    """Everything AFTER spike sorting: build the SortingAnalyzer, compute
    metrics + quality-check labels, export to Phy, and (optionally) clean up
    intermediates. Shared by the main pipeline (process_single_file) and the
    standalone continue-after-sorting script so both stay in sync.

    Set cleanup=False to keep the intermediate folders (e.g. when re-running
    only the post-sorting steps and you want the inputs preserved).
    """
    output_dir = Path(output_dir)

    # 7. SORTING ANALYZER
    print("Initializing SortingAnalyzer (New API)...")
    analyzer_folder = output_dir / "sorting_analyzer"
    if analyzer_folder.exists():
        shutil.rmtree(analyzer_folder)

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="binary_folder",
        folder=analyzer_folder,
        overwrite=True,
        sparse=True
    )

    # 8. COMPUTE METRICS
    print("Computing analyzer features...")
    job_kwargs = {'n_jobs': n_jobs, 'chunk_duration': '1s', 'progress_bar': True}

    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500, **job_kwargs)
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, **job_kwargs)
    analyzer.compute("templates", **job_kwargs)
    analyzer.compute("noise_levels", **job_kwargs)
    analyzer.compute("principal_components", n_components=3, mode='by_channel_local', **job_kwargs)

    # Quality metrics. The base three are useful in Phy; the labeling-critical
    # ones feed the isolation-trio curation. Metric NAME -> OUTPUT columns:
    #   'rp_violation' -> 'rp_contamination'
    #   'mahalanobis'  -> 'l_ratio', 'isolation_distance'  (needs PCA extension)
    # snr + num_spikes gate the NOISE category, so keep them in the always-computed
    # base set (they have no heavy dependencies).
    qm_base = ['snr', 'num_spikes', 'isi_violation', 'firing_rate']
    qm_label = ['rp_violation', 'mahalanobis']  # the three isolation-trio columns
    qm_amp = ['presence_ratio', 'amplitude_cutoff', 'amplitude_median']
    try:
        analyzer.compute("spike_amplitudes", **job_kwargs)  # needed for amplitude_* metrics
    except Exception as e:
        print(f"[QC] spike_amplitudes failed ({e}); dropping amplitude metrics.")
        qm_amp = [m for m in qm_amp if not m.startswith('amplitude')]
    # Best-effort, degrading gracefully: full set -> base+labeling -> base only,
    # so an unknown/failing metric name never loses the labeling-critical ones.
    for attempt in (qm_base + qm_label + qm_amp, qm_base + qm_label, qm_base):
        try:
            analyzer.compute("quality_metrics", metric_names=attempt, **job_kwargs)
            break
        except Exception as e:
            print(f"[QC] quality_metrics {attempt} failed ({e}); trying a smaller set.")

    # 8b. AUTOMATED QUALITY CHECK — classic tetrode isolation trio (best-effort)
    qc_labels = label_units_quality_check(analyzer)

    # 9. PHY EXPORT
    phy_output_folder = output_dir / "phy_export"
    if phy_output_folder.exists():
        shutil.rmtree(phy_output_folder)

    print(f"Exporting results to Phy: {phy_output_folder}")
    si.export_to_phy(
        sorting_analyzer=analyzer,
        output_folder=phy_output_folder,
        compute_pc_features=True,
        compute_amplitudes=True,
        remove_if_exists=True,
        copy_binary=True, # CHANGED TO TRUE: Necessary so we can delete the intermediate binary folder
        **job_kwargs
    )

    # 9b. Persist quality-check labels INTO the phy export so they survive the
    #     cleanup below. We OVERWRITE Phy's native 'group' (cluster_group.tsv)
    #     with good/mua/noise so units pre-sort into groups when Phy opens.
    if qc_labels is not None:
        try:
            import pandas as pd
            qc_labels.to_csv(phy_output_folder / "quality_check_labels.csv")
            # Phy reads cluster_group.tsv keyed by 0-based cluster_id, in the
            # same unit order export_to_phy used (analyzer.sorting.unit_ids).
            unit_ids = list(analyzer.sorting.unit_ids)
            labels_by_unit = qc_labels["quality_label"].to_dict()
            tsv = pd.DataFrame({
                "cluster_id": range(len(unit_ids)),
                "group": [labels_by_unit.get(u, "unsorted") for u in unit_ids],
            })
            tsv.to_csv(phy_output_folder / "cluster_group.tsv", sep="\t", index=False)
            print(f"[QC] Wrote quality_check_labels.csv and set Phy's group "
                  f"(cluster_group.tsv) to good/mua/noise in {phy_output_folder}.")
        except Exception as e:
            print(f"[QC] Could not write label files ({e}).")

    # 10. CLEANUP INTERMEDIATE FILES
    if cleanup:
        print("\nCleaning up intermediate files...")
        for item in output_dir.iterdir():
            if item.name != "phy_export":
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    print(f"Could not delete {item.name}: {e}")
        print(f"Cleanup complete! Only phy_export remains.")

    print(f"Done processing {file_stem}!")
    print(f"To open Phy, run:\nphy template-gui {phy_output_folder}/params.py\n")


# ──────────────────────────────────────────────────────────────────────────────
# RECOMPUTE METRICS ON A MANUALLY-CURATED PHY FOLDER  (runner step 'r')
# ──────────────────────────────────────────────────────────────────────────────
# After you curate in Phy (merge / split / label good-mua-noise), the merges and
# splits only live in the phy_export folder. This recomputes ALL quality metrics
# and waveform (template) metrics on that CURATED sorting and writes them back
# into the phy folder — WITHOUT re-exporting (so your manual curation is kept).

def _read_phy_params(phy_folder):
    """Parse phy's params.py (simple `key = value` assignments) into a dict."""
    ns = {}
    exec((Path(phy_folder) / "params.py").read_text(), {}, ns)
    return {k: v for k, v in ns.items() if not k.startswith("__")}


def _load_recording_from_phy(phy_folder):
    """Rebuild the (preprocessed) recording from phy's recording.dat + params.py,
    attaching the probe geometry from channel_positions.npy."""
    phy = Path(phy_folder)
    p = _read_phy_params(phy)

    dat = Path(str(p["dat_path"]))
    if not dat.is_absolute() or not dat.exists():
        dat = phy / dat.name  # params.dat_path is usually just 'recording.dat'

    rec = si.read_binary(
        file_paths=[str(dat)],
        sampling_frequency=float(p["sample_rate"]),
        dtype=p["dtype"],
        num_channels=int(p["n_channels_dat"]),
        file_offset=int(p.get("offset", 0)),
    )

    pos_file = phy / "channel_positions.npy"
    if pos_file.exists():
        import probeinterface as pi
        positions = np.load(pos_file)
        probe = pi.Probe(ndim=positions.shape[1])
        probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
        probe.set_device_channel_indices(np.arange(positions.shape[0]))
        rec = rec.set_probe(probe)
    return rec


def _quality_metric_names():
    """Return (misc_names, pca_names) valid for the installed spikeinterface
    version, so 'recompute everything' auto-adapts (PCA request names differ:
    isolation_distance/l_ratio on 0.103, mahalanobis on 0.104)."""
    def _call(fn_name):
        fn = getattr(si, fn_name, None)
        if fn is None:
            try:
                import importlib
                fn = getattr(importlib.import_module("spikeinterface.qualitymetrics"),
                             fn_name, None)
            except Exception:
                fn = None
        try:
            return list(fn()) if fn is not None else []
        except Exception:
            return []

    misc = _call("get_quality_metric_list") or \
        ['num_spikes', 'firing_rate', 'presence_ratio', 'snr', 'isi_violation', 'rp_violation']
    pca = _call("get_quality_pca_metric_list")
    return misc, pca


def _write_df_as_phy_tsv(df, phy_folder):
    """Write each metric column as a phy cluster_<metric>.tsv (cluster_id keyed by
    the curated unit ids), so the recomputed metrics show up as columns in Phy."""
    phy = Path(phy_folder)
    for col in df.columns:
        out = df[[col]].copy()
        out.index.name = "cluster_id"
        out.reset_index().to_csv(phy / f"cluster_{col}.tsv", sep="\t", index=False)


def recompute_curated_metrics(phy_folder, n_jobs=4):
    """Recompute all quality + template (waveform) metrics on the curated phy
    sorting and write them back into the phy folder. Best-effort; never raises.

    Writes: curated_quality_metrics.csv, curated_template_metrics.csv, one
    cluster_<metric>.tsv per metric, and OVERWRITES Phy's cluster_group.tsv with
    the refreshed good/mua/noise quality-check label.
    """
    phy = Path(phy_folder)
    if not (phy / "params.py").exists():
        print(f"[recompute] Not a phy folder (no params.py): {phy}; skipping.")
        return

    print(f"[recompute] Loading curated sorting from {phy} ...")
    sorting = si.read_phy(phy, load_all_cluster_properties=True)
    recording = _load_recording_from_phy(phy)
    print(f"[recompute] {sorting.get_num_units()} curated unit(s), "
          f"{recording.get_num_channels()} channel(s).")

    analyzer = si.create_sorting_analyzer(sorting, recording, format="memory", sparse=True)

    job_kwargs = {"n_jobs": n_jobs, "chunk_duration": "1s", "progress_bar": True}
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500, **job_kwargs)
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, **job_kwargs)
    analyzer.compute("templates", **job_kwargs)
    analyzer.compute("noise_levels", **job_kwargs)
    # PCA is only needed for the isolation metrics (l_ratio / isolation_distance).
    try:
        analyzer.compute("principal_components", n_components=3,
                         mode="by_channel_local", **job_kwargs)
    except Exception as e:
        print(f"[recompute] principal_components failed ({e}); isolation metrics skipped.")

    # ONLY the metrics the quality check uses: snr + num_spikes (noise gate),
    # rp_violation -> rp_contamination, and the isolation metrics l_ratio /
    # isolation_distance. The PCA request name is version-dependent:
    # 'mahalanobis' on 0.104, 'isolation_distance'/'l_ratio' on 0.103.
    misc, pca = _quality_metric_names()
    qc_names = [m for m in ('snr', 'num_spikes', 'rp_violation') if m in set(misc)]
    if 'mahalanobis' in pca:
        pca_names = ['mahalanobis']
    else:
        pca_names = [m for m in ('isolation_distance', 'l_ratio') if m in pca]
    for names in (qc_names + pca_names, qc_names):
        if not names:
            continue
        try:
            analyzer.compute("quality_metrics", metric_names=names, **job_kwargs)
            print(f"[recompute] quality_metrics computed: {names}")
            break
        except Exception as e:
            print(f"[recompute] quality_metrics {names} failed ({e}); trying without PCA.")

    # Waveform (template) metrics — single-channel (tetrode-appropriate).
    try:
        analyzer.compute("template_metrics", include_multi_channel_metrics=False, **job_kwargs)
        print("[recompute] template_metrics computed.")
    except Exception as e:
        print(f"[recompute] template_metrics failed ({e}).")

    # Persist results into the phy folder (no re-export -> curation preserved).
    wrote = []
    for ext_name, csv_name in (("quality_metrics", "curated_quality_metrics.csv"),
                               ("template_metrics", "curated_template_metrics.csv")):
        try:
            ext = analyzer.get_extension(ext_name)
            if ext is None:
                continue
            df = ext.get_data()
            df.to_csv(phy / csv_name)
            _write_df_as_phy_tsv(df, phy)
            wrote.append(csv_name)
        except Exception as e:
            print(f"[recompute] Could not save {ext_name} ({e}).")

    # Refresh the good/mua/noise quality-check label on the curated units and
    # OVERWRITE Phy's native 'group' (cluster_group.tsv) with it.
    try:
        qc = label_units_quality_check(analyzer)
        if qc is not None:
            qc.to_csv(phy / "quality_check_labels.csv")
            tsv = qc.rename(columns={"quality_label": "group"})[["group"]].copy()
            tsv.index.name = "cluster_id"
            tsv.reset_index().to_csv(phy / "cluster_group.tsv", sep="\t", index=False)
            wrote.append("cluster_group.tsv")
    except Exception as e:
        print(f"[recompute] Quality-check relabel failed ({e}).")

    print(f"[recompute] Done. Wrote to {phy}: {wrote if wrote else 'nothing'}")
