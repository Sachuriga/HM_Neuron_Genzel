"""
Continue the pipeline AFTER spike sorting, without re-sorting.

For each already-sorted recording it rebuilds the SortingAnalyzer from the saved
sorting + preprocessed recording, computes quality/template metrics, runs the
tetrode-tuned BombCell labeling, and exports to Phy — i.e. the exact same
post-sorting steps as sorting.py (they share analyze_and_export()).

It looks inside the OUTPUT (op) folder for the per-recording subfolders that
sorting.py creates:  <stem>_<sorter>_sorting_output/ , each containing
'final_sorting_result/' (the saved sorting) and 'processed_binary/' (the saved
preprocessed recording). If only a partial 'sorting_analyzer/' survived, that is
used as a fallback.

Usage:
    python continue_sorting.py --output_folder <op_folder> [--n_jobs 4] [--keep-intermediates]

Notes:
- This needs the intermediate folders that exist when a sort crashed AFTER
  saving the sorting but BEFORE cleanup (e.g. the metrics-step failures). If a
  folder was fully cleaned up and only 'phy_export/' remains, there is nothing
  to safely re-analyze from here — re-run sorting for that recording instead.
"""

import sys
import argparse
import traceback
from pathlib import Path

import spikeinterface.full as si

# Reuse the exact post-sorting logic from sorting.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sorting import analyze_and_export  # noqa: E402


def _load_extractor(folder):
    """Load a saved SpikeInterface extractor folder across API versions."""
    last_err = None
    for fn_name in ("load", "load_extractor"):
        fn = getattr(si, fn_name, None)
        if fn is None:
            continue
        try:
            return fn(folder)
        except Exception as e:  # try the next loader
            last_err = e
    raise RuntimeError(f"Could not load '{folder}': {last_err}")


def load_sorting_and_recording(out_dir):
    """Return (sorting, recording, source_description) or (None, None, None)."""
    out = Path(out_dir)
    sorting_folder = out / "final_sorting_result"
    binary_folder = out / "processed_binary"
    analyzer_folder = out / "sorting_analyzer"

    # Preferred: the raw saved sorting + preprocessed recording (rebuild fresh).
    if sorting_folder.exists() and binary_folder.exists():
        sorting = _load_extractor(sorting_folder)
        recording = _load_extractor(binary_folder)
        return sorting, recording, "final_sorting_result + processed_binary"

    # Fallback: an existing analyzer that still carries its recording.
    if analyzer_folder.exists():
        try:
            analyzer = si.load_sorting_analyzer(analyzer_folder)
            if analyzer.recording is not None:
                return analyzer.sorting, analyzer.recording, "sorting_analyzer"
            print(f"  'sorting_analyzer' has no recording attached; cannot use it.")
        except Exception as e:
            print(f"  Could not load 'sorting_analyzer': {e}")

    return None, None, None


def find_sorting_outputs(output_folder):
    """Find the per-recording *_sorting_output subfolders (or the folder itself)."""
    op = Path(output_folder)
    subs = sorted(op.glob("*_sorting_output"))
    if not subs and op.name.endswith("_sorting_output"):
        subs = [op]
    return subs


def run(output_folder, n_jobs=4, cleanup=True):
    subs = find_sorting_outputs(output_folder)
    if not subs:
        print(f"No '*_sorting_output' folders found under '{output_folder}'. "
              f"Nothing to continue.")
        return

    print(f"Found {len(subs)} sorted recording(s) to continue.")
    for i, out_dir in enumerate(subs, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(subs)}] Continuing: {out_dir.name}")
        print(f"{'=' * 60}")

        # If already finalized (only phy_export), skip rather than re-do.
        if (out_dir / "phy_export").exists() and not (out_dir / "final_sorting_result").exists() \
                and not (out_dir / "sorting_analyzer").exists():
            print("  Already finalized (only 'phy_export' remains) — skipping. "
                  "Re-run sorting if you need to regenerate metrics/labels.")
            continue

        sorting, recording, source = load_sorting_and_recording(out_dir)
        if sorting is None:
            print("  No usable sorting + recording found "
                  "(need 'final_sorting_result' + 'processed_binary', or a "
                  "'sorting_analyzer' with a recording). Skipping.")
            continue

        print(f"  Loaded from: {source}")
        try:
            analyze_and_export(
                sorting, recording, out_dir,
                n_jobs=n_jobs, file_stem=out_dir.name, cleanup=cleanup,
            )
        except Exception as e:
            print(f"  Failed to continue '{out_dir.name}': {e}")
            traceback.print_exc()
            print("  Skipping to next recording...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continue post-sorting steps (analyzer, metrics, BombCell "
                    "labels, Phy export) without re-sorting.")
    parser.add_argument("--output_folder", required=True,
                        help="op* folder containing the *_sorting_output subfolders.")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Do NOT delete intermediate folders after export "
                             "(default is to clean up, leaving only phy_export).")
    # --config is accepted for runner compatibility but not required here.
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        run(args.output_folder, n_jobs=args.n_jobs, cleanup=not args.keep_intermediates)
    except Exception as e:
        print(f"\n[FATAL] continue_sorting crashed for '{args.output_folder}':\n{e}")
        traceback.print_exc()
        # Exit 0 so the batch runner continues to the next folder.
        sys.exit(0)
