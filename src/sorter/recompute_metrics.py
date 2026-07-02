"""
Recompute quality + waveform (template) metrics on a MANUALLY-CURATED phy folder
(runner step 'r'), AFTER you have merged / split / labeled units in Phy.

For each per-recording phy_export/ under the output (op) folder it:
  - loads the curated sorting (si.read_phy — merges/splits/labels applied),
  - rebuilds the preprocessed recording from recording.dat + params.py,
  - recomputes ALL quality metrics + template (waveform) metrics on the curated
    units, and writes them back into the phy folder as CSVs + cluster_*.tsv
    (visible in Phy) — WITHOUT re-exporting, so your manual curation is kept.

Usage:
    python recompute_metrics.py --output_folder <op_folder> [--n_jobs 4]
"""

import sys
import argparse
import traceback
from pathlib import Path

# Reuse the shared recompute logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sorter_common import recompute_curated_metrics  # noqa: E402


def find_phy_folders(output_folder):
    """Find curated phy_export folders under the op folder.

    Handles being pointed at: the op folder, a *_sorting_output folder, or a
    phy_export folder directly.
    """
    op = Path(output_folder)
    if (op / "params.py").exists():          # already a phy folder
        return [op]
    phys = sorted(op.glob("*_sorting_output/phy_export"))
    if not phys:
        phys = sorted(op.glob("*/phy_export"))   # any subfolder/phy_export
    if not phys and (op / "phy_export").exists():
        phys = [op / "phy_export"]
    return phys


def run(output_folder, n_jobs=4):
    phys = find_phy_folders(output_folder)
    if not phys:
        print(f"No curated 'phy_export' folder found under '{output_folder}'. "
              f"Nothing to recompute.")
        return

    print(f"Found {len(phys)} curated phy folder(s) to recompute.")
    for i, phy in enumerate(phys, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(phys)}] Recomputing metrics: {phy}")
        print(f"{'=' * 60}")
        try:
            recompute_curated_metrics(phy, n_jobs=n_jobs)
        except Exception as e:
            print(f"  Failed to recompute '{phy}': {e}")
            traceback.print_exc()
            print("  Skipping to next phy folder...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recompute quality + waveform metrics on a curated phy folder.")
    parser.add_argument("--output_folder", required=True,
                        help="op* folder (or a *_sorting_output / phy_export folder).")
    parser.add_argument("--n_jobs", type=int, default=4)
    # --config is accepted for runner compatibility but not required here.
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        run(args.output_folder, n_jobs=args.n_jobs)
    except Exception as e:
        print(f"\n[FATAL] recompute_metrics crashed for '{args.output_folder}':\n{e}")
        traceback.print_exc()
        sys.exit(0)  # exit 0 so the batch runner continues
