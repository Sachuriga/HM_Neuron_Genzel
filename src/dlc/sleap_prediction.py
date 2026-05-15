"""Run SLEAP inference on rat videos using the trained model."""
from pathlib import Path
import sleap_io as sio
from sleap_nn.predict import run_inference

# ---------- Config ----------
MODEL_PATH = Path(
    r"\\genzelneuron.z.science.ru.nl\genzelneuron\code\HM_Sleap\src\models\sachi_test-2"
)
OUTPUT_DIR = Path(
    r"\\genzelneuron.z.science.ru.nl\genzelneuron\code\HM_Sleap\src\predictions"
)
VIDEOS = [
    Path(r"\\genzelneuron.z.science.ru.nl\genzelneuron\data\HM_sleap\collected_frames_Rat1_20200914.mp4"),
    # add more videos here
]
DEVICE = "cuda"        # "cuda" / "cuda:0" / "cpu"
BATCH_SIZE = 8         # bump to 16 or 32 if VRAM allows
# ----------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def predict_one(video_path: Path) -> Path:
    """Run inference on one video. Returns the output .slp path."""
    out_path = OUTPUT_DIR / f"{video_path.stem}.predictions.slp"
    print(f"\n=== {video_path.name} -> {out_path.name} ===")

    run_inference(
        data_path=str(video_path),
        model_paths=[str(MODEL_PATH)],
        output_path=str(out_path),
        make_labels=True,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        # frames=None  -> entire video (this is the default; don't pass anything)
    )

    # Sanity check — the silent-0-frame bug bit you once, don't trust it
    labels = sio.load_slp(str(out_path))
    n_frames = len(labels.labeled_frames)
    n_total = labels.videos[0].shape[0]
    pct = 100.0 * n_frames / n_total if n_total else 0
    print(f"   predicted {n_frames}/{n_total} frames ({pct:.1f}%)")
    if n_frames < 0.5 * n_total:
        print(f"   WARNING: fewer than half the frames have predictions")

    return out_path


def main():
    results = []
    for video in VIDEOS:
        if not video.exists():
            print(f"SKIP (not found): {video}")
            continue
        try:
            results.append(predict_one(video))
        except Exception as e:
            print(f"FAILED on {video.name}: {e}")
    print(f"\nDone. {len(results)} file(s) written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()