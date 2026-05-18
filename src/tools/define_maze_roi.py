"""
define_maze_roi.py — draw a polygon on a video frame to define the maze boundary.

Run this ONCE when the camera is set up.  The result is saved as
src/tools/maze_roi.txt and committed to the repo so every user gets the
same ROI without having to redraw it.

The tracker loads src/tools/maze_roi.txt automatically at startup.

Usage:
    python src/tools/define_maze_roi.py --video path/to/stitched.mp4
    python src/tools/define_maze_roi.py --video path/to/stitched.mp4 --frame 500

Output:
    src/tools/maze_roi.txt  (always saved here — commit this file to the repo)

Controls:
    Left click   : add a vertex
    Right click  : remove the last vertex
    Enter/Space  : save polygon and exit
    Esc          : exit without saving
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Must match the display resolution used by the tracker
DISPLAY_W, DISPLAY_H = 1176, 712

WINDOW_NAME   = "Define Maze ROI"
LINE_COLOR    = (0, 255, 0)
VERTEX_COLOR  = (0, 200, 255)
CLOSE_COLOR   = (0, 165, 255)
FILL_ALPHA    = 0.15
VERTEX_RADIUS = 6

INSTRUCTIONS = [
    "Left click  : add vertex",
    "Right click : remove last vertex",
    "Enter/Space : save & exit",
    "Esc         : cancel",
]


class _ROIDrawer:
    def __init__(self, base_frame: np.ndarray):
        self.base   = base_frame.copy()
        self.points: list[tuple[int, int]] = []

    # ------------------------------------------------------------------ drawing

    def render(self) -> np.ndarray:
        img = self.base.copy()
        pts = self.points
        n   = len(pts)

        if n >= 2:
            for i in range(n - 1):
                cv2.line(img, pts[i], pts[i + 1], LINE_COLOR, 2, cv2.LINE_AA)

        if n >= 3:
            # closing edge (dashed-style: just draw it in a different colour)
            cv2.line(img, pts[-1], pts[0], CLOSE_COLOR, 1, cv2.LINE_AA)
            # semi-transparent fill
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (0, 255, 0))
            cv2.addWeighted(overlay, FILL_ALPHA, img, 1 - FILL_ALPHA, 0, img)

        for pt in pts:
            cv2.circle(img, pt, VERTEX_RADIUS, VERTEX_COLOR, -1, cv2.LINE_AA)

        # instructions overlay
        for idx, text in enumerate(INSTRUCTIONS):
            y = 25 + idx * 24
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # vertex count
        status = f"Vertices: {n}  (need >= 3 to save)"
        cv2.putText(img, status, (10, DISPLAY_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        return img

    # ------------------------------------------------------------------ events

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.imshow(WINDOW_NAME, self.render())
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()
            cv2.imshow(WINDOW_NAME, self.render())


# --------------------------------------------------------------------------- #

def _extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx   = max(0, min(frame_idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit(f"ERROR: could not read frame {idx} from {video_path}")
    print(f"Using frame {idx} / {total - 1}  ({video_path.name})")
    return cv2.resize(frame, (DISPLAY_W, DISPLAY_H))


def main():
    # Always save next to this script: src/tools/maze_roi.txt
    out_path = Path(__file__).parent / "maze_roi.txt"

    parser = argparse.ArgumentParser(description="Draw maze ROI polygon on a video frame")
    parser.add_argument("--video", required=True, help="Path to video (e.g. stitched.mp4)")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to display (default: 0)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"ERROR: video not found: {video_path}")

    frame  = _extract_frame(video_path, args.frame)
    drawer = _ROIDrawer(frame)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback(WINDOW_NAME, drawer.on_mouse)
    cv2.imshow(WINDOW_NAME, drawer.render())

    print("Click to define the maze polygon.  Enter/Space to save,  Esc to cancel.")

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 32):   # Enter or Space
            break
        if k == 27:         # Esc
            print("Cancelled — nothing saved.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    if len(drawer.points) < 3:
        sys.exit("Need at least 3 vertices to form a polygon — nothing saved.")

    lines = [
        f"# Maze ROI polygon — display resolution {DISPLAY_W}x{DISPLAY_H}",
        f"# Source: {video_path.name}  frame {args.frame}",
        f"# Commit this file to the repo. Do not edit manually.",
        f"# Format: x,y  (one vertex per line)",
    ]
    for x, y in drawer.points:
        lines.append(f"{x},{y}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {len(drawer.points)}-vertex ROI → {out_path}")
    print("Commit src/tools/maze_roi.txt to the repo.")


if __name__ == "__main__":
    main()
