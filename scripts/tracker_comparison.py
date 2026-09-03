"""Compare tracking strategies by replaying saved detections (no GPU).

    PYTHONPATH=src .venv/bin/python scripts/tracker_comparison.py \
        results/nds ~/Downloads/nds-monitoring \
        [--segment 160850:40200:900] [--workers 12]

Metrics per tracker (aggregated over all videos):
- tracks: confirmed track count (lower is better ONLY if objects aren't merged)
- >=5s: tracks lasting at least 5 seconds
- med/mean len: track length distribution (longer = less fragmentation)
- frag/min: terminations followed <=2 s later by a birth within 80 px —
  fragments the tracker failed to bridge internally (lower is better)
- births/min: new confirmed tracks per minute

--segment camera:start_frame:n_frames renders a 2x2 side-by-side video
(sort | bytetrack | bytetrack-long | ocsort) over the same source frames so
id stability can be judged visually.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from avap.bytetrack import ByteTracker            # noqa: E402
from avap.frame import ObjectMeta                 # noqa: E402
from avap.kalman_tracker import SortTracker       # noqa: E402
from avap.ocsort import OcSortTracker             # noqa: E402
from avap.tracker import IouTracker               # noqa: E402

FPS = 25

STRATEGIES = {
    "iou":            lambda: IouTracker(iou_threshold=0.3, max_age=10),
    "sort":           lambda: SortTracker(0.3, max_age=15, min_hits=3),
    "bytetrack":      lambda: ByteTracker(0.3, max_age=25, min_hits=3),
    "bytetrack-long": lambda: ByteTracker(0.3, max_age=62, min_hits=5),
    "ocsort":         lambda: OcSortTracker(0.25, max_age=62, min_hits=3),
    "ocsort-long":    lambda: OcSortTracker(0.25, max_age=90, min_hits=5),
}
GRID_STRATEGIES = ["sort", "bytetrack", "bytetrack-long", "ocsort"]


def load_dets(jsonl: str):
    frames = []
    for line in open(jsonl):
        d = json.loads(line)
        frames.append([(tuple(o["bbox"]), o["conf"], o["label"], o["class_id"])
                       for o in d["objects"]])
    return frames


def replay(args):
    jsonl, strategy, keep_window = args
    frames = load_dets(jsonl)
    tracker = STRATEGIES[strategy]()
    tr = {}
    window: list[list] = []
    for fi, dets in enumerate(frames):
        objs = [ObjectMeta(class_id=c, confidence=conf, bbox=b, label=lb)
                for b, conf, lb, c in dets]
        tracked = tracker.update(objs)
        if keep_window and keep_window[0] <= fi < keep_window[0] + keep_window[1]:
            window.append([(o.track_id, o.label, o.confidence, o.bbox)
                           for o in tracked if o.track_id is not None])
        for o in tracked:
            if o.track_id is None:
                continue
            cx, cy = (o.bbox[0] + o.bbox[2]) / 2, (o.bbox[1] + o.bbox[3]) / 2
            rec = tr.setdefault(o.track_id, {"first": fi, "fpos": (cx, cy),
                                             "last": fi, "lpos": (cx, cy),
                                             "len": 0})
            rec["last"], rec["lpos"] = fi, (cx, cy)
            rec["len"] += 1

    lengths = np.array([v["len"] for v in tr.values()]) if tr else np.array([0])
    minutes = len(frames) / FPS / 60
    # fragmentation proxy: a track ends, another begins nearby soon after
    starts = sorted(tr.values(), key=lambda v: v["first"])
    frags = 0
    start_frames = np.array([w["first"] for w in starts])
    for v in tr.values():
        lo = int(np.searchsorted(start_frames, v["last"] + 1))
        hi = int(np.searchsorted(start_frames, v["last"] + 2 * FPS, side="right"))
        for w in starts[lo:hi]:
            if np.hypot(w["fpos"][0] - v["lpos"][0],
                        w["fpos"][1] - v["lpos"][1]) <= 80:
                frags += 1
                break
    return {"video": Path(jsonl).parent.name, "strategy": strategy,
            "tracks": len(tr), "ge5s": int((lengths >= 5 * FPS).sum()),
            "med_len": float(np.median(lengths)),
            "mean_len": float(lengths.mean()),
            "frag_per_min": frags / minutes, "births_per_min": len(tr) / minutes,
            "window": window}


def render_grid(source_video: str, start: int, n_frames: int,
                per_strategy_windows: dict[str, list], out_path: str):
    import cv2

    def color(tid):
        rng = np.random.default_rng(tid)
        return tuple(int(c) for c in rng.integers(60, 255, 3))

    cap = cv2.VideoCapture(source_video)
    for _ in range(start):
        cap.grab()
    writer = None
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        tiles = []
        for name in GRID_STRATEGIES:
            tile = frame.copy()
            win = per_strategy_windows[name]
            if i < len(win):
                for tid, label, conf, bbox in win[i]:
                    x1, y1, x2, y2 = (int(v) for v in bbox)
                    c = color(tid)
                    cv2.rectangle(tile, (x1, y1), (x2, y2), c, 2)
                    cv2.putText(tile, f"#{tid}", (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            cv2.putText(tile, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
            tiles.append(tile)
        top = np.hstack(tiles[:2])
        bottom = np.hstack(tiles[2:])
        grid = np.vstack([top, bottom])
        if writer is None:
            writer = cv2.VideoWriter(out_path + ".raw.mp4",
                                     cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                     (grid.shape[1], grid.shape[0]))
        writer.write(grid)
    if writer:
        writer.release()
    cap.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i",
                    out_path + ".raw.mp4", "-c:v", "libx264", "-preset",
                    "veryfast", "-pix_fmt", "yuv420p", out_path], check=True)
    Path(out_path + ".raw.mp4").unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("videos_root")
    ap.add_argument("--segment", default=None,
                    help="camera:start_frame:n_frames for the 2x2 grid video")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    jsonls = sorted(str(p) for p in Path(args.results_dir).rglob("detections.jsonl"))
    seg_cam, seg_start, seg_n = (None, 0, 0)
    if args.segment:
        seg_cam, s, n = args.segment.split(":")
        seg_start, seg_n = int(s), int(n)

    jobs = []
    for j in jsonls:
        cam = Path(j).parent.parent.name
        window = ((seg_start, seg_n)
                  if cam == seg_cam else None)
        for strat in STRATEGIES:
            jobs.append((j, strat, window if strat in GRID_STRATEGIES else None))

    with Pool(args.workers) as pool:
        results = pool.map(replay, jobs)

    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        for k in ("tracks", "ge5s", "med_len", "mean_len",
                  "frag_per_min", "births_per_min"):
            agg[r["strategy"]][k].append(r[k])

    print(f"\n{'strategy':>15} {'tracks':>7} {'>=5s':>6} {'med_len':>8} "
          f"{'mean_len':>8} {'frag/min':>9} {'births/min':>10}")
    for strat in STRATEGIES:
        a = agg[strat]
        print(f"{strat:>15} {sum(a['tracks']):>7} {sum(a['ge5s']):>6} "
              f"{np.mean(a['med_len']):>8.1f} {np.mean(a['mean_len']):>8.1f} "
              f"{np.mean(a['frag_per_min']):>9.2f} "
              f"{np.mean(a['births_per_min']):>10.1f}")

    if seg_cam:
        windows = {r["strategy"]: r["window"] for r in results if r["window"]}
        # find the source video for that camera's processed stem
        stem = next(Path(j).parent.name for j in jsonls
                    if Path(j).parent.parent.name == seg_cam)
        src = next(Path(args.videos_root).rglob(f"{stem}.mp4"))
        out = str(Path(args.results_dir) / f"tracker_grid_{seg_cam}.mp4")
        render_grid(str(src), seg_start, seg_n, windows, out)
        print(f"\ngrid video: {out}")


if __name__ == "__main__":
    main()
