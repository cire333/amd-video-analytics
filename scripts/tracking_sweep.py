"""Offline tracking-parameter sweep against a hand-counted ground truth.

Replays saved AMD detections (detections.jsonl) through SortTracker with
different configurations — no GPU rerun, no core-code changes — and counts
VEHICLE tracks, excluding the smallest 12% of bounding boxes by area (the
same absolute pixel cutoff is applied to the DeepStream KITTI tracks for a
fair read). Reports total tracks and per-6-minute-window counts to compare
against the manual count.

    python scripts/tracking_sweep.py <detections.jsonl> <kitti_track_dir>

Ground truth (hand count by Eric, 2026-09-01): ~166 vehicles / 6 min,
extrapolated ~830 over the 29.8-min video.
"""
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from avap.frame import ObjectMeta            # noqa: E402
from avap.kalman_tracker import SortTracker  # noqa: E402
from avap.tracker import IouTracker          # noqa: E402

VEHICLES = {"car", "truck", "bus", "motorcycle"}
FPS = 15
WINDOW_FRAMES = 6 * 60 * FPS  # 6-minute windows, matching the hand count
GT_PER_WINDOW = 166
GT_TOTAL = 830


def load_frames(jsonl_path):
    frames = []
    for line in open(jsonl_path):
        d = json.loads(line)
        frames.append([(tuple(o["bbox"]), o["conf"], o["label"])
                       for o in d["objects"] if o["label"] in VEHICLES])
    return frames


def area(bbox):
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def stitch(tracks, gap_max, dist_max):
    """Tracklet stitching: greedily chain a track with the nearest track that
    STARTS after it ends (within gap_max frames, start within dist_max px of
    the end position). Overlapping-in-time tracks are never stitched — two
    vehicles visible simultaneously are distinct.
    tracks: {tid: dict(first, last, fpos, lpos, len)} -> merged dict."""
    ids = sorted(tracks, key=lambda t: tracks[t]["first"])
    consumed = set()
    merged = {}
    for t in ids:
        if t in consumed:
            continue
        cur = dict(tracks[t])
        while True:
            best, best_d = None, dist_max
            for b in ids:
                if b in consumed or b == t:
                    continue
                fb = tracks[b]
                gap = fb["first"] - cur["last"]
                if gap <= 0 or gap > gap_max:
                    continue
                d = np.hypot(fb["fpos"][0] - cur["lpos"][0],
                             fb["fpos"][1] - cur["lpos"][1])
                if d < best_d:
                    best, best_d = b, d
            if best is None:
                break
            fb = tracks[best]
            consumed.add(best)
            cur["last"], cur["lpos"] = fb["last"], fb["lpos"]
            cur["len"] += fb["len"]
        merged[t] = cur
    return merged


def replay(args):
    (frames, cutoff, conf_min, mechanism, max_age, min_hits,
     min_disp, min_len, stitch_gap) = args
    if mechanism == "sort":
        tracker = SortTracker(iou_threshold=0.3, max_age=max_age, min_hits=min_hits)
    else:
        tracker = IouTracker(iou_threshold=0.3, max_age=max_age)

    tr = {}
    for fi, dets in enumerate(frames):
        objs = [ObjectMeta(class_id=0, confidence=c, bbox=b, label=lb)
                for b, c, lb in dets if area(b) >= cutoff and c >= conf_min]
        for o in tracker.update(objs):
            if o.track_id is None:
                continue
            cx, cy = (o.bbox[0] + o.bbox[2]) / 2, (o.bbox[1] + o.bbox[3]) / 2
            if o.track_id not in tr:
                tr[o.track_id] = {"first": fi, "fpos": (cx, cy), "last": fi,
                                  "lpos": (cx, cy), "len": 0}
            tr[o.track_id]["last"] = fi
            tr[o.track_id]["lpos"] = (cx, cy)
            tr[o.track_id]["len"] += 1

    if stitch_gap > 0:
        tr = stitch(tr, gap_max=stitch_gap, dist_max=80)

    kept = {t for t, v in tr.items() if v["len"] >= min_len}
    if min_disp > 0:
        kept = {t for t in kept
                if np.hypot(tr[t]["lpos"][0] - tr[t]["fpos"][0],
                            tr[t]["lpos"][1] - tr[t]["fpos"][1]) >= min_disp}

    windows = defaultdict(int)
    for t in kept:
        windows[tr[t]["first"] // WINDOW_FRAMES] += 1
    n_windows = (len(frames) + WINDOW_FRAMES - 1) // WINDOW_FRAMES
    per_window = [windows.get(w, 0) for w in range(n_windows)]
    return {
        "mechanism": mechanism, "conf_min": conf_min, "max_age": max_age,
        "min_hits": min_hits, "min_disp": min_disp, "min_len": min_len,
        "stitch_gap": stitch_gap,
        "tracks": len(kept), "per_window": per_window,
        "mean_len": (np.mean([tr[t]["len"] for t in kept]) if kept else 0.0),
    }


def ds_track_counts(track_dir, cutoff, min_disp=0):
    per_track = defaultdict(list)  # tid -> [(frame, cx, cy, label, area)]
    for fn in sorted(os.listdir(track_dir)):
        fi = int(fn.split("_")[-1].split(".")[0])
        for ln in open(os.path.join(track_dir, fn)):
            p = ln.split()
            if len(p) < 17:
                continue
            label = " ".join(p[:-16])
            if label not in VEHICLES:
                continue
            x1, y1, x2, y2 = map(float, p[-12:-8])
            if (x2 - x1) * (y2 - y1) < cutoff:
                continue
            per_track[int(p[-16])].append(
                (fi, (x1 + x2) / 2, (y1 + y2) / 2))
    kept = {}
    for tid, rows in per_track.items():
        if len(rows) < 3:
            continue
        rows.sort()
        disp = np.hypot(rows[-1][1] - rows[0][1], rows[-1][2] - rows[0][2])
        if disp < min_disp:
            continue
        kept[tid] = rows[0][0]
    windows = defaultdict(int)
    for tid, f0 in kept.items():
        windows[f0 // WINDOW_FRAMES] += 1
    return len(kept), [windows.get(w, 0) for w in range(5)]


def main():
    jsonl_path, track_dir = sys.argv[1], sys.argv[2]
    frames = load_frames(jsonl_path)

    all_areas = np.array([area(b) for dets in frames for b, _, _ in dets])
    cutoff = float(np.percentile(all_areas, 12))
    side = int(np.sqrt(cutoff))
    print(f"vehicle detections: {len(all_areas)}  |  12th-pct area cutoff: "
          f"{cutoff:.0f} px^2 (~{side}x{side})  |  frames: {len(frames)} "
          f"({len(frames) / FPS / 60:.1f} min)")
    print(f"ground truth: ~{GT_PER_WINDOW}/6min window, ~{GT_TOTAL} total\n")

    grid = []
    for conf_min in (0.40, 0.50):
        for max_age, min_hits in ((30, 3), (45, 8)):
            for min_disp in (50, 100):
                for min_len in (0, 15, 30):          # 0 / 1s / 2s
                    for stitch_gap in (0, 45, 90):   # off / 3s / 6s
                        grid.append((frames, cutoff, conf_min, "sort", max_age,
                                     min_hits, min_disp, min_len, stitch_gap))

    with Pool(min(8, os.cpu_count())) as pool:
        results = pool.map(replay, grid)

    print(f"{'mech':>6} {'conf':>5} {'age':>4} {'hits':>4} {'disp':>4} "
          f"{'mlen':>4} {'stit':>4} {'tracks':>6} {'d830':>6} "
          f"{'per-6min-window':>28} {'meanlen':>7}")
    results.sort(key=lambda r: abs(r["tracks"] - GT_TOTAL))
    for r in results[:20]:
        pw = " ".join(f"{v:>4}" for v in r["per_window"])
        print(f"{r['mechanism']:>6} {r['conf_min']:>5.2f} {r['max_age']:>4} "
              f"{r['min_hits']:>4} {r['min_disp']:>4} {r['min_len']:>4} "
              f"{r['stitch_gap']:>4} {r['tracks']:>6} "
              f"{r['tracks'] - GT_TOTAL:>+6} {pw:>28} {r['mean_len']:>7.1f}")

    for disp in (0, 50):
        n, pw = ds_track_counts(track_dir, cutoff, disp)
        pws = " ".join(f"{v:>4}" for v in pw)
        print(f"\nDeepStream IOU tracker (same cutoff, disp>={disp}): "
              f"{n} tracks ({n - GT_TOTAL:+d} vs GT)  windows: {pws}")


if __name__ == "__main__":
    main()
