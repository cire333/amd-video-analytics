#!/usr/bin/env python3
"""NvDCF (visual, HIP) vs motion-only trackers on NDS footage, same metrics as
scripts/tracker_comparison.py but with video frames decoded for the visual tracker.

    PYTHONPATH=src python scripts/nvdcf_comparison.py results/nds ~/Downloads/nds-monitoring \
        --cameras 160850 164079 --frames 15000 --strategies ocsort-long nvdcf nvdcf-perf

Per camera the saved detections (detections.jsonl) are replayed frame-aligned
with the source video (cv2 decode -> device upload). Reports tracks, >=5s
tracks, median/mean length, frag/min, births/min, plus how many boxes the DCF
bridged during detector misses and the tracker's ms/frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from avap.bytetrack import ByteTracker            # noqa: E402
from avap.frame import ObjectMeta                 # noqa: E402
from avap.nvdcf import FrameUploader, NvDcfConfig, NvDcfTracker  # noqa: E402
from avap.ocsort import OcSortTracker             # noqa: E402

FPS = 25


def nvdcf_cfg(profile: str) -> NvDcfConfig:
    """Profiles mirror DS's perf/accuracy yml values (feature level, HOG, lr, sigma,
    association thresholds); shadow age matches the ocsort-long memory (3.6 s)."""
    common = dict(maxShadowTrackingAge=90, probationAge=4, earlyTerminationAge=2,
                  checkClassMatch=0, minDetectorConfidence=0.1, tentativeDetectorConfidence=0.3,
                  maxVisualOnlyAge=60)
    if profile == "perf":
        return NvDcfConfig(featureImgSizeLevel=2, useHog=0, filterLr=0.075, gaussianSigma=2.0,
                           minIouDiff4NewTarget=0.74, minTrackerConfidence=0.4,
                           minMatchingScore4Overall=0.43, minMatchingScore4SizeSimilarity=0.36,
                           minMatchingScore4Iou=0.26, minMatchingScore4VisualSimilarity=0.54,
                           matchingScoreWeight4VisualSimilarity=0.34,
                           matchingScoreWeight4SizeSimilarity=0.44, matchingScoreWeight4Iou=0.37,
                           processNoiseVar4Loc=1.5, processNoiseVar4Size=1.3, processNoiseVar4Vel=0.03,
                           measurementNoiseVar4Detector=3.0, measurementNoiseVar4Tracker=8.2, **common)
    # accuracy-like
    return NvDcfConfig(featureImgSizeLevel=3, useHog=1, filterLr=0.0767, gaussianSigma=2.0,
                       minIouDiff4NewTarget=0.37, minTrackerConfidence=0.25,
                       minMatchingScore4Overall=0.05, minMatchingScore4SizeSimilarity=0.35,
                       minMatchingScore4Iou=0.05, minMatchingScore4VisualSimilarity=0.5,
                       matchingScoreWeight4VisualSimilarity=0.4,
                       matchingScoreWeight4SizeSimilarity=0.6, matchingScoreWeight4Iou=0.4,
                       processNoiseVar4Loc=60.0, processNoiseVar4Size=15.0, processNoiseVar4Vel=13.0,
                       measurementNoiseVar4Detector=100.0, measurementNoiseVar4Tracker=293.0, **common)


def make_tracker(name: str):
    if name == "ocsort-long":
        return OcSortTracker(0.25, max_age=90, min_hits=5), False
    if name == "bytetrack-long":
        return ByteTracker(0.3, max_age=62, min_hits=5), False
    if name == "nvdcf":
        return NvDcfTracker(nvdcf_cfg("accuracy"), backend="hip"), True
    if name == "nvdcf-perf":
        return NvDcfTracker(nvdcf_cfg("perf"), backend="hip"), True
    if name == "nvdcf-noframes":       # same association/lifecycle, no pixels
        return NvDcfTracker(nvdcf_cfg("accuracy"), backend="numpy"), False
    raise SystemExit(f"unknown strategy {name}")


def load_dets(jsonl: str, limit: int):
    frames = []
    with open(jsonl) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            d = json.loads(line)
            frames.append([(tuple(o["bbox"]), o["conf"], o["label"], o["class_id"])
                           for o in d["objects"]])
    return frames


def metrics(tr: dict, n_frames: int) -> dict:
    lengths = np.array([v["len"] for v in tr.values()]) if tr else np.array([0])
    minutes = n_frames / FPS / 60
    starts = sorted(tr.values(), key=lambda v: v["first"])
    start_frames = np.array([w["first"] for w in starts])
    frags = 0
    for v in tr.values():
        lo = int(np.searchsorted(start_frames, v["last"] + 1))
        hi = int(np.searchsorted(start_frames, v["last"] + 2 * FPS, side="right"))
        for w in starts[lo:hi]:
            if np.hypot(w["fpos"][0] - v["lpos"][0], w["fpos"][1] - v["lpos"][1]) <= 80:
                frags += 1
                break
    return {"tracks": len(tr), "ge5s": int((lengths >= 5 * FPS).sum()),
            "med_len": float(np.median(lengths)), "mean_len": float(lengths.mean()),
            "frag_per_min": frags / minutes, "births_per_min": len(tr) / minutes}


def replay(jsonl: str, video: str, strategy: str, limit: int, uploader: FrameUploader | None):
    import cv2
    frames = load_dets(jsonl, limit)
    tracker, needs_frames = make_tracker(strategy)
    cap = cv2.VideoCapture(video) if needs_frames else None
    tr: dict = {}
    bridged = 0
    t_track = 0.0
    for fi, dets in enumerate(frames):
        objs = [ObjectMeta(class_id=c, confidence=conf, bbox=b, label=lb) for b, conf, lb, c in dets]
        vf = None
        if needs_frames:
            ok, img = cap.read()
            if not ok:
                break
            vf = uploader.upload(img)
        t0 = time.perf_counter()
        tracked = tracker.update(objs, vf) if needs_frames else tracker.update(objs)
        t_track += time.perf_counter() - t0
        for o in tracked:
            if o.track_id is None:
                continue
            if o.tracked_by == "dcf":
                bridged += 1
            cx, cy = (o.bbox[0] + o.bbox[2]) / 2, (o.bbox[1] + o.bbox[3]) / 2
            rec = tr.setdefault(o.track_id, {"first": fi, "fpos": (cx, cy), "last": fi,
                                             "lpos": (cx, cy), "len": 0})
            rec["last"], rec["lpos"] = fi, (cx, cy)
            rec["len"] += 1
    if cap is not None:
        cap.release()
    m = metrics(tr, len(frames))
    m.update({"bridged": bridged, "ms_per_frame": 1e3 * t_track / max(len(frames), 1),
              "frames": len(frames)})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("videos_root")
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--frames", type=int, default=15000)
    ap.add_argument("--strategies", nargs="+", default=["ocsort-long", "nvdcf-noframes", "nvdcf-perf", "nvdcf"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    jsonls = sorted(str(p) for p in Path(args.results_dir).rglob("detections.jsonl"))
    if args.cameras:
        jsonls = [j for j in jsonls if Path(j).parent.parent.name in args.cameras]
    uploader = FrameUploader() if any(s.startswith("nvdcf") and s != "nvdcf-noframes"
                                      for s in args.strategies) else None
    agg = defaultdict(lambda: defaultdict(list))
    rows = []
    for j in jsonls:
        cam, stem = Path(j).parent.parent.name, Path(j).parent.name
        video = next(Path(args.videos_root).rglob(f"{stem}.mp4"), None)
        if video is None:
            print(f"skip {cam}/{stem}: no video"); continue
        for strat in args.strategies:
            t0 = time.time()
            m = replay(j, str(video), strat, args.frames, uploader)
            m.update({"camera": cam, "strategy": strat, "wall_s": time.time() - t0})
            rows.append(m)
            print(f"{cam} {strat:>15}: tracks={m['tracks']:5d} >=5s={m['ge5s']:4d} med={m['med_len']:6.1f} "
                  f"mean={m['mean_len']:6.1f} frag/min={m['frag_per_min']:6.2f} births/min={m['births_per_min']:6.1f} "
                  f"bridged={m['bridged']:6d} {m['ms_per_frame']:5.2f} ms/frame", flush=True)
            for k in ("tracks", "ge5s", "med_len", "mean_len", "frag_per_min", "births_per_min", "bridged", "ms_per_frame"):
                agg[strat][k].append(m[k])
    if uploader:
        uploader.close()
    print(f"\n{'strategy':>15} {'tracks':>7} {'>=5s':>6} {'med_len':>8} {'mean_len':>8} {'frag/min':>9} "
          f"{'births/min':>10} {'bridged':>8} {'ms/frame':>8}")
    for strat in args.strategies:
        a = agg[strat]
        if not a:
            continue
        print(f"{strat:>15} {sum(a['tracks']):>7} {sum(a['ge5s']):>6} {np.mean(a['med_len']):>8.1f} "
              f"{np.mean(a['mean_len']):>8.1f} {np.mean(a['frag_per_min']):>9.2f} "
              f"{np.mean(a['births_per_min']):>10.1f} {sum(a['bridged']):>8} {np.mean(a['ms_per_frame']):>8.2f}")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
