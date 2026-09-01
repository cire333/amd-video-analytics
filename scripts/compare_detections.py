"""Cross-stack comparison: AMD pipeline (detections.jsonl) vs DeepStream
(KITTI dumps). Greedy IoU matching per frame; reports agreement on what
was detected, class labels, confidences, and track statistics.

    python scripts/compare_detections.py <amd.jsonl> <kitti_det_dir> <kitti_track_dir> <report.md>
"""
import json
import os
import sys
from collections import Counter, defaultdict

IOU_MATCH = 0.5


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def load_amd(path):
    frames = {}
    for line in open(path):
        d = json.loads(line)
        frames[d["frame"]] = [
            {"label": o["label"], "conf": o["conf"], "bbox": o["bbox"],
             "track_id": o["track_id"]}
            for o in d["objects"]
        ]
    return frames


def load_kitti(det_dir, track_dir):
    frames, tracks = {}, {}
    # KITTI: <label> trunc occl alpha x1 y1 x2 y2 h w l x y z ry score
    # (track files insert the track id after the label). Labels can contain
    # spaces ("traffic light"), so anchor on the 15 numeric tail fields.
    for fn in sorted(os.listdir(det_dir)):
        idx = int(fn.split("_")[-1].split(".")[0])
        objs = []
        for line in open(os.path.join(det_dir, fn)):
            p = line.split()
            if len(p) < 16:
                continue
            objs.append({"label": " ".join(p[:-15]), "conf": float(p[-1]),
                         "bbox": [float(v) for v in p[-12:-8]], "track_id": None})
        frames[idx] = objs
    for fn in sorted(os.listdir(track_dir)):
        idx = int(fn.split("_")[-1].split(".")[0])
        objs = []
        for line in open(os.path.join(track_dir, fn)):
            p = line.split()
            if len(p) < 17:
                continue
            objs.append({"label": " ".join(p[:-16]), "track_id": int(p[-16]),
                         "bbox": [float(v) for v in p[-12:-8]]})
        tracks[idx] = objs
    return frames, tracks


def track_stats(per_frame_objs):
    lengths = defaultdict(int)
    for objs in per_frame_objs.values():
        for o in objs:
            if o["track_id"] is not None:
                lengths[o["track_id"]] += 1
    n = len(lengths)
    if not n:
        return 0, 0.0, 0
    vals = sorted(lengths.values())
    return n, sum(vals) / n, sum(1 for v in vals if v >= 10)


def main():
    amd_path, det_dir, track_dir, report_path = sys.argv[1:5]
    amd = load_amd(amd_path)
    ds, ds_tracks = load_kitti(det_dir, track_dir)

    frames = sorted(set(amd) & set(ds))
    matched = amd_only = ds_only = 0
    class_agree = 0
    iou_sum = conf_diff_sum = 0.0
    amd_cls, ds_cls = Counter(), Counter()
    mismatch_pairs = Counter()
    per_frame_counts = []

    for fi in frames:
        a_objs, d_objs = list(amd[fi]), list(ds[fi])
        amd_cls.update(o["label"] for o in a_objs)
        ds_cls.update(o["label"] for o in d_objs)
        per_frame_counts.append((fi, len(a_objs), len(d_objs)))

        pairs = sorted(
            ((iou(a["bbox"], d["bbox"]), i, j)
             for i, a in enumerate(a_objs) for j, d in enumerate(d_objs)),
            reverse=True)
        used_a, used_d = set(), set()
        for v, i, j in pairs:
            if v < IOU_MATCH or i in used_a or j in used_d:
                continue
            used_a.add(i); used_d.add(j)
            matched += 1
            iou_sum += v
            conf_diff_sum += abs(a_objs[i]["conf"] - d_objs[j]["conf"])
            if a_objs[i]["label"] == d_objs[j]["label"]:
                class_agree += 1
            else:
                mismatch_pairs[(a_objs[i]["label"], d_objs[j]["label"])] += 1
        amd_only += len(a_objs) - len(used_a)
        ds_only += len(d_objs) - len(used_d)

    amd_n_tracks, amd_mean_len, amd_long = track_stats(amd)
    ds_n_tracks, ds_mean_len, ds_long = track_stats(ds_tracks)

    total_amd = sum(amd_cls.values())
    total_ds = sum(ds_cls.values())
    lines = [
        "# AMD pipeline vs NVIDIA DeepStream — YOLO26-m comparison",
        "",
        f"Frames compared: **{len(frames)}** | IoU match threshold: {IOU_MATCH}",
        "",
        "## Detection agreement",
        "",
        f"| | AMD (R9700/MIGraphX) | DeepStream (3090/TensorRT) |",
        f"|---|---|---|",
        f"| total detections | {total_amd} | {total_ds} |",
        f"| per class | {dict(amd_cls)} | {dict(ds_cls)} |",
        "",
        f"- **Matched (same object, IoU>={IOU_MATCH}): {matched}** — "
        f"{matched / max(total_amd, 1):.1%} of AMD, {matched / max(total_ds, 1):.1%} of DS",
        f"- AMD-only detections: {amd_only} | DS-only detections: {ds_only}",
        f"- Mean IoU of matched pairs: {iou_sum / max(matched, 1):.3f}",
        f"- Class label agreement on matched pairs: {class_agree}/{matched} "
        f"({class_agree / max(matched, 1):.1%})",
        f"- Class disagreements (AMD label, DS label): "
        f"{dict(mismatch_pairs) or 'none'}",
        f"- Mean |confidence delta| on matched pairs: "
        f"{conf_diff_sum / max(matched, 1):.4f}",
        "",
        "## Tracking",
        "",
        f"| | AMD (greedy IoU tracker) | DeepStream (NvDsTracker IOU) |",
        f"|---|---|---|",
        f"| unique track ids | {amd_n_tracks} | {ds_n_tracks} |",
        f"| mean track length (frames) | {amd_mean_len:.1f} | {ds_mean_len:.1f} |",
        f"| tracks >= 10 frames | {amd_long} | {ds_long} |",
        "",
        "## Per-frame object counts (first/mid/last)",
        "",
        "| frame | AMD | DS |",
        "|---|---|---|",
    ]
    for fi, na, nd in (per_frame_counts[:3]
                       + per_frame_counts[len(per_frame_counts) // 2 - 1:
                                          len(per_frame_counts) // 2 + 2]
                       + per_frame_counts[-3:]):
        lines.append(f"| {fi} | {na} | {nd} |")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
