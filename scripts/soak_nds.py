"""12-hour soak: the 13-camera NDS batch on repeat (release validation).

    nohup sg render -c "PYTHONPATH=/opt/rocm/lib:src python scripts/soak_nds.py \
        <input_dir> <model.onnx> <work_dir> --hours 12" &

Each cycle runs process_nds_batch (one full ~30-min video per camera, 13
parallel workers: decode -> MIGraphX fp16 @1280 -> ByteTrack -> jsonl +
VCN-encoded annotated video), validates every output (detections present,
annotated frame count == processed frames), records GPU temp/VRAM and
throughput, then deletes the ~20 GB of outputs before the next cycle.
Cycle 1 keeps one annotated sample. A failing cycle is recorded and the
soak continues. Stops early if free disk drops below 40 GB.

Results: <work_dir>/soak_summary.jsonl (one line per cycle) + soak.log.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIN_FREE_GB = 40


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def gpu_stats() -> dict:
    out = {}
    try:
        r = subprocess.run(["rocm-smi", "--json", "-t", "--showmemuse",
                            "--showpower"], capture_output=True, text=True,
                           timeout=30)
        data = json.loads(r.stdout)
        card = next(v for k, v in data.items() if k.startswith("card"))
        for key, val in card.items():
            lk = key.lower()
            if "temperature" in lk and "junction" in lk:
                out["temp_junction_c"] = float(val)
            elif "temperature" in lk and "edge" in lk:
                out["temp_edge_c"] = float(val)
            elif "memory use" in lk or "vram" in lk:
                out["gpu_mem_pct"] = val
            elif "power" in lk:
                out["power_w"] = val
    except Exception as e:  # stats are best-effort
        out["error"] = repr(e)
    return out


def validate_cycle(out_dir: Path) -> dict:
    """Every camera dir must have detections + a playable annotated video
    whose frame count matches the jsonl line count."""
    ok, failures, total_frames = 0, [], 0
    for vdir in sorted(out_dir.glob("*/*/")):
        name = f"{vdir.parent.name}/{vdir.name}"
        jl = vdir / "detections.jsonl"
        mp4 = vdir / "annotated.mp4"
        if not jl.exists() or not mp4.exists():
            failures.append(f"{name}: missing outputs")
            continue
        n_json = sum(1 for _ in open(jl))
        try:
            n_vid = int(subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-count_packets", "-show_entries", "stream=nb_read_packets",
                 "-of", "csv=p=0", str(mp4)],
                capture_output=True, text=True, timeout=120).stdout.strip())
        except Exception as e:
            failures.append(f"{name}: ffprobe failed {e!r}")
            continue
        if n_json == 0:
            failures.append(f"{name}: empty detections")
        elif n_vid != n_json:
            failures.append(f"{name}: video {n_vid} != jsonl {n_json} frames")
        else:
            ok += 1
            total_frames += n_json
    return {"videos_ok": ok, "failures": failures, "frames": total_frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("model_onnx")
    ap.add_argument("work_dir")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=13)
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    summary_path = work / "soak_summary.jsonl"
    deadline = time.time() + args.hours * 3600
    cycle = 0
    print(f"soak until {time.strftime('%F %T', time.localtime(deadline))}",
          flush=True)

    while time.time() < deadline:
        cycle += 1
        if free_gb(str(work)) < MIN_FREE_GB:
            rec = {"cycle": cycle, "aborted": f"free disk < {MIN_FREE_GB} GB"}
            summary_path.open("a").write(json.dumps(rec) + "\n")
            print(rec, flush=True)
            break
        out_dir = work / f"cycle_{cycle:03d}"
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "process_nds_batch.py"),
             args.input_dir, args.model_onnx, str(out_dir),
             "--workers", str(args.workers), "--limit-per-camera", "1"],
            capture_output=True, text=True)
        dt = time.time() - t0
        result = validate_cycle(out_dir)
        rec = {
            "cycle": cycle,
            "started": time.strftime("%F %T", time.localtime(t0)),
            "duration_s": round(dt, 1),
            "returncode": proc.returncode,
            "agg_fps": round(result["frames"] / dt, 1) if dt else 0,
            **result,
            "gpu": gpu_stats(),
            "free_gb": round(free_gb(str(work)), 1),
        }
        if proc.returncode != 0:
            rec["stderr_tail"] = proc.stderr[-500:]
        summary_path.open("a").write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

        # keep one annotated sample from cycle 1, then reclaim the ~20 GB
        if cycle == 1:
            sample = next(iter(sorted(out_dir.glob("*/*/annotated.mp4"))), None)
            if sample:
                shutil.copy(sample, work / "sample_annotated_cycle1.mp4")
        shutil.rmtree(out_dir, ignore_errors=True)

    print("soak complete", flush=True)


if __name__ == "__main__":
    main()
