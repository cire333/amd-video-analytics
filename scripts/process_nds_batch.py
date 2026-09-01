"""Batch-process a directory tree of customer videos in parallel on the AMD GPU.

    PYTHONPATH=/opt/rocm/lib:src .venv/bin/python scripts/process_nds_batch.py \
        <input_dir> <model.onnx> <out_dir> \
        [--workers N] [--limit-per-camera N] [--conf 0.3] [--quant fp16]

Layout assumption: <input_dir>/<camera_id>/**/*.mp4 (camera id = first path
component). For each video, writes
    <out_dir>/<camera_id>/<video-stem>/detections.jsonl   (standard output)
    <out_dir>/<camera_id>/<video-stem>/annotated.mp4      (boxes + track ids)

Workers are separate processes (spawn); the quantized+compiled MIGraphX
program is saved once (.mxr) and loaded by every worker, so only the first
process pays the ~2-minute compile. Model input size is read from the ONNX.
A labels JSON next to the model (<model>.labels.json) provides class names.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".ts", ".mov"}

_worker: dict = {}  # per-process state (model, labels, device)


def video_fps(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames,duration", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip().split(",")
    try:
        frames, dur = float(out[0]), float(out[1])
        return max(1.0, round(frames / dur, 2))
    except (ValueError, IndexError):
        return 25.0


def compile_or_load(onnx_path: str, quant: str, device_ordinal: int):
    """Load the cached compiled program if present, else compile + save."""
    import migraphx
    mxr = f"{onnx_path}.{quant}.mxr"
    if os.path.exists(mxr):
        try:
            prog = migraphx.load(mxr)
            name = prog.get_parameter_names()[0]
            return migraphx, prog, name
        except Exception:
            pass  # stale/incompatible cache: recompile
    prog = migraphx.parse_onnx(onnx_path)
    name = prog.get_parameter_names()[0]
    if quant == "fp16":
        migraphx.quantize_fp16(prog)
    prog.compile(migraphx.get_target("gpu"))
    try:
        migraphx.save(prog, mxr)
    except Exception:
        pass  # cache is an optimization, not a requirement
    return migraphx, prog, name


def init_worker(onnx_path: str, quant: str, conf: float):
    import numpy as np
    from avap.capabilities import probe_devices, require_decode_device

    dev = require_decode_device(probe_devices())
    mgx, prog, input_name = compile_or_load(onnx_path, quant, dev.device_ordinal)
    shape = prog.get_parameter_shapes()[input_name].lens()  # (1, 3, H, W)
    labels_file = onnx_path.rsplit(".onnx", 1)[0] + ".labels.json"
    labels = (json.load(open(labels_file))
              if os.path.exists(labels_file) else [str(i) for i in range(1000)])
    warm = np.ascontiguousarray(np.zeros(shape, dtype=np.float32))
    prog.run({input_name: mgx.argument(warm)})
    _worker.update(mgx=mgx, prog=prog, input_name=input_name, dev=dev,
                   imgsz=shape[2], labels=labels, conf=conf)


def process_video(job: tuple[str, str, str]) -> dict:
    try:
        return _process_video(job)
    except Exception as e:  # isolate: one bad video must not sink the batch
        print(f"[{job[1]}/{Path(job[0]).stem}] FAILED: {e}", flush=True)
        return {"camera": job[1], "video": Path(job[0]).stem,
                "frames": 0, "seconds": 0.0, "error": str(e)}


def _process_video(job: tuple[str, str, str]) -> dict:
    import cv2
    import numpy as np
    from avap import _core
    from avap.bytetrack import ByteTracker
    from avap.frame import ObjectMeta

    video, camera, out_dir = job
    dev, imgsz, conf = _worker["dev"], _worker["imgsz"], _worker["conf"]
    mgx, prog, input_name = _worker["mgx"], _worker["prog"], _worker["input_name"]
    labels = _worker["labels"]

    stem = Path(video).stem
    vdir = Path(out_dir) / camera / stem
    vdir.mkdir(parents=True, exist_ok=True)
    fps = video_fps(video)
    tracker = ByteTracker(iou_threshold=0.3, max_age=int(fps), min_hits=3)

    def color(tid):
        rng = np.random.default_rng(tid or 0)
        return tuple(int(c) for c in rng.integers(60, 255, 3))

    dec = _core.Decoder(video, dev.drm_render_node)
    writer = None
    n = 0
    t0 = time.time()
    try:
        with open(vdir / "detections.jsonl", "w") as jf:
            while True:
                f = dec.next_frame()
                if f is None:
                    break
                planes = [tuple(p) for p in f.planes]
                src = (f.crop_x, f.crop_y, f.crop_w, f.crop_h)
                args = (planes, src, (imgsz, imgsz), f.full_range,
                        f.color_matrix != "bt601", dev.device_ordinal)
                if f.dmabuf_fd >= 0:
                    chw = _core.nv12_dmabuf_to_rgb(
                        f.dmabuf_fd, f.width, f.height, planes,
                        f.drm_modifier, *args[1:])
                else:
                    chw = _core.nv12_host_to_rgb(f.host_data, *args)

                inp = np.ascontiguousarray(chw[None])
                out = np.array(prog.run({input_name: mgx.argument(inp)})[0])[0]
                del inp
                sx, sy = f.crop_w / imgsz, f.crop_h / imgsz
                dets = [ObjectMeta(class_id=int(c), confidence=float(s),
                                   bbox=(float(x1) * sx, float(y1) * sy,
                                         float(x2) * sx, float(y2) * sy),
                                   label=labels[int(c)] if int(c) < len(labels)
                                   else str(int(c)))
                        for x1, y1, x2, y2, s, c in out if s >= conf]
                tracked = tracker.update(dets)

                jf.write(json.dumps({
                    "frame": n, "pts_us": f.pts_us,
                    "objects": [{"track_id": o.track_id, "label": o.label,
                                 "class_id": o.class_id,
                                 "conf": round(o.confidence, 4),
                                 "bbox": [round(v, 1) for v in o.bbox]}
                                for o in tracked]}) + "\n")

                bgr = (np.clip(chw, 0, 1) * 255).astype(np.uint8
                        ).transpose(1, 2, 0)[:, :, ::-1]
                bgr = cv2.resize(bgr, (f.crop_w, f.crop_h))
                for o in tracked:
                    x1, y1, x2, y2 = (int(v) for v in o.bbox)
                    c = color(o.track_id)
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), c, 2)
                    tag = (f"#{o.track_id} " if o.track_id else "") + \
                        f"{o.label} {o.confidence:.2f}"
                    cv2.putText(bgr, tag, (x1, max(12, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(vdir / "annotated_raw.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (bgr.shape[1], bgr.shape[0]))
                writer.write(bgr)
                n += 1
    finally:
        if writer is not None:
            writer.release()
        dec.close()

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(vdir / "annotated_raw.mp4"),
                    "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", str(vdir / "annotated.mp4")],
                   check=True)
    os.remove(vdir / "annotated_raw.mp4")

    dt = time.time() - t0
    print(f"[{camera}/{stem}] {n} frames in {dt:.0f}s ({n / dt:.1f} fps)",
          flush=True)
    return {"camera": camera, "video": stem, "frames": n, "seconds": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("model_onnx")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-per-camera", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--quant", default="fp16", choices=("fp32", "fp16"))
    args = ap.parse_args()

    root = Path(args.input_dir)
    by_camera: dict[str, list[str]] = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in VIDEO_EXTS and p.is_file():
            cam = p.relative_to(root).parts[0]
            by_camera.setdefault(cam, []).append(str(p))
    jobs = []
    for cam, vids in sorted(by_camera.items()):
        for v in vids[:args.limit_per_camera]:
            jobs.append((v, cam, args.out_dir))
    print(f"{len(jobs)} video(s) from {len(by_camera)} camera(s); "
          f"{args.workers} workers", flush=True)

    # Pre-compile the .mxr cache in the parent (a spawn-safe subprocess of
    # its own) so workers all fast-path the load.
    init_worker(args.model_onnx, args.quant, args.conf)

    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(args.workers, initializer=init_worker,
                  initargs=(args.model_onnx, args.quant, args.conf)) as pool:
        results = pool.map(process_video, jobs)
    total = sum(r["frames"] for r in results)
    failed = [r for r in results if r.get("error")]
    dt = time.time() - t0
    print(f"\nDONE: {len(results) - len(failed)}/{len(results)} videos, "
          f"{total} frames in {dt / 60:.1f} min "
          f"(aggregate {total / dt:.1f} fps)", flush=True)
    for r in failed:
        print(f"  failed: {r['camera']}/{r['video']}: {r['error']}", flush=True)


if __name__ == "__main__":
    main()
