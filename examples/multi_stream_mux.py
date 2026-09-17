"""DeepStream-shaped multi-stream run on the R9700: N files -> StreamMux -> canvas -> batched YOLO.

    PYTHONPATH=/opt/rocm/lib python examples/multi_stream_mux.py \
        --model ~/.cache/avap/models/yolo26m_b4_640.onnx --batch-size 4 \
        results/ds_work/sample_1080p_h264.mp4 results/ds_work/sample_720p.mp4 \
        results/ds_work/sample_qHD.mp4 results/ds_work/1933_A22.mp4

Mirrors deepstream-app's [streammux] group: batch-size, batched-push-timeout,
width/height, enable-padding, live-source. Prints the batch composition
histogram (compare with docs/nvstreammux_reverse_engineering.md) and
throughput; --jsonl writes per-frame detections with the NvDsFrameMeta fields.
"""
import argparse
import collections
import json
import logging
import os
import sys
import time

from avap.model_zoo import MigraphxModel
from avap.muxed_pipeline import MuxedPipeline
from avap.streammux import MuxConfig
from avap.streaming_labels import COCO_LABELS
from avap.nvdcf import NvDcfConfig, NvDcfTracker
from avap.ocsort import OcSortTracker
from avap.tracker import TrackerBank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--model", required=True, help="batched ONNX (batch dim == --batch-size)")
    ap.add_argument("--quant", default="fp16", choices=["fp32", "fp16"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--timeout-us", type=int, default=40000)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--padding", type=int, default=1)
    ap.add_argument("--live", type=int, default=0)
    ap.add_argument("--policy", default="legacy", choices=["legacy", "new"])
    ap.add_argument("--pace-fps", type=float, default=None,
                    help="throttle each decoder to emulate live cameras")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--track", action="store_true", help="OC-SORT (motion only)")
    ap.add_argument("--nvdcf", action="store_true", help="NvDCF-class visual tracker on the canvas")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--max-seconds", type=float, default=120)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    model = MigraphxModel(args.model, args.quant)
    assert model.input_shape[0] == args.batch_size, \
        f"model batch {model.input_shape[0]} != --batch-size {args.batch_size}"
    cfg = MuxConfig(batch_size=args.batch_size, batched_push_timeout_us=args.timeout_us,
                    live_source=bool(args.live), policy=args.policy,
                    width=args.width, height=args.height, enable_padding=bool(args.padding))

    out = open(args.jsonl, "w") if args.jsonl else None
    per_source = collections.Counter()
    objs_per_source = collections.Counter()

    def sink(fm):
        per_source[fm.source_id] += 1
        objs_per_source[fm.source_id] += len(fm.objects)
        if out:
            out.write(json.dumps({
                "source_id": fm.source_id, "pad_index": fm.mux.pad_index,
                "frame_num": fm.mux.frame_num, "batch_index": fm.batch_index,
                "batch_id": fm.mux.batch_id, "buf_pts_us": fm.mux.buf_pts // 1000,
                "ntp": fm.mux.ntp_timestamp,
                "source_frame": [fm.mux.source_frame_width, fm.mux.source_frame_height],
                "objects": [{"label": o.label, "conf": round(o.confidence, 3),
                             "bbox": [round(v, 1) for v in o.bbox],
                             "track_id": o.track_id} for o in fm.objects]}) + "\n")

    pipe = MuxedPipeline(model, cfg, sink, conf_threshold=args.conf,
                         tracker_bank=(TrackerBank(lambda: NvDcfTracker(NvDcfConfig(checkClassMatch=0), backend="hip"))
                                       if args.nvdcf else
                                       TrackerBank(lambda: OcSortTracker(0.25, max_age=90, min_hits=5))
                                       if args.track else None),
                         pace_fps=args.pace_fps, labels=COCO_LABELS)
    comps = collections.Counter()
    sizes = collections.Counter()
    orig_on_batch = pipe._on_batch

    def counting_on_batch(batch):
        comps[batch.pad_indices()] += 1
        sizes[len(batch.frames)] += 1
        orig_on_batch(batch)
    pipe._runner.on_batch = counting_on_batch

    for i, uri in enumerate(args.sources):
        pipe.add_source(f"src{i}:{os.path.basename(uri)}", uri)
    t0 = time.monotonic()
    pipe.start()
    finished = pipe.wait(args.max_seconds)
    pipe.stop()
    if out:
        out.close()
    dt = time.monotonic() - t0
    st = pipe.stats
    print("\n=== StreamMux run", "(EOS reached)" if finished else "(timed out)")
    print(f"sources: {len(args.sources)}  policy={args.policy} live={args.live} "
          f"bs={args.batch_size} timeout={args.timeout_us}us canvas={args.width}x{args.height} pad={args.padding}")
    for pad, s in pipe.sources.items():
        print(f"  pad {pad} {s.source_id}: decoded {s.frames}, emitted {per_source[s.source_id]}, "
              f"objects {objs_per_source[s.source_id]}, mux drops {pipe.mux.pads[pad].dropped}"
              + (f", ERROR {s.error}" if s.error else ""))
    print(f"batches: {st['batches']}  frames: {st['frames']}  wall {dt:.1f}s -> "
          f"{st['frames'] / dt:.1f} frames/s, {st['batches'] / dt:.1f} batches/s")
    if st["batches"]:
        print(f"per batch: canvas+D2H {1e3 * st['canvas_s'] / st['batches']:.2f} ms, "
              f"inference {1e3 * st['infer_s'] / st['batches']:.2f} ms")
    print("batch size histogram:", dict(sorted(sizes.items())))
    print("top compositions (pad order):", comps.most_common(6))
    print("events:", collections.Counter(e.kind for e in pipe.events))
    return 0 if finished else 1


if __name__ == "__main__":
    sys.exit(main())
