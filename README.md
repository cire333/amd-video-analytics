# AMD Video Analytics Pipeline (avap)

A DeepStream-equivalent multi-stream video analytics pipeline for AMD
hardware: VAAPI/VCN decode → dmabuf→HIP bridge → dynamic batching → ONNX
Runtime (MIGraphX) inference → per-source tracking.

Design rationale and the full architecture live in
[docs/architecture.md](docs/architecture.md). Environment bring-up:
[docs/SETUP.md](docs/SETUP.md).

## Stack decisions

- **Python orchestration + small pybind11/C++ extension** (`avap._core`) for
  the parts that need privileged access: FFmpeg→VAAPI decode, dmabuf export,
  HIP external-memory import, fused NV12→RGB/crop/resize kernel. Everything
  DeepStream does in C GStreamer plugins that *isn't* memory magic — registry,
  batcher, model graph, tracker — is plain Python.
- **ONNX Runtime + MIGraphX EP** as the TensorRT analog; the same ONNX file
  runs on the 3090 (CUDA EP) for parity verification.
- **V1 milestone**: one stream end-to-end on the R9700 before multi-stream.

## Layout

```
src/avap/          Python package
  capabilities.py    device probe (decode presence, gfx generation, VRAM)
  registry.py        mutable stream registry (hot add/remove)
  decoder.py         per-stream worker + reconnect backoff
  ringbuffer.py      bounded queue, drop-oldest, explicit fd close
  batcher.py         timestamp-windowed variable-shape batching
  bridge.py          dmabuf->HIP wrapper (device-mismatch checks, ROI fusion)
  roi.py             normalized polygon ROIs, mask cache, un-projection
  graph.py           model DAG + ORT executor (MIGraphX/ROCm/CUDA/CPU EPs)
  tracker.py         per-source tracker bank (plug in the real Kalman tracker)
  pipeline.py        v1 orchestrator
cpp/               native extension (avap._core)
  vaapi_decoder.*    FFmpeg demux -> VCN decode -> vaExportSurfaceHandle
  hip_bridge.*       hipImportExternalMemory + kernel dispatch
  kernels.hip        fused NV12->RGB + crop + bilinear resize
tests/             pure-Python tests (no GPU required)
examples/          single_stream.py — the v1 milestone runner
scripts/           setup_system.sh (sudo), verify_env.sh
```

## Status / roadmap

- [x] Architecture (docs/architecture.md)
- [x] Pure-Python pipeline layer + tests
- [x] C++ extension written (decode + bridge + kernel)
- [x] Environment: ROCm 7.2.4 installed, gfx1201 visible (scripts/setup_system.sh)
- [x] Extension builds (VAAPI decode + HIP bridge + kernel)
- [x] Bridge verified on R9700: dmabuf→HIP import works; decode surfaces are
      tiled (GFX12 modifier), handled via driver-detile host fallback —
      pixel-correct vs CPU reference (see docs/SETUP.md findings)
- [ ] Zero-copy decode path: evaluate rocDecode vs GFX12 detile kernel
- [ ] V1: single stream e2e with a real detector; 3090 parity comparison
- [x] nvstreammux reverse-engineered on the 3090 and ported: `avap.streammux`
      (legacy + new batching policies, NvDsFrameMeta-equivalent metadata),
      `avap.canvas` (fused NV12→RGB→letterbox batch tensor on the GPU),
      `avap.muxed_pipeline` + `examples/multi_stream_mux.py` — 4 streams,
      batched YOLO, 138 fps on the R9700 (docs/nvstreammux_reverse_engineering.md)
- [x] NvDCF-class visual tracker: `avap.nvdcf` — per-target discriminative
      correlation filters (gray + colour-name + gradient channels) on HIP/hipFFT,
      NvDCF's association (visual x IoU x size) and shadow tracking, DeepStream
      yml-compatible config; reads pixels straight from the mux canvas
      (docs/nvdcf_tracker.md)
- [ ] Multi-stream hot add/remove under load; fd-leak soak test
- [ ] Zero-copy inference input (ORT IOBinding / DLPack), HIP-stream overlap
- [ ] Per-GFX-gen tuning; detile kernel if linear export profiles badly

## Streaming API

```python
from avap import AMDStream, AMDGPUManager

stream = AMDStream(
    data_location="rtsp://cam/live",       # local path, s3://, rtsp://, http(s)://
    region_of_interest=(0.0, 0.3, 1.0, 0.9),  # bbox or polygon, normalized;
                                              # fused into the GPU conversion kernel
    model="yolo26m",                       # zoo (yolo26n/s/m/l/x) or custom .onnx
    model_quant="fp16",                    # fp32 | fp16 | int8 (stream-calibrated)
    tracker_type="bytetrack",              # iou | sort | bytetrack | BYO object
    output_location="kafka://broker:9092/dets",  # kafka:// kinesis:// s3:// sqs:// or file
    batch_size=1,                          # 1 = realtime; >1 = batched inference
    output_format="json",                  # json | csv | parquet
    output_format_template=None,           # optional per-record str.format
    frame_sample_rate=10,                  # max fps sampled from the source
)
stream.start_stream()

mgr = AMDGPUManager(device_id=0)
mgr.add_stream(stream)
mgr.start_streams()   # sequential; a failed start is logged and isolated
```

## Model daisy-chaining (device-resident)

`ModelChain` runs N models back-to-back with the frame staying in GPU
memory between hops (the DeepStream/NVMM analog): one upload, N inferences
against pre-allocated device buffers, one download. Adjacent shapes are
validated at build time. Measured on the R9700 (fp16): a 5-model chain at
640x640 drops 6.7 -> 2.2 ms/frame; a 4-model chain at 1280x1280 drops
32.8 -> 9.8 ms/frame (70% less latency). Reproduce with
`scripts/bench_chain.py`.

```python
from avap import ModelChain
chain = ModelChain(["enhance.onnx", "denoise.onnx", "detector.onnx"],
                   quant="fp16")
out = chain(frame_chw[None])

# or directly in the light API — a list of models becomes a chain:
AMDStream(..., model=["enhance.onnx", "yolo26m"], model_quant="fp16")
```

### Retry & error reporting

```python
from avap import AMDStream, AMDGPUManager, RetryPolicy

def report(event):     # -> your logger / metrics / alerting
    print(event.type, event.source_id, event.attempt, event.error)

stream = AMDStream(
    "rtsp://cam/live", ...,
    retry_policy=RetryPolicy(max_retries=10,       # None = retry forever
                             initial_backoff_s=1, max_backoff_s=60,
                             backoff_multiplier=2, reset_after_s=30),
    on_event=report,   # connecting/connected/disconnected/reconnecting/
)                      # gave_up/eof/sink_error

mgr = AMDGPUManager(config={"on_event": report,          # fleet-wide defaults
                            "retry_policy": RetryPolicy(max_retries=None)})
```

Live sources enter the retry cycle even if down at startup; files fail
fast. Network I/O is deadline-bounded in the decoder (15 s connect, 30 s
read stall), `stop_stream()` aborts a blocked read immediately, sink
failures drop the record and emit `sink_error` instead of stalling decode,
and `mgr.status()` reports restarts / sink_errors / last_error per stream.

## Advanced API

`avap.advanced` — the integrated pipeline layer reverse-engineered from the
DeepStream ingestion system: PGIE/SGIE-style secondary inference on object
crops (child detections, classifiers, embedding stages), hierarchical
object metadata, named probe points, the LPR cascade (plate detect -> OCR
-> DINOv2 re-ID with plate voting and persistent vehicle re-identification),
wire-compatible DataRecord publishing with per-source batching, and
per-source FPS metrics. See [docs/advanced_api.md](docs/advanced_api.md).
