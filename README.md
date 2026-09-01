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
- [ ] Multi-stream + hot add/remove under load; fd-leak soak test
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
