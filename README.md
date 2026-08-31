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
- [ ] Environment: ROCm ≥ 6.4 install, render group (scripts/setup_system.sh)
- [ ] Extension builds; decode-only smoke test (AVAP_WITH_HIP=OFF)
- [ ] Bridge verified on R9700: linear vs tiled export, dmabuf import (SETUP.md risks)
- [ ] V1: single stream e2e with a real detector; 3090 parity comparison
- [ ] Multi-stream + hot add/remove under load; fd-leak soak test
- [ ] Zero-copy inference input (ORT IOBinding / DLPack), HIP-stream overlap
- [ ] Per-GFX-gen tuning; detile kernel if linear export profiles badly
