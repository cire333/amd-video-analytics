# Environment setup

Target box: Ubuntu 24.04, AMD Radeon AI PRO R9700 (Navi 48, **gfx1201**) +
RTX 3090 (parity reference). gfx1201 requires **ROCm >= 6.4**.

## 1. System packages (sudo — run yourself)

```bash
./scripts/setup_system.sh
```

Then **log out and back in** (render/video group membership), and check:

```bash
./scripts/verify_env.sh
```

## 2. Python environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip scikit-build-core pybind11 numpy pytest
```

## 3. ONNX Runtime

- **AMD box**: AMD publishes ROCm/MIGraphX-enabled ONNX Runtime wheels on
  repo.radeon.com (name/index varies by ROCm release — check
  https://repo.radeon.com/rocm/manylinux/ for the current one matching the
  installed ROCm). The MIGraphX EP is the TensorRT analog.
- **3090 parity runs**: `pip install onnxruntime-gpu` in a separate venv.
- **CPU-only development** (pipeline logic, no GPU): `pip install onnxruntime`.

## 4. Build the extension

```bash
pip install -e . --no-build-isolation
```

Before ROCm is installed you can still build and test the decode path:

```bash
AVAP_WITH_HIP=OFF pip install -e . --no-build-isolation
```

## 5. Smoke test

```bash
pytest                                   # pure-Python layer, no GPU needed
python examples/single_stream.py video.mp4 detector.onnx   # full e2e
```

## Bring-up findings (verified on the R9700, ROCm 7.2.4, Mesa 25.2.8 — 2026-08-31)

1. **Tiled export — CONFIRMED**: radeonsi exports decode surfaces with a GFX12
   swizzle modifier (`0x0200000000082305`), not linear. `AMD_DEBUG=notiling`
   makes export linear but **breaks VCN motion compensation** (reference
   frames must stay natively tiled) — intra frames decode clean, P-frames
   smear. Do not use it. Current handling: the decoder falls back to
   `av_hwframe_transfer_data` (driver-side detile to host) for tiled
   surfaces; the zero-copy dmabuf path activates automatically when a
   surface is linear. Production fix candidates: **rocDecode** (AMD's
   NVDEC-analog, packaged as `rocdecode` in the ROCm repo — owns this
   problem internally) or a GFX12 detile kernel from addrlib.
2. **`hipImportExternalMemory` on a dmabuf — WORKS**: OpaqueFd handle type
   accepts VAAPI dmabuf fds on ROCm 7.2.4 / gfx1201, and
   `hipExternalMemoryGetMappedBuffer` + kernel reads produce pixel-correct
   output (verified against CPU-decoded reference frames).
3. **gfx1201 — WORKS**: visible in rocminfo, kernels compile and run.
4. **Mixed-toolchain gotcha**: pybind11's default LTO (GCC slim-LTO objects)
   is silently discarded by the ROCm clang link step used when a `.hip`
   file is in the target — the module loses `PyInit__core`. CMakeLists uses
   `NO_EXTRAS` to prevent this.
