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

## Known bring-up risks (verify on hardware, in this order)

1. **Tiled export** — if `vaExportSurfaceHandle` returns a non-linear DRM
   modifier on radeonsi, the bridge raises. Options: request linear via
   surface attribs at decode time, or add the detile kernel (architecture
   §5 edge case 1). Decide from what the R9700 actually exports.
2. **`hipImportExternalMemory` on a dmabuf** — the OpaqueFd handle type is
   expected to accept dmabuf fds on ROCm/Linux, but this is the least
   documented link in the chain. If it rejects the fd, fallbacks:
   `hipExtImportBuffer`-style HSA interop, or (slow but correct) vaDeriveImage
   + host copy to keep e2e moving while we investigate.
3. **gfx1201 in the ROCm release actually installed** — `rocminfo | grep gfx`
   must show gfx1201; if not, pin a newer ROCm from repo.radeon.com.
