"""Off-the-shelf models + quantized MIGraphX compilation.

Model zoo names auto-download and export to ONNX on first use (cached in
~/.cache/avap/models). A path to a custom .onnx is accepted anywhere a zoo
name is. Quantization: fp32, fp16 (quantize_fp16), int8 (quantize_int8
with calibration tensors collected from the live stream).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL_ZOO = {
    # Ultralytics end-to-end (NMS-free) detectors; output (N, 300, 6).
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
}
QUANT_MODES = ("fp32", "fp16", "int8")
INT8_CALIBRATION_FRAMES = 32

_CACHE = Path(os.environ.get("AVAP_MODEL_CACHE",
                             Path.home() / ".cache" / "avap" / "models"))
_MXR_CACHE = Path(os.environ.get("AVAP_MXR_CACHE",
                                 Path.home() / ".cache" / "avap" / "mxr"))


def _cache_key(onnx_path: str, quant: str, offload_copy: bool) -> str:
    """Cache identity for a compiled program: the exact model file, the
    quantization, the parameter convention, the GPU ISA, and the ROCm
    install (compiled code is arch- and version-specific)."""
    import hashlib
    from .capabilities import probe_devices
    st = os.stat(onnx_path)
    archs = ",".join(sorted({d.gcn_arch for d in probe_devices()})) or "unknown"
    rocm = os.path.realpath("/opt/rocm")
    raw = "|".join([os.path.abspath(onnx_path), str(st.st_size),
                    str(int(st.st_mtime)), quant, str(offload_copy),
                    archs, rocm])
    return hashlib.sha1(raw.encode()).hexdigest()


def compile_or_load(onnx_path: str, quant: str, offload_copy: bool = True):
    """Compile an ONNX with MIGraphX, or load the cached compiled program
    (.mxr) — turning ~2 min of startup into a sub-second load. The cache is
    best-effort: any load/save problem falls back to a fresh compile.
    Disable with AVAP_NO_MXR_CACHE=1. int8 is never cached (its compilation
    depends on stream-collected calibration data)."""
    import migraphx
    use_cache = (quant != "int8"
                 and os.environ.get("AVAP_NO_MXR_CACHE") != "1")
    mxr = None
    if use_cache:
        mxr = _MXR_CACHE / f"{_cache_key(onnx_path, quant, offload_copy)}.mxr"
        if mxr.exists():
            try:
                return migraphx.load(str(mxr)), True
            except Exception:
                pass  # stale/corrupt cache: recompile below

    prog = migraphx.parse_onnx(onnx_path)
    if quant == "fp16":
        migraphx.quantize_fp16(prog)
    prog.compile(migraphx.get_target("gpu"), offload_copy=offload_copy)
    if mxr is not None:
        try:
            _MXR_CACHE.mkdir(parents=True, exist_ok=True)
            tmp = mxr.with_suffix(".tmp")   # atomic vs concurrent workers
            migraphx.save(prog, str(tmp))
            os.replace(tmp, mxr)
        except Exception:
            pass
    return prog, False


def resolve_model(model: str, batch_size: int = 1, imgsz: int = 640) -> str:
    """Zoo name -> cached ONNX path (downloading/exporting on first use);
    a filesystem path to an .onnx is passed through."""
    if model.endswith(".onnx"):
        if not os.path.exists(model):
            raise FileNotFoundError(f"custom model not found: {model}")
        return model
    if model not in MODEL_ZOO:
        raise ValueError(f"unknown model {model!r}; zoo: {sorted(MODEL_ZOO)} "
                         "or pass a path to a custom .onnx")
    _CACHE.mkdir(parents=True, exist_ok=True)
    out = _CACHE / f"{model}_b{batch_size}_{imgsz}.onnx"
    if out.exists():
        return str(out)
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "the model zoo needs `pip install ultralytics` (one-time export); "
            "alternatively pass a path to an already-exported .onnx") from e
    exported = YOLO(f"{model}.pt").export(format="onnx", imgsz=imgsz,
                                          batch=batch_size, dynamic=False,
                                          simplify=True)
    os.replace(exported, out)
    return str(out)


class MigraphxModel:
    """Compiled MIGraphX program with quantization and safe buffer lifetime.

    INT8 defers compilation until `calibrate()` has been fed
    INT8_CALIBRATION_FRAMES preprocessed tensors from the live stream.
    """

    def __init__(self, onnx_path: str, quant: str = "fp16", device_ordinal: int = 0):
        if quant not in QUANT_MODES:
            raise ValueError(f"model_quant must be one of {QUANT_MODES}")
        import migraphx
        self._mgx = migraphx
        self.quant = quant
        self.device_ordinal = device_ordinal
        self._calib: list[np.ndarray] = []
        self._compiled = False
        if quant != "int8":
            self._prog, self.loaded_from_cache = compile_or_load(
                onnx_path, quant, offload_copy=True)
            self.input_name = self._prog.get_parameter_names()[0]
            self.input_shape = self._prog.get_parameter_shapes()[
                self.input_name].lens()
            self._warmup()
            self._compiled = True
        else:
            self.loaded_from_cache = False
            self._prog = migraphx.parse_onnx(onnx_path)
            self.input_name = self._prog.get_parameter_names()[0]
            self.input_shape = self._prog.get_parameter_shapes()[
                self.input_name].lens()

    @property
    def ready(self) -> bool:
        return self._compiled

    def calibrate(self, tensor: np.ndarray) -> bool:
        """Feed one preprocessed batch tensor; compiles when enough have
        been collected. Returns True once the model is ready."""
        if self._compiled:
            return True
        self._calib.append(np.ascontiguousarray(tensor))
        if len(self._calib) >= INT8_CALIBRATION_FRAMES:
            self._compile()
        return self._compiled

    def _compile(self) -> None:
        # int8 path only (fp32/fp16 go through compile_or_load in __init__)
        target = self._mgx.get_target("gpu")
        data = [{self.input_name: self._mgx.argument(t)} for t in self._calib]
        self._mgx.quantize_int8(self._prog, target, calibration=data)
        self._calib.clear()
        self._prog.compile(target)
        self._warmup()
        self._compiled = True

    def _warmup(self) -> None:
        # first-run sanity: warm up and keep the input buffer alive through
        # run() — migraphx.argument borrows the numpy buffer (no copy)
        warm = np.ascontiguousarray(
            np.zeros(self.input_shape, dtype=np.float32))
        self._prog.run({self.input_name: self._mgx.argument(warm)})

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if not self._compiled:
            raise RuntimeError("int8 model not calibrated yet")
        arr = np.ascontiguousarray(batch, dtype=np.float32)
        out = self._prog.run({self.input_name: self._mgx.argument(arr)})
        result = np.array(out[0])
        del arr  # keep alive until after run
        return result
