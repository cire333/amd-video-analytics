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
        self._prog = migraphx.parse_onnx(onnx_path)
        self.input_name = self._prog.get_parameter_names()[0]
        self.input_shape = self._prog.get_parameter_shapes()[self.input_name].lens()
        self._calib: list[np.ndarray] = []
        self._compiled = False
        if quant != "int8":
            self._compile()

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
        target = self._mgx.get_target("gpu")
        if self.quant == "fp16":
            self._mgx.quantize_fp16(self._prog)
        elif self.quant == "int8":
            data = [{self.input_name: self._mgx.argument(t)} for t in self._calib]
            self._mgx.quantize_int8(self._prog, target, calibration=data)
            self._calib.clear()
        self._prog.compile(target)
        # first-run sanity: warm up and keep the input buffer alive through
        # run() — migraphx.argument borrows the numpy buffer (no copy)
        warm = np.ascontiguousarray(
            np.zeros(self.input_shape, dtype=np.float32))
        self._prog.run({self.input_name: self._mgx.argument(warm)})
        self._compiled = True

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if not self._compiled:
            raise RuntimeError("int8 model not calibrated yet")
        arr = np.ascontiguousarray(batch, dtype=np.float32)
        out = self._prog.run({self.input_name: self._mgx.argument(arr)})
        result = np.array(out[0])
        del arr  # keep alive until after run
        return result
