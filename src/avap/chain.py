"""ModelChain — daisy-chained MIGraphX models with device-resident handoff.

The DeepStream/NVMM analog: a frame is uploaded to the GPU once, flows
through every model in the chain as a device tensor, and only the final
output is copied back. Interior hops cost ~0 transfer (vs a full PCIe
round trip per hop with independent models).

Mechanics: each model is compiled with offload_copy=False, so its
parameters are device arguments. Output buffers are pre-allocated once and
reused every frame; model i's output buffer is passed directly as model
i+1's input. Adjacent shapes are validated at build time.

    chain = ModelChain(["enhance.onnx", "denoise.onnx", "yolo26m"],
                       quant="fp16")
    out = chain(frame_chw[None])    # one H2D, N inferences, one D2H

Not thread-safe (buffers are reused): use one chain per worker thread, or
serialize calls — same rule as a single MigraphxModel.
"""
from __future__ import annotations

import numpy as np

from .model_zoo import QUANT_MODES, resolve_model

OUTPUT_PREFIX = "main:#output_"


class ModelChain:
    def __init__(self, models: list[str], quant: str = "fp16",
                 device_ordinal: int = 0, batch_size: int = 1):
        if len(models) < 1:
            raise ValueError("ModelChain needs at least one model")
        if quant not in QUANT_MODES:
            raise ValueError(f"quant must be one of {QUANT_MODES}")
        if quant == "int8":
            raise ValueError("int8 chains are not supported yet (calibration "
                             "per hop); use fp16 or fp32")
        import migraphx
        self._mgx = migraphx
        self.device_ordinal = device_ordinal
        self.quant = quant
        self.ready = True  # duck-types MigraphxModel for the stream loop

        self._progs = []
        self._in_names: list[str] = []
        self._out_names: list[list[str]] = []
        shapes: list[dict] = []
        for m in models:
            onnx = m if m.endswith(".onnx") else resolve_model(m, batch_size)
            prog = migraphx.parse_onnx(onnx)
            if quant == "fp16":
                migraphx.quantize_fp16(prog)
            prog.compile(migraphx.get_target("gpu"), offload_copy=False)
            params = prog.get_parameter_names()
            outs = sorted(p for p in params if p.startswith(OUTPUT_PREFIX))
            ins = [p for p in params if not p.startswith(OUTPUT_PREFIX)]
            if len(ins) != 1:
                raise ValueError(f"{m}: chain models must have exactly one "
                                 f"input, got {ins}")
            self._progs.append(prog)
            self._in_names.append(ins[0])
            self._out_names.append(outs)
            shapes.append(prog.get_parameter_shapes())

        # interior models must be single-output, and each hop's shape must
        # match the next model's input
        for i in range(len(models) - 1):
            if len(self._out_names[i]) != 1:
                raise ValueError(
                    f"{models[i]}: interior chain models must have exactly "
                    f"one output, got {len(self._out_names[i])}")
            out_lens = shapes[i][self._out_names[i][0]].lens()
            in_lens = shapes[i + 1][self._in_names[i + 1]].lens()
            if list(out_lens) != list(in_lens):
                raise ValueError(
                    f"shape mismatch at hop {i}: {models[i]} outputs "
                    f"{list(out_lens)} but {models[i + 1]} expects "
                    f"{list(in_lens)}")

        self.input_shape = tuple(shapes[0][self._in_names[0]].lens())
        # pre-allocate every hop's output buffer once (reused per frame)
        self._buffers = [
            [migraphx.allocate_gpu(shapes[i][o]) for o in self._out_names[i]]
            for i in range(len(models))
        ]
        # warmup (also flushes any first-run issues)
        self(np.zeros(self.input_shape, dtype=np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray | list[np.ndarray]:
        """One H2D upload, chained device-resident inference, one D2H copy.
        Returns the final model's output (list if it has several)."""
        mgx = self._mgx
        arr = np.ascontiguousarray(x, dtype=np.float32)
        cur = mgx.to_gpu(mgx.argument(arr))
        for i, prog in enumerate(self._progs):
            args = {self._in_names[i]: cur}
            for name, buf in zip(self._out_names[i], self._buffers[i]):
                args[name] = buf
            prog.run(args)
            cur = self._buffers[i][0]
        del arr  # keep host buffer alive through the (synchronous) run
        finals = [np.array(mgx.from_gpu(b)) for b in self._buffers[-1]]
        return finals[0] if len(finals) == 1 else finals

    def __len__(self) -> int:
        return len(self._progs)
