"""Device-resident frames and models for the GPU SGIE crop path.

DeviceFrame: a full-resolution CHW float RGB frame living in GPU memory —
produced directly by the decode bridge (no D2H) or uploaded once from a
host image. DeviceModel: a MIGraphX program compiled offload_copy=False
whose pre-allocated device input buffer is filled by the HIP crop+resize
kernel, so per-object secondary inference costs no PCIe frame traffic —
only the model's (tiny) outputs are copied back.
"""
from __future__ import annotations

import numpy as np


def device_api_available() -> bool:
    from .. import _core
    return hasattr(_core, "rgb_crop_resize_device")


class DeviceFrame:
    """CHW float32 RGB frame in device memory."""

    def __init__(self, handle: int, width: int, height: int,
                 device_ordinal: int, _owner=None):
        self.handle = handle
        self.width = width
        self.height = height
        self.device_ordinal = device_ordinal
        self._owner = _owner   # None => we own via free_device_buffer

    @classmethod
    def from_decoded(cls, f, device_ordinal: int) -> "DeviceFrame":
        """Convert a decoded NV12 frame to device RGB without touching host."""
        from .. import _core
        planes = [tuple(p) for p in f.planes]
        src = (f.crop_x, f.crop_y, f.crop_w, f.crop_h)
        args = (planes, src, (f.crop_w, f.crop_h), f.full_range,
                f.color_matrix != "bt601", device_ordinal)
        if f.dmabuf_fd >= 0:
            handle = _core.nv12_dmabuf_to_device_rgb(
                f.dmabuf_fd, f.width, f.height, planes, f.drm_modifier,
                *args[1:])
        else:
            handle = _core.nv12_host_to_device_rgb(f.host_data, *args)
        return cls(handle, f.crop_w, f.crop_h, device_ordinal)

    @classmethod
    def from_host(cls, rgb_hwc: np.ndarray, device_ordinal: int) -> "DeviceFrame":
        """Upload a host HWC uint8 RGB image (one H2D copy). The backing
        memory is a MIGraphX device argument kept alive by this object."""
        import migraphx
        chw = np.ascontiguousarray(
            rgb_hwc.astype(np.float32).transpose(2, 0, 1) / 255.0)
        arg = migraphx.to_gpu(migraphx.argument(chw))
        return cls(arg.data_ptr(), rgb_hwc.shape[1], rgb_hwc.shape[0],
                   device_ordinal, _owner=arg)

    def to_host(self) -> np.ndarray:
        """One D2H copy -> HWC uint8 RGB (for probes/annotation)."""
        from .. import _core
        chw = _core.device_rgb_to_host(self.handle, self.width, self.height)
        return (np.clip(chw, 0.0, 1.0) * 255).astype(np.uint8).transpose(1, 2, 0)

    def close(self) -> None:
        if self.handle and self._owner is None:
            from .. import _core
            _core.free_device_buffer(self.handle)
        self.handle = 0
        self._owner = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class DeviceModel:
    """MIGraphX program with device-resident I/O for crop-based inference.

    The input buffer is allocated once; run_crop() fills it with the HIP
    crop+resize kernel straight from a DeviceFrame and runs the program.
    Only outputs are copied to host. Not thread-safe (buffers are reused).
    """

    OUTPUT_PREFIX = "main:#output_"

    def __init__(self, onnx_path: str, quant: str = "fp16",
                 device_ordinal: int = 0):
        import migraphx
        self._mgx = migraphx
        self.device_ordinal = device_ordinal
        prog = migraphx.parse_onnx(onnx_path)
        if quant == "fp16":
            migraphx.quantize_fp16(prog)
        elif quant != "fp32":
            raise ValueError("DeviceModel supports fp32/fp16")
        prog.compile(migraphx.get_target("gpu"), offload_copy=False)
        self._prog = prog

        params = prog.get_parameter_names()
        outs = sorted(p for p in params if p.startswith(self.OUTPUT_PREFIX))
        ins = [p for p in params if not p.startswith(self.OUTPUT_PREFIX)]
        if len(ins) != 1:
            raise ValueError(f"{onnx_path}: expected one input, got {ins}")
        shapes = prog.get_parameter_shapes()
        self._in_name = ins[0]
        self.input_shape = tuple(shapes[self._in_name].lens())  # (1,3,H,W)
        self._in_arg = migraphx.allocate_gpu(shapes[self._in_name])
        self._in_ptr = self._in_arg.data_ptr()
        self._out_names = outs
        self._out_args = [migraphx.allocate_gpu(shapes[o]) for o in outs]
        self._args = {self._in_name: self._in_arg,
                      **dict(zip(outs, self._out_args))}
        self.run_full = None  # populated lazily if ever needed

        # warmup
        prog.run(self._args)
        migraphx.gpu_sync()

    def run_crop(self, frame: DeviceFrame, crop: tuple[int, int, int, int]
                 ) -> np.ndarray | list[np.ndarray]:
        """crop = (x, y, w, h) in frame pixels; filled into the model input
        on-GPU, inference run, outputs copied to host."""
        from .. import _core
        h, w = self.input_shape[2], self.input_shape[3]
        _core.rgb_crop_resize_device(frame.handle, frame.width, frame.height,
                                     crop, self._in_ptr, w, h,
                                     self.device_ordinal)
        res = self._prog.run(self._args)
        if self._out_args:
            outs = [np.array(self._mgx.from_gpu(a)) for a in self._out_args]
        else:
            # fully const-folded program: no output parameters exist; run()
            # returns the outputs itself (device or host arguments)
            outs = []
            for a in res:
                try:
                    outs.append(np.array(self._mgx.from_gpu(a)))
                except Exception:
                    outs.append(np.array(a))
        return outs[0] if len(outs) == 1 else outs
