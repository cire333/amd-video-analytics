"""VCN hardware video encoding — the NVENC analog.

    from avap import VideoEncoder
    enc = VideoEncoder("out.mp4", 1920, 1080, fps=25)        # or rtsp://...
    enc.write(rgb_hwc_uint8)          # host frame (annotated, etc.)
    enc.write_device(device_frame)    # advanced-API DeviceFrame: RGB->NV12
                                      # on-GPU, only NV12 (1.5 B/px) crosses PCIe
    enc.close()

Codecs: h264 (default), hevc, av1 — all VCN-accelerated through VAAPI.
Containers by extension (mp4/mkv/...); rtsp:// pushes to an RTSP server.
Replaces the cv2/x264 software path that limited annotated output to ~10
fps per worker.
"""
from __future__ import annotations

import numpy as np

from .capabilities import probe_devices, require_decode_device


class VideoEncoder:
    def __init__(self, output: str, width: int, height: int, fps: float = 25.0,
                 codec: str = "h264", bitrate: int = 8_000_000,
                 render_node: str | None = None, device_ordinal: int | None = None,
                 full_range: bool = False, bt709: bool = True):
        if width % 2 or height % 2:
            raise ValueError("encoder needs even width/height")
        from . import _core
        if render_node is None or device_ordinal is None:
            dev = require_decode_device(probe_devices())
            render_node = render_node or dev.drm_render_node
            device_ordinal = (dev.device_ordinal if device_ordinal is None
                              else device_ordinal)
        self.width, self.height = width, height
        self.device_ordinal = device_ordinal
        self.full_range, self.bt709 = full_range, bt709
        self.frames_written = 0
        self._core = _core
        self._enc = _core.VcnEncoder(output, codec, width, height, float(fps),
                                     int(bitrate), render_node)

    # -- host frames -----------------------------------------------------------

    def write(self, rgb_hwc: np.ndarray) -> None:
        """Encode one HWC uint8 RGB frame (converted to NV12 on CPU)."""
        if rgb_hwc.shape[:2] != (self.height, self.width):
            raise ValueError(f"frame is {rgb_hwc.shape[1]}x{rgb_hwc.shape[0]}, "
                             f"encoder is {self.width}x{self.height}")
        self._enc.write_nv12(self._rgb_to_nv12_cpu(rgb_hwc))
        self.frames_written += 1

    def write_bgr(self, bgr_hwc: np.ndarray) -> None:
        """Convenience for cv2-produced frames."""
        self.write(bgr_hwc[:, :, ::-1])

    # -- device frames (advanced API) ------------------------------------------

    def write_device(self, device_frame) -> None:
        """Encode an avap.advanced DeviceFrame: RGB->NV12 conversion runs as
        a HIP kernel on the device; only the NV12 bytes cross PCIe."""
        if (device_frame.width, device_frame.height) != (self.width, self.height):
            raise ValueError("DeviceFrame dimensions do not match encoder")
        nv12 = self._core.device_rgb_to_nv12_host(
            device_frame.handle, self.width, self.height,
            self.full_range, self.bt709, self.device_ordinal)
        self._enc.write_nv12(nv12)
        self.frames_written += 1

    def close(self) -> None:
        self._enc.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- helpers ----------------------------------------------------------------

    def _rgb_to_nv12_cpu(self, rgb: np.ndarray) -> bytes:
        # cv2's I420 conversion is BT.601 limited-range; the device path
        # (write_device) honors the bt709/full_range flags exactly.
        import cv2
        h, w = self.height, self.width
        i420 = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
        y = i420[:h]
        u = i420[h:h + h // 4].reshape(h // 2, w // 2)
        v = i420[h + h // 4:].reshape(h // 2, w // 2)
        uv = np.empty((h // 2, w), dtype=np.uint8)
        uv[:, 0::2] = u
        uv[:, 1::2] = v
        return np.vstack([y, uv]).tobytes()
