"""GPU bridge wrapper (architecture §5): dmabuf -> HIP -> RGB tensor.

The C++ extension imports the dmabuf as HIP external memory and runs one
fused kernel: NV12 -> RGB + ROI crop + resize, honoring per-plane
offset/pitch and color range/matrix. The full RGB frame is never
materialized.

V1 NOTE: the extension currently copies the result back to host (numpy)
and ORT re-uploads it. Correct first; the zero-copy path (ORT IOBinding /
DLPack on the device tensor) is a planned optimization once e2e works.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame import ColorMatrix, ColorRange, RawFrame
from .roi import RoiConfig, RoiTransform


class DeviceMismatchError(RuntimeError):
    """dmabuf is tied to a DRM render node; importing into a HIP context on
    a different device fails or silently binds wrong. Fail loud instead."""


@dataclass
class ConvertedFrame:
    tensor: np.ndarray          # float32 CHW RGB, normalized 0-1, model input size
    transform: RoiTransform     # maps model-input coords back to full frame
    frame_width: int            # full-frame (crop rect) dims
    frame_height: int


class HipBridge:
    def __init__(self, device_ordinal: int):
        from . import _core
        self._core = _core
        self.device_ordinal = device_ordinal

    def convert(self, frame: RawFrame, target_hw: tuple[int, int],
                roi: RoiConfig | None = None) -> ConvertedFrame:
        """Import + fused NV12->RGB/crop/resize. Consumes the frame's fd."""
        if frame.device_ordinal != self.device_ordinal:
            raise DeviceMismatchError(
                f"{frame.source_id}: frame from device {frame.device_ordinal}, "
                f"bridge on {self.device_ordinal}"
            )
        fw, fh = frame.crop.width, frame.crop.height
        if roi is not None:
            rx, ry, rw, rh = roi.crop_rect_px(fw, fh)
            src = (frame.crop.x + rx, frame.crop.y + ry, rw, rh)
        else:
            rx = ry = 0
            src = (frame.crop.x, frame.crop.y, fw, fh)

        th, tw = target_hw
        tensor = self._core.nv12_dmabuf_to_rgb(
            frame.dmabuf_fd,
            frame.width, frame.height,
            [(p.offset, p.pitch) for p in frame.planes],
            frame.drm_modifier,
            src,                      # (x, y, w, h) within the surface
            (tw, th),
            frame.color_range is ColorRange.FULL,
            frame.color_matrix is not ColorMatrix.BT601,  # default UNKNOWN -> BT.709
            self.device_ordinal,
        )
        frame.dmabuf_fd = -1  # ownership transferred; extension closed it

        return ConvertedFrame(
            tensor=tensor,
            transform=RoiTransform(
                offset_x=rx, offset_y=ry,
                scale_x=src[2] / tw, scale_y=src[3] / th,
            ),
            frame_width=fw, frame_height=fh,
        )
