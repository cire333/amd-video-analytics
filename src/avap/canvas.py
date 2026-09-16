"""BatchCanvas — the scaling half of nvstreammux (docs/nvstreammux_reverse_engineering.md §2).

Turns a MuxBatch of RawFrames into one device-resident float32 tensor
[N, 3, H, W] (RGB, 0-1), one slot per frame:

* ``enable_padding=False``: anisotropic stretch to W x H (nvstreammux default).
* ``enable_padding=True``: aspect-preserving letterbox, centred, black fill:
  ``scale = min(W/sw, H/sh)``, ``dst = floor(src*scale)``, ``offset = (canvas-dst)//2``
  (verified pixel-exact against the 3090: 1280x720→640x640 lands at (0,140),
  1000x700→1920x1056 at (206,0) as 1508x1056).
* Optional per-source ROI crop (fused into the same kernel, architecture §8).

Each slot comes with a ``CanvasTransform`` mapping canvas pixels back to
source pixels, which is what DeepStream does when it rescales detections from
the mux canvas to ``image_resolution``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame import ColorMatrix, ColorRange, RawFrame
from .roi import RoiConfig
from .streammux import MuxBatch


@dataclass(frozen=True)
class CanvasTransform:
    """canvas px -> source (full-frame) px for one slot."""
    dst_x: int
    dst_y: int
    dst_w: int
    dst_h: int
    src_x: int            # ROI/crop origin in the source frame
    src_y: int
    src_w: int
    src_h: int
    frame_width: int      # full source frame (crop rect) dims
    frame_height: int

    @property
    def scale_x(self) -> float:
        return self.src_w / self.dst_w

    @property
    def scale_y(self) -> float:
        return self.src_h / self.dst_h

    def unproject(self, bbox):
        x1, y1, x2, y2 = bbox
        return (self.src_x + (x1 - self.dst_x) * self.scale_x,
                self.src_y + (y1 - self.dst_y) * self.scale_y,
                self.src_x + (x2 - self.dst_x) * self.scale_x,
                self.src_y + (y2 - self.dst_y) * self.scale_y)

    def unproject_clipped(self, bbox):
        x1, y1, x2, y2 = self.unproject(bbox)
        return (min(max(x1, 0.0), self.frame_width), min(max(y1, 0.0), self.frame_height),
                min(max(x2, 0.0), self.frame_width), min(max(y2, 0.0), self.frame_height))


def letterbox_geometry(src_w: int, src_h: int, canvas_w: int, canvas_h: int,
                       enable_padding: bool) -> tuple[int, int, int, int]:
    """(dst_x, dst_y, dst_w, dst_h) exactly as nvstreammux places a frame."""
    if not enable_padding:
        return 0, 0, canvas_w, canvas_h
    scale = min(canvas_w / src_w, canvas_h / src_h)
    dst_w = max(1, int(src_w * scale))
    dst_h = max(1, int(src_h * scale))
    return (canvas_w - dst_w) // 2, (canvas_h - dst_h) // 2, dst_w, dst_h


class BatchCanvas:
    """Owns the [N,3,H,W] device buffer and renders MuxBatches into it."""

    def __init__(self, batch_size: int, width: int, height: int, device_ordinal: int = 0,
                 enable_padding: bool = False, interpolation: str = "bilinear",
                 default_matrix_for_untagged: ColorMatrix = ColorMatrix.BT601):
        from . import _core
        self._core = _core
        if interpolation not in ("bilinear", "nearest"):
            raise ValueError("interpolation must be 'bilinear' or 'nearest'")
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.device_ordinal = device_ordinal
        self.enable_padding = enable_padding
        self.nearest = interpolation == "nearest"
        self.default_matrix = default_matrix_for_untagged
        self.slot_floats = 3 * width * height
        self.nbytes = batch_size * self.slot_floats * 4
        self.ptr = _core.device_alloc(self.nbytes, device_ordinal)
        _core.device_memset(self.ptr, self.nbytes, 0)
        self._rendered = 0

    def close(self) -> None:
        if self.ptr:
            self._core.device_free(self.ptr)
            self.ptr = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def slot_ptr(self, slot: int) -> int:
        return self.ptr + slot * self.slot_floats * 4

    # ------------------------------------------------------------------ render
    def render_frame(self, slot: int, frame: RawFrame, roi: RoiConfig | None = None) -> CanvasTransform:
        """Fused NV12->RGB(+crop)->letterbox into ``slot``. Consumes frame.dmabuf_fd."""
        if slot >= self.batch_size:
            raise IndexError(f"slot {slot} >= batch_size {self.batch_size}")
        if frame.device_ordinal != self.device_ordinal:
            raise RuntimeError(f"{frame.source_id}: frame on device {frame.device_ordinal}, "
                               f"canvas on {self.device_ordinal}")
        fw, fh = frame.crop.width, frame.crop.height
        if roi is not None:
            rx, ry, rw, rh = roi.crop_rect_px(fw, fh)
            src = (frame.crop.x + rx, frame.crop.y + ry, rw, rh)
        else:
            rx = ry = 0
            src = (frame.crop.x, frame.crop.y, fw, fh)
        dst = letterbox_geometry(src[2], src[3], self.width, self.height, self.enable_padding)
        planes = [(p.offset, p.pitch) for p in frame.planes]
        full_range = frame.color_range is ColorRange.FULL
        matrix = frame.color_matrix if frame.color_matrix is not ColorMatrix.UNKNOWN else self.default_matrix
        bt709 = matrix is ColorMatrix.BT709
        common = (src, full_range, bt709, self.device_ordinal, self.slot_ptr(slot),
                  (self.width, self.height), dst, self.nearest)
        if frame.dmabuf_fd >= 0:
            fd = frame.dmabuf_fd
            frame.dmabuf_fd = -1  # ownership moves to the import
            self._core.nv12_dmabuf_to_canvas(fd, frame.width, frame.height, planes,
                                             frame.drm_modifier, *common)
        elif frame.host_data is not None:
            self._core.nv12_host_to_canvas(frame.host_data, planes, *common)
        else:
            raise RuntimeError(f"{frame.source_id}: frame has neither dmabuf nor host data")
        return CanvasTransform(*dst, src[0] - frame.crop.x, src[1] - frame.crop.y,
                               src[2], src[3], fw, fh)

    def render(self, batch: MuxBatch, rois: dict | None = None,
               zero_unused: bool = True) -> list[CanvasTransform]:
        """Render every frame of ``batch`` (payload = RawFrame) into slots
        0..n-1; unused slots are zeroed so a fixed-shape model sees black."""
        if len(batch.frames) > self.batch_size:
            raise ValueError(f"batch has {len(batch.frames)} frames > canvas batch_size "
                             f"{self.batch_size}")
        transforms = []
        for slot, meta in enumerate(batch.frames):
            frame: RawFrame = meta.payload
            roi = None if rois is None else rois.get(frame.source_id)
            transforms.append(self.render_frame(slot, frame, roi))
        n = len(batch.frames)
        if zero_unused and n < self.batch_size:
            self._core.device_memset(self.slot_ptr(n), (self.batch_size - n) * self.slot_floats * 4, 0)
        self._rendered = n
        return transforms

    # ------------------------------------------------------------------ readback
    def to_host(self, n: int | None = None) -> np.ndarray:
        """Copy [n,3,H,W] back to host (debug / non-zero-copy inference)."""
        n = self.batch_size if n is None else n
        return self._core.device_to_host_f32(self.ptr, [n, 3, self.height, self.width])

    def device_tensor_ptr(self) -> int:
        """Raw device pointer for MIGraphX/ORT IO binding (float32, NCHW)."""
        return self.ptr
