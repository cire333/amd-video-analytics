"""Frame and metadata types.

RawFrame is the unit that crosses the decoder -> bridge boundary; the
FrameMeta/ObjectMeta hierarchy rides alongside tensors downstream (the
NvDsBatchMeta/NvDsFrameMeta/NvDsObjectMeta analog).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class ColorRange(Enum):
    LIMITED = "limited"  # 16-235
    FULL = "full"        # 0-255


class ColorMatrix(Enum):
    BT601 = "bt601"
    BT709 = "bt709"
    UNKNOWN = "unknown"  # municipal encoders mistag; bridge defaults to BT709 for HD


@dataclass(frozen=True)
class CropRect:
    """Real content dims from the decoder — decoded surface may be padded."""
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class PlaneLayout:
    offset: int
    pitch: int  # aligned row pitch; NEVER assume pitch == width


@dataclass
class RawFrame:
    """A decoded NV12 surface exported as a dmabuf, pre-import.

    Owns its fd: exactly one of close() (dropped) or the bridge import
    (ownership transferred) must run, or fds leak toward the ulimit.
    """
    source_id: str
    pts: int                       # stream PTS, microseconds (container timebase)
    recv_us: int                   # local monotonic arrival time — the batcher
                                   # windows on this; PTS timebases differ per source
    width: int                     # surface (padded) dims
    height: int
    crop: CropRect
    dmabuf_fd: int
    planes: tuple[PlaneLayout, ...]  # Y then interleaved UV
    drm_modifier: int              # DRM format modifier; 0 == linear
    color_range: ColorRange
    color_matrix: ColorMatrix
    device_ordinal: int
    # Driver-detiled NV12 in host memory (same planes semantics) when the
    # surface was tiled and dmabuf export wasn't usable; dmabuf_fd is -1 then.
    host_data: bytes | None = None

    def close(self) -> None:
        if self.dmabuf_fd >= 0:
            os.close(self.dmabuf_fd)
            self.dmabuf_fd = -1

    @property
    def is_linear(self) -> bool:
        return self.drm_modifier == 0


@dataclass
class ObjectMeta:
    """One detection/track, always in FULL-FRAME pixel coordinates.

    ROI gating happens before detection; results are un-projected back
    before this object is created (architecture §8).
    """
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 full-frame
    track_id: int | None = None
    label: str = ""


@dataclass
class FrameMeta:
    source_id: str
    pts: int
    frame_width: int   # full-frame dims (crop rect), for un-projection sanity
    frame_height: int
    objects: list[ObjectMeta] = field(default_factory=list)


@dataclass
class BatchMeta:
    frames: list[FrameMeta] = field(default_factory=list)
