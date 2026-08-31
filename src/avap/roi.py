"""ROI-gated inference (architecture §8).

ROIs are polygons in NORMALIZED coordinates (0-1) so mid-stream resolution
changes re-project instead of silently pointing at the wrong pixels.
Crop (to the polygon's bounding rect) buys compute; mask (inside the rect)
only stops out-of-region distraction. ROI gates inference input, NOT
tracking: detections are un-projected to full-frame before the tracker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoiTransform:
    """Maps ROI-crop-local coords back to full-frame pixels."""
    offset_x: int
    offset_y: int
    scale_x: float  # crop px per model-input px (1.0 if crop fed at native size)
    scale_y: float

    def unproject(self, bbox_local: tuple[float, float, float, float]
                  ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox_local
        return (
            x1 * self.scale_x + self.offset_x,
            y1 * self.scale_y + self.offset_y,
            x2 * self.scale_x + self.offset_x,
            y2 * self.scale_y + self.offset_y,
        )


class RoiConfig:
    """One source's ROI: normalized polygon; mask generated once per
    resolution actually seen, not per frame."""

    def __init__(self, polygon_norm: list[tuple[float, float]]):
        if len(polygon_norm) < 3:
            raise ValueError("ROI polygon needs >= 3 vertices")
        for x, y in polygon_norm:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"ROI vertices must be normalized 0-1, got ({x}, {y})")
        self.polygon_norm = polygon_norm
        self._mask_cache: dict[tuple[int, int], np.ndarray] = {}

    def crop_rect_px(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Axis-aligned bounding rect of the polygon, pixel coords (x, y, w, h).

        Even-aligned x/y/w/h so NV12 chroma subsampling stays valid.
        """
        xs = [p[0] * frame_w for p in self.polygon_norm]
        ys = [p[1] * frame_h for p in self.polygon_norm]
        x0 = int(min(xs)) & ~1
        y0 = int(min(ys)) & ~1
        x1 = min(frame_w, (int(max(xs)) + 2) & ~1)
        y1 = min(frame_h, (int(max(ys)) + 2) & ~1)
        return x0, y0, x1 - x0, y1 - y0

    def mask(self, frame_w: int, frame_h: int) -> np.ndarray:
        """uint8 mask over the crop rect (1 inside polygon). Cached per resolution."""
        key = (frame_w, frame_h)
        if key not in self._mask_cache:
            self._mask_cache[key] = self._rasterize(frame_w, frame_h)
        return self._mask_cache[key]

    def _rasterize(self, frame_w: int, frame_h: int) -> np.ndarray:
        cx, cy, cw, ch = self.crop_rect_px(frame_w, frame_h)
        poly = np.array(
            [(x * frame_w - cx, y * frame_h - cy) for x, y in self.polygon_norm]
        )
        yy, xx = np.mgrid[0:ch, 0:cw]
        # even-odd rule point-in-polygon, vectorized over the crop rect
        inside = np.zeros((ch, cw), dtype=bool)
        n = len(poly)
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            crosses = (y0 <= yy) != (y1 <= yy)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_at = x0 + (yy - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crosses & (xx < x_at)
        return inside.astype(np.uint8)
