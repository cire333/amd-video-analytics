"""Annotation utilities: draw tracked objects and write annotated video
through the VCN hardware encoder.

    from avap.annotate import AnnotatedVideo, draw_objects

    out = AnnotatedVideo("annotated.mp4", fps=25)      # or rtsp://...
    out.write(rgb, tracked_objects)                    # draws + VCN-encodes
    out.close()

Works with both APIs' object types (anything with .bbox, .track_id,
.label, .confidence — avap.ObjectMeta and avap.advanced.AdvObjectMeta).
"""
from __future__ import annotations

import numpy as np

from .encode import VideoEncoder


def _color(track_id: int | None) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id or 0)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw_objects(img: np.ndarray, objects, thickness: int = 2,
                 font_scale: float = 0.5) -> np.ndarray:
    """Draw boxes + '#id label conf' tags in place; returns img.
    Colors are keyed by track id, so they match across frames and across
    the batch scripts' historical output."""
    import cv2
    for o in objects:
        x1, y1, x2, y2 = (int(v) for v in o.bbox)
        c = _color(getattr(o, "track_id", None))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thickness)
        tag = ""
        if getattr(o, "track_id", None) is not None:
            tag += f"#{o.track_id} "
        tag += f"{getattr(o, 'label', '')} {getattr(o, 'confidence', 0):.2f}"
        cv2.putText(img, tag, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, c, thickness)
    return img


class AnnotatedVideo:
    """Draw + hardware-encode annotated video. The encoder is created
    lazily from the first frame's dimensions (odd dims are cropped by one
    pixel — encoders need even sizes)."""

    def __init__(self, output: str, fps: float = 25.0, codec: str = "h264",
                 bitrate: int = 8_000_000, is_bgr: bool = False):
        self.output = output
        self.fps = fps
        self.codec = codec
        self.bitrate = bitrate
        self.is_bgr = is_bgr
        self.frames_written = 0
        self._enc: VideoEncoder | None = None

    def write(self, frame: np.ndarray, objects=()) -> None:
        img = frame.copy()
        if objects:
            draw_objects(img, objects)
        h, w = img.shape[:2]
        img = img[:h - h % 2, :w - w % 2]
        if self._enc is None:
            self._enc = VideoEncoder(self.output, img.shape[1], img.shape[0],
                                     fps=self.fps, codec=self.codec,
                                     bitrate=self.bitrate)
        if self.is_bgr:
            self._enc.write_bgr(img)
        else:
            self._enc.write(img)
        self.frames_written += 1

    def close(self) -> None:
        if self._enc is not None:
            self._enc.close()
            self._enc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
