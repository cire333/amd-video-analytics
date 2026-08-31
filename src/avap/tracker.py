"""Tracker bank (architecture §10). Hardware-agnostic.

Per-source tracker state, keyed by source_id, always fed FULL-FRAME
coordinates (never ROI-local — a turn across ROIs must stay one track).

The GridMatrix Kalman tracker plugs in by implementing TrackerProtocol;
IouTracker below is a minimal stand-in so the pipeline runs end-to-end.
"""
from __future__ import annotations

from typing import Callable, Protocol

from .frame import ObjectMeta


class TrackerProtocol(Protocol):
    def update(self, detections: list[ObjectMeta]) -> list[ObjectMeta]:
        """Assign track_ids; returns the tracked objects."""
        ...


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class IouTracker:
    """Greedy IoU association placeholder — replace with the real Kalman tracker."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[int, tuple[tuple[float, float, float, float], int]] = {}
        self._next_id = 1

    def update(self, detections: list[ObjectMeta]) -> list[ObjectMeta]:
        # age existing tracks
        self._tracks = {tid: (bbox, age + 1) for tid, (bbox, age) in self._tracks.items()
                        if age + 1 <= self.max_age}
        unmatched = set(self._tracks)
        for det in sorted(detections, key=lambda d: -d.confidence):
            best_tid, best_iou = None, self.iou_threshold
            for tid in unmatched:
                iou = _iou(det.bbox, self._tracks[tid][0])
                if iou > best_iou:
                    best_tid, best_iou = tid, iou
            if best_tid is None:
                best_tid = self._next_id
                self._next_id += 1
            else:
                unmatched.discard(best_tid)
            det.track_id = best_tid
            self._tracks[best_tid] = (det.bbox, 0)
        return detections


class TrackerBank:
    def __init__(self, tracker_factory: Callable[[], TrackerProtocol] = IouTracker):
        self._factory = tracker_factory
        self._trackers: dict[str, TrackerProtocol] = {}

    def update(self, source_id: str, detections_full_frame: list[ObjectMeta]
               ) -> list[ObjectMeta]:
        if source_id not in self._trackers:
            self._trackers[source_id] = self._factory()
        return self._trackers[source_id].update(detections_full_frame)

    def remove(self, source_id: str) -> None:
        self._trackers.pop(source_id, None)
