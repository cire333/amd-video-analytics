"""SORT-style Kalman tracker (constant-velocity, Hungarian assignment).

Implements TrackerProtocol. State per track: [cx, cy, s, r, vcx, vcy, vs]
(center, box area, aspect ratio, velocities) — the standard SORT
formulation, numpy-only.

Design choices for traffic scenes:
- Association is CLASS-AGNOSTIC: car/truck confidence flicker must not
  split a physical track. The reported label is a per-track majority vote.
- Detections from unconfirmed (tentative) tracks are still returned, with
  track_id=None until the track survives `min_hits` frames — so detection
  counts are unaffected by tracking, matching how the comparison scripts
  consume output.
"""
from __future__ import annotations

import numpy as np

from .frame import ObjectMeta


def _bbox_to_z(bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w * h, w / max(h, 1e-6)])


def _x_to_bbox(x) -> tuple[float, float, float, float]:
    cx, cy, s, r = x[0], x[1], max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))[:, None]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))[None, :]
    return inter / np.clip(area_a + area_b - inter, 1e-6, None)


class _KalmanBoxFilter:
    """Constant-velocity Kalman filter over [cx, cy, s, r]."""

    def __init__(self, bbox):
        dim_x, dim_z = 7, 4
        self.F = np.eye(dim_x)
        for i in range(3):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((dim_z, dim_x))
        self.H[:4, :4] = np.eye(4)

        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.Q = np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 1e-4])

        self.x = np.zeros(dim_x)
        self.x[:4] = _bbox_to_z(bbox)

    def predict(self) -> None:
        if self.x[2] + self.x[6] <= 0:  # area would go negative
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, bbox) -> None:
        z = _bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def bbox(self):
        return _x_to_bbox(self.x)


class _Track:
    def __init__(self, track_id: int, det: ObjectMeta):
        self.id = track_id
        self.kf = _KalmanBoxFilter(det.bbox)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.label_votes: dict[str, int] = {det.label: 1}

    @property
    def label(self) -> str:
        return max(self.label_votes, key=self.label_votes.get)


class SortTracker:
    """TrackerProtocol implementation. One instance per source."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15,
                 min_hits: int = 3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[ObjectMeta]) -> list[ObjectMeta]:
        for t in self._tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1

        if detections and self._tracks:
            det_boxes = np.array([d.bbox for d in detections], dtype=np.float64)
            trk_boxes = np.array([t.kf.bbox for t in self._tracks], dtype=np.float64)
            iou = _iou_matrix(det_boxes, trk_boxes)
            matches = self._assign(iou)
        else:
            matches = {}

        matched_tracks = set()
        for di, det in enumerate(detections):
            ti = matches.get(di)
            if ti is None:
                track = _Track(self._next_id, det)
                self._next_id += 1
                self._tracks.append(track)
            else:
                track = self._tracks[ti]
                track.kf.update(det.bbox)
                track.hits += 1
                track.time_since_update = 0
                track.label_votes[det.label] = track.label_votes.get(det.label, 0) + 1
                matched_tracks.add(ti)
            confirmed = track.hits >= self.min_hits
            det.track_id = track.id if confirmed else None
            if confirmed:
                det.label = track.label  # majority vote smooths car/truck flicker

        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= self.max_age]
        return detections

    def _assign(self, iou: np.ndarray) -> dict[int, int]:
        """Hungarian assignment (scipy), greedy fallback; gated by IoU."""
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(-iou)
            return {int(r): int(c) for r, c in zip(rows, cols)
                    if iou[r, c] >= self.iou_threshold}
        except ImportError:
            matches: dict[int, int] = {}
            used_c: set[int] = set()
            order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None),
                                               iou.shape))[0]
            for r, c in order:
                if iou[r, c] < self.iou_threshold:
                    break
                if r in matches or c in used_c:
                    continue
                matches[int(r)] = int(c)
                used_c.add(int(c))
            return matches
