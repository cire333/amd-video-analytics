"""ByteTrack-style tracker on the SORT Kalman infrastructure.

Two-stage association (Zhang et al. 2022): high-confidence detections match
first; LOW-confidence detections (which SORT would discard) then match
against still-unmatched tracks, keeping tracks alive through partial
occlusion and detector flicker. New tracks spawn only from high-confidence
detections. Implements TrackerProtocol.
"""
from __future__ import annotations

import numpy as np

from .frame import ObjectMeta
from .kalman_tracker import SortTracker, _iou_matrix, _Track


class ByteTracker(SortTracker):
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30,
                 min_hits: int = 3, high_conf: float = 0.5,
                 low_conf: float = 0.1):
        super().__init__(iou_threshold, max_age, min_hits)
        self.high_conf = high_conf
        self.low_conf = low_conf

    def update(self, detections: list[ObjectMeta]) -> list[ObjectMeta]:
        for t in self._tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1

        high = [d for d in detections if d.confidence >= self.high_conf]
        low = [d for d in detections
               if self.low_conf <= d.confidence < self.high_conf]

        # stage 1: high-confidence dets vs all tracks
        matched_tracks: set[int] = set()
        m1 = self._match(high, list(range(len(self._tracks))))
        for di, det in enumerate(high):
            ti = m1.get(di)
            if ti is None:
                track = _Track(self._next_id, det)
                self._next_id += 1
                self._tracks.append(track)
            else:
                track = self._tracks[ti]
                self._hit(track, det)
                matched_tracks.add(ti)
            self._emit(track, det)

        # stage 2: low-confidence dets vs remaining tracks (rescue pass);
        # unmatched low-conf dets do NOT spawn tracks
        remaining = [i for i in range(len(self._tracks))
                     if i not in matched_tracks
                     and self._tracks[i].time_since_update == 1]
        m2 = self._match(low, remaining)
        for di, det in enumerate(low):
            ti = m2.get(di)
            if ti is None:
                det.track_id = None
                continue
            track = self._tracks[ti]
            self._hit(track, det)
            self._emit(track, det)

        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= self.max_age]
        return detections

    # -- helpers -------------------------------------------------------------

    def _match(self, dets: list[ObjectMeta], track_idx: list[int]) -> dict[int, int]:
        if not dets or not track_idx:
            return {}
        iou = _iou_matrix(
            np.array([d.bbox for d in dets], dtype=np.float64),
            np.array([self._tracks[i].kf.bbox for i in track_idx], dtype=np.float64))
        local = self._assign(iou)
        return {di: track_idx[ci] for di, ci in local.items()}

    def _hit(self, track: _Track, det: ObjectMeta) -> None:
        track.kf.update(det.bbox)
        track.hits += 1
        track.time_since_update = 0
        track.label_votes[det.label] = track.label_votes.get(det.label, 0) + 1

    def _emit(self, track: _Track, det: ObjectMeta) -> None:
        confirmed = track.hits >= self.min_hits
        det.track_id = track.id if confirmed else None
        if confirmed:
            det.label = track.label
