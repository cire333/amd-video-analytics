"""OC-SORT-style tracker (Cao et al. 2023), motion-only, on the SORT
Kalman infrastructure. Implements TrackerProtocol.

Two observation-centric ideas that specifically fight occlusion ID churn:

- OCM (motion consistency): association cost blends IoU with agreement
  between a track's observed velocity direction and the direction implied
  by linking its last observation to the candidate detection. Two crossing
  vehicles with similar boxes stop swapping ids because their headings
  differ.
- ORU (re-update): when a track is recovered after being lost k frames,
  the filter is re-run along a virtual trajectory interpolated between the
  last real observation and the recovering one, undoing the drift the
  blind Kalman prediction accumulated during the gap.

Like ByteTrack, a low-confidence second association pass rescues partially
occluded objects without letting low-confidence noise spawn tracks.
"""
from __future__ import annotations

import numpy as np

from .frame import ObjectMeta
from .kalman_tracker import _iou_matrix, _KalmanBoxFilter, _Track


def _center(bbox) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])


class _OcTrack(_Track):
    def __init__(self, track_id: int, det: ObjectMeta, frame_idx: int):
        super().__init__(track_id, det)
        self.last_obs = det.bbox            # last REAL observation
        self.prev_obs = None                # observation before that
        self.last_obs_frame = frame_idx

    @property
    def obs_velocity(self) -> np.ndarray | None:
        """Direction of motion from the last two real observations."""
        if self.prev_obs is None:
            return None
        v = _center(self.last_obs) - _center(self.prev_obs)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else None


class OcSortTracker:
    def __init__(self, iou_threshold: float = 0.25, max_age: int = 30,
                 min_hits: int = 3, velocity_weight: float = 0.2,
                 high_conf: float = 0.5, low_conf: float = 0.1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.velocity_weight = velocity_weight
        self.high_conf = high_conf
        self.low_conf = low_conf
        self._tracks: list[_OcTrack] = []
        self._next_id = 1
        self._frame = 0

    def update(self, detections: list[ObjectMeta]) -> list[ObjectMeta]:
        self._frame += 1
        for t in self._tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1

        high = [d for d in detections if d.confidence >= self.high_conf]
        low = [d for d in detections
               if self.low_conf <= d.confidence < self.high_conf]

        matched: set[int] = set()
        m1 = self._associate(high, list(range(len(self._tracks))))
        for di, det in enumerate(high):
            ti = m1.get(di)
            if ti is None:
                self._tracks.append(_OcTrack(self._next_id, det, self._frame))
                self._next_id += 1
                track = self._tracks[-1]
            else:
                track = self._tracks[ti]
                self._hit(track, det)
                matched.add(ti)
            self._emit(track, det)

        remaining = [i for i in range(len(self._tracks))
                     if i not in matched and self._tracks[i].time_since_update == 1]
        m2 = self._associate(low, remaining)
        for di, det in enumerate(low):
            ti = m2.get(di)
            if ti is None:
                det.track_id = None
                continue
            self._hit(self._tracks[ti], det)
            self._emit(self._tracks[ti], det)

        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= self.max_age]
        return detections

    # -- association with OCM velocity-consistency cost ----------------------

    def _associate(self, dets: list[ObjectMeta], track_idx: list[int]
                   ) -> dict[int, int]:
        if not dets or not track_idx:
            return {}
        det_boxes = np.array([d.bbox for d in dets], dtype=np.float64)
        trk_boxes = np.array([self._tracks[i].kf.bbox for i in track_idx],
                             dtype=np.float64)
        score = _iou_matrix(det_boxes, trk_boxes)

        for cj, ti in enumerate(track_idx):
            trk = self._tracks[ti]
            v = trk.obs_velocity
            if v is None:
                continue
            anchor = _center(trk.last_obs)
            for di in range(len(dets)):
                d = _center(dets[di].bbox) - anchor
                n = np.linalg.norm(d)
                if n < 1e-6:
                    continue
                # cos similarity in [-1, 1] -> bonus in [-w, +w]
                score[di, cj] += self.velocity_weight * float(v @ (d / n))

        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(-score)
            pairs = zip(rows, cols)
        except ImportError:
            pairs = []
            used_r, used_c = set(), set()
            for r, c in np.dstack(np.unravel_index(
                    np.argsort(-score, axis=None), score.shape))[0]:
                if r in used_r or c in used_c:
                    continue
                used_r.add(int(r)); used_c.add(int(c))
                pairs.append((int(r), int(c)))
        # gate on raw IoU, not the blended score
        iou = _iou_matrix(det_boxes, trk_boxes)
        return {int(r): track_idx[int(c)] for r, c in pairs
                if iou[int(r), int(c)] >= self.iou_threshold}

    # -- update with ORU ------------------------------------------------------

    def _hit(self, track: _OcTrack, det: ObjectMeta) -> None:
        gap = self._frame - track.last_obs_frame
        if gap > 1:
            # ORU: re-update along the virtual trajectory between the last
            # real observation and this one, undoing blind-prediction drift
            last = np.array(track.last_obs, dtype=np.float64)
            cur = np.array(det.bbox, dtype=np.float64)
            track.kf = _KalmanBoxFilter(track.last_obs)
            for k in range(1, gap + 1):
                track.kf.predict()
                track.kf.update(tuple(last + (cur - last) * (k / gap)))
        else:
            track.kf.update(det.bbox)
        track.prev_obs = track.last_obs
        track.last_obs = det.bbox
        track.last_obs_frame = self._frame
        track.hits += 1
        track.time_since_update = 0
        track.label_votes[det.label] = track.label_votes.get(det.label, 0) + 1

    def _emit(self, track: _OcTrack, det: ObjectMeta) -> None:
        confirmed = track.hits >= self.min_hits
        det.track_id = track.id if confirmed else None
        if confirmed:
            det.label = track.label
