"""NvDCF-class visual tracker for AMD (docs/nvdcf_tracker.md).

DeepStream's NvDCF = Kalman state estimator + discriminative correlation
filter (DCF) visual tracker per target + data association that blends
visual similarity, IoU and size similarity + shadow tracking. This module
reproduces that structure with the same knobs (config keys are DeepStream's
``config_tracker_NvDCF_*.yml`` names, and ``NvDcfConfig.from_yaml`` loads
those files directly), running the correlation filters on HIP + hipFFT
(``HipDcfEngine``) or numpy (``NumpyDcfEngine``, reference/CPU fallback).

The visual term is what motion-only trackers (SORT/OC-SORT/ByteTrack) lack:
when the detector misses an object, the DCF localizes it from appearance and
the track keeps its id and a box (``ObjectMeta.confidence`` = tracker
confidence, ``tracked_by`` = "dcf"); when two similar boxes cross, the
correlation response ratio disambiguates them.

What is not here (yet): NvDCF's learned ColorNames table (replaced by a soft
11-prototype colour assignment), the TAO ReID re-association, and trajectory
re-association — see the doc for the gap list.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from .frame import ObjectMeta

log = logging.getLogger(__name__)

FEATURE_SIZE_LEVELS = {1: 24, 2: 32, 3: 40, 4: 48, 5: 64}   # featureImgSizeLevel -> S

TENTATIVE, ACTIVE, SHADOW = "tentative", "active", "shadow"


# --------------------------------------------------------------------------- config

@dataclass
class NvDcfConfig:
    # BaseConfig
    minDetectorConfidence: float = 0.1894
    # TargetManagement
    maxTargetsPerStream: int = 150
    minIouDiff4NewTarget: float = 0.3686
    minTrackerConfidence: float = 0.1513
    probationAge: int = 2
    maxShadowTrackingAge: int = 42
    earlyTerminationAge: int = 1
    enableBboxUnClipping: int = 1
    # DataAssociator
    checkClassMatch: int = 1
    minMatchingScore4Overall: float = 0.0222
    minMatchingScore4SizeSimilarity: float = 0.3552
    minMatchingScore4Iou: float = 0.0548
    minMatchingScore4VisualSimilarity: float = 0.5043
    matchingScoreWeight4VisualSimilarity: float = 0.3951
    matchingScoreWeight4SizeSimilarity: float = 0.6003
    matchingScoreWeight4Iou: float = 0.4033
    tentativeDetectorConfidence: float = 0.1024
    minMatchingScore4TentativeIou: float = 0.2852
    # StateEstimator (SIMPLE: constant velocity on location)
    processNoiseVar4Loc: float = 6810.8668
    processNoiseVar4Size: float = 1541.8647
    processNoiseVar4Vel: float = 1348.4874
    measurementNoiseVar4Detector: float = 100.0
    measurementNoiseVar4Tracker: float = 293.3238
    # VisualTracker
    visualTrackerType: int = 1
    useColorNames: int = 1
    useHog: int = 1
    featureImgSizeLevel: int = 3
    featureFocusOffsetFactor_y: float = -0.1054
    filterLr: float = 0.0767
    filterChannelWeightsLr: float = 0.0339
    gaussianSigma: float = 0.5687
    # --- avap extensions (not in DS) ---
    searchRegionPaddingScale: float = 1.5     # search window = bbox * (1 + this)
    useGray: int = 1
    filterLambda: float = 1e-2
    maxVisualOnlyAge: int = 42               # consecutive DCF-only frames before shadow
    maxVisualDisplacementRatio: float = 0.5   # DCF jump > this * bbox size -> distrust it
    classAgnosticLabels: int = 1              # majority-vote label (car/truck flicker)

    @property
    def feature_size(self) -> int:
        return FEATURE_SIZE_LEVELS.get(int(self.featureImgSizeLevel), 40)

    @classmethod
    def from_yaml(cls, path: str) -> "NvDcfConfig":
        """Load a DeepStream config_tracker_NvDCF_*.yml (OpenCV-style header ok)."""
        import yaml
        text = open(path).read()
        lines = [l for l in text.splitlines() if not l.startswith("%YAML")]
        doc = yaml.safe_load("\n".join(lines)) or {}
        known = {f.name for f in fields(cls)}
        kw: dict[str, Any] = {}
        for section in doc.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    if k in known:
                        kw[k] = v
        return cls(**kw)

    def weights_sum(self) -> float:
        return (self.matchingScoreWeight4VisualSimilarity + self.matchingScoreWeight4SizeSimilarity
                + self.matchingScoreWeight4Iou) or 1.0


# --------------------------------------------------------------------------- frames

@dataclass
class VisualFrame:
    """A device CHW float32 RGB frame plus the affine that maps SOURCE pixels
    (the coordinate system of the detections) to frame pixels:
    fx = ox + x * sx, fy = oy + y * sy. For a BatchCanvas slot: sx = dst_w/src_w,
    ox = dst_x - src_x*sx (see MuxedPipeline)."""
    ptr: int
    width: int
    height: int
    sx: float = 1.0
    sy: float = 1.0
    ox: float = 0.0
    oy: float = 0.0
    host: np.ndarray | None = None      # optional CHW float copy (numpy engine)

    @property
    def affine(self) -> tuple[float, float, float, float]:
        return (float(self.sx), float(self.sy), float(self.ox), float(self.oy))


class FrameUploader:
    """Keeps one device buffer per resolution and uploads host frames into it
    (offline evaluation / CPU decoders). Frames: HWC uint8 BGR or RGB, or CHW float."""

    def __init__(self, device_ordinal: int = 0):
        from . import _core
        self._core = _core
        self.device = device_ordinal
        self._buf: dict[tuple[int, int], int] = {}

    def upload(self, img: np.ndarray, bgr: bool = True, keep_host: bool = False) -> VisualFrame:
        if img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
            rgb = img[..., ::-1] if bgr else img
            chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        elif img.ndim == 3 and img.shape[0] == 3:
            chw = np.ascontiguousarray(img, dtype=np.float32)
        else:
            raise ValueError(f"unsupported frame shape {img.shape}/{img.dtype}")
        h, w = chw.shape[1], chw.shape[2]
        key = (w, h)
        if key not in self._buf:
            self._buf[key] = self._core.device_alloc(chw.nbytes, self.device)
        self._core.device_upload_f32(self._buf[key], chw)
        return VisualFrame(self._buf[key], w, h, host=chw if keep_host else None)

    def close(self) -> None:
        for p in self._buf.values():
            self._core.device_free(p)
        self._buf.clear()


# --------------------------------------------------------------------------- DCF engines

_CN_PROTO = np.array([
    [0.05, 0.05, 0.05], [0.10, 0.20, 0.85], [0.45, 0.28, 0.12], [0.50, 0.50, 0.50],
    [0.15, 0.65, 0.20], [0.95, 0.55, 0.10], [0.95, 0.55, 0.75], [0.55, 0.20, 0.70],
    [0.85, 0.10, 0.10], [0.95, 0.95, 0.95], [0.95, 0.90, 0.15]], dtype=np.float32)


class NumpyDcfEngine:
    """Reference implementation of cpp/dcf.hip (same features, window,
    Gaussian and MOSSE update) on host frames. ``frame.host`` must be set."""

    def __init__(self, max_targets: int, feature_size: int, use_gray=True, use_colornames=True,
                 use_hog=False, lam=1e-2, gaussian_sigma=2.0, focus_offset_y=0.0):
        self.N, self.S = max_targets, feature_size
        self.use_gray, self.use_cn, self.use_hog = use_gray, use_colornames, use_hog
        self.C = int(use_gray) + 11 * int(use_colornames) + 9 * int(use_hog)
        self.lam = lam
        S = self.S
        self.K = S // 2 + 1
        self.A = np.zeros((self.N, self.C, S, self.K), np.complex64)
        self.B = np.zeros((self.N, S, self.K), np.float32)
        cy = 0.5 * (S - 1) + focus_offset_y * S
        ry = (np.arange(S) - cy) / (0.5 * (S - 1))
        wy = np.where(np.abs(ry) < 1, 0.5 * (1 + np.cos(np.pi * ry)), 0.0)
        wx = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(S) + 0.5) / S))
        self.hann = (wy[:, None] * wx[None, :]).astype(np.float32)
        self.set_gaussian_sigma(gaussian_sigma)

    @property
    def channels(self): return self.C
    @property
    def feature_size(self): return self.S

    def set_gaussian_sigma(self, sigma: float) -> None:
        S = self.S
        d = np.arange(S); d = np.where(d <= S // 2, d, d - S)
        g = np.exp(-(d[None, :] ** 2 + d[:, None] ** 2) / (2 * sigma * sigma))
        self.ghat = np.fft.rfft2(g).astype(np.complex64)

    def clear_slot(self, slot: int) -> None:
        self.A[slot] = 0; self.B[slot] = 0

    # -- feature extraction (mirrors extract_kernel) --
    def _sample(self, frame: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        H, W = frame.shape[1], frame.shape[2]
        fx = np.clip(fx, 0, W - 1); fy = np.clip(fy, 0, H - 1)
        x0 = np.floor(fx).astype(int); y0 = np.floor(fy).astype(int)
        x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
        ax = (fx - x0)[None]; ay = (fy - y0)[None]
        v00 = frame[:, y0, x0]; v01 = frame[:, y0, x1]; v10 = frame[:, y1, x0]; v11 = frame[:, y1, x1]
        return (v00 * (1 - ax) + v01 * ax) * (1 - ay) + (v10 * (1 - ax) + v11 * ax) * ay   # 3,S,S

    def extract_features(self, frame: VisualFrame, windows: np.ndarray) -> np.ndarray:
        img = frame.host
        assert img is not None, "NumpyDcfEngine needs VisualFrame.host"
        S = self.S
        out = np.zeros((len(windows), self.C, S, S), np.float32)
        idx = (np.arange(S) + 0.5) / S
        for t, (cx, cy, w, h) in enumerate(windows):
            w, h = max(w, 1.0), max(h, 1.0)
            u = cx - w / 2 + idx * w; v = cy - h / 2 + idx * h
            U, V = np.meshgrid(u, v)
            fx, fy = frame.ox + U * frame.sx, frame.oy + V * frame.sy
            rgb = self._sample(img, fx, fy)
            c = 0
            gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            if self.use_gray:
                out[t, c] = gray; c += 1
            if self.use_cn:
                d2 = ((rgb[None] - _CN_PROTO[:, :, None, None]) ** 2).sum(1)      # 11,S,S
                p = np.exp(-d2 / (2 * 0.15 ** 2)); p /= p.sum(0, keepdims=True) + 1e-6
                out[t, c:c + 11] = p; c += 11
            if self.use_hog:
                du, dv = w / S, h / S
                gxp = self._gray(img, frame.ox + (U + du) * frame.sx, fy)
                gxm = self._gray(img, frame.ox + (U - du) * frame.sx, fy)
                gyp = self._gray(img, fx, frame.oy + (V + dv) * frame.sy)
                gym = self._gray(img, fx, frame.oy + (V - dv) * frame.sy)
                gx, gy = gxp - gxm, gyp - gym
                mag = np.sqrt(gx * gx + gy * gy)
                ang = np.arctan2(gy, gx); ang = np.where(ang < 0, ang + np.pi, ang)
                b = ang / np.pi * 9
                b0 = np.floor(b).astype(int) % 9; b1 = (b0 + 1) % 9; f1 = b - np.floor(b)
                for k in range(9):
                    out[t, c + k] = np.where(b0 == k, mag * (1 - f1), 0) + np.where(b1 == k, mag * f1, 0)
                c += 9
        # per-patch, per-channel mean removal (a flat background must not
        # correlate with itself), then the Hann window — same as the HIP kernel
        out -= out.mean(axis=(2, 3), keepdims=True)
        out *= self.hann[None, None]
        return out

    def _gray(self, img, fx, fy):
        rgb = self._sample(img, fx, fy)
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

    def localize(self, frame: VisualFrame, windows: np.ndarray, slots: np.ndarray) -> np.ndarray:
        feats = self.extract_features(frame, windows)
        fhat = np.fft.rfft2(feats, axes=(-2, -1))
        resp = np.zeros((len(windows), self.S, self.S), np.float32)
        for t, slot in enumerate(slots):
            num = (self.A[slot] * fhat[t]).sum(0)
            rhat = num / (self.B[slot] + self.lam)
            resp[t] = np.fft.irfft2(rhat, s=(self.S, self.S))
        return resp

    def update(self, frame: VisualFrame, windows: np.ndarray, slots: np.ndarray,
               init: np.ndarray, lr: float) -> None:
        feats = self.extract_features(frame, windows)
        fhat = np.fft.rfft2(feats, axes=(-2, -1))
        for t, slot in enumerate(slots):
            a_lr = 1.0 if init[t] else lr
            self.A[slot] = (1 - a_lr) * self.A[slot] + a_lr * (self.ghat[None] * np.conj(fhat[t]))
            self.B[slot] = (1 - a_lr) * self.B[slot] + a_lr * (np.abs(fhat[t]) ** 2).sum(0)


class HipDcfEngine:
    """cpp/dcf.hip via avap._core.DcfEngine (hipFFT)."""

    def __init__(self, max_targets: int, feature_size: int, use_gray=True, use_colornames=True,
                 use_hog=False, lam=1e-2, gaussian_sigma=2.0, focus_offset_y=0.0, device_ordinal=0):
        from . import _core
        p = _core.DcfParams()
        p.max_targets, p.feature_size = int(max_targets), int(feature_size)
        p.use_gray, p.use_colornames, p.use_hog = bool(use_gray), bool(use_colornames), bool(use_hog)
        p.lambda_, p.gaussian_sigma, p.focus_offset_y = float(lam), float(gaussian_sigma), float(focus_offset_y)
        p.device_ordinal = int(device_ordinal)
        self._e = _core.DcfEngine(p)
        self.N, self.S, self.C = max_targets, feature_size, self._e.channels

    @property
    def channels(self): return self.C
    @property
    def feature_size(self): return self.S

    def set_gaussian_sigma(self, sigma): self._e.set_gaussian_sigma(float(sigma))
    def clear_slot(self, slot): self._e.clear_slot(int(slot))

    def extract_features(self, frame: VisualFrame, windows: np.ndarray) -> np.ndarray:
        return self._e.extract_features(frame.ptr, frame.width, frame.height, frame.affine,
                                        np.ascontiguousarray(windows, np.float32))

    def localize(self, frame: VisualFrame, windows: np.ndarray, slots: np.ndarray) -> np.ndarray:
        return self._e.localize(frame.ptr, frame.width, frame.height, frame.affine,
                                np.ascontiguousarray(windows, np.float32),
                                np.ascontiguousarray(slots, np.int32))

    def update(self, frame: VisualFrame, windows, slots, init, lr: float) -> None:
        self._e.update(frame.ptr, frame.width, frame.height, frame.affine,
                       np.ascontiguousarray(windows, np.float32),
                       np.ascontiguousarray(slots, np.int32),
                       np.ascontiguousarray(init, np.uint8), float(lr))


def make_engine(cfg: NvDcfConfig, backend: str = "auto", device_ordinal: int = 0):
    kw = dict(max_targets=cfg.maxTargetsPerStream, feature_size=cfg.feature_size,
              use_gray=bool(cfg.useGray), use_colornames=bool(cfg.useColorNames),
              use_hog=bool(cfg.useHog), lam=cfg.filterLambda, gaussian_sigma=cfg.gaussianSigma,
              focus_offset_y=cfg.featureFocusOffsetFactor_y)
    if backend == "numpy":
        return NumpyDcfEngine(**kw)
    try:
        return HipDcfEngine(device_ordinal=device_ordinal, **kw)
    except Exception as e:  # noqa: BLE001
        if backend == "hip":
            raise
        log.warning("HipDcfEngine unavailable (%s); using NumpyDcfEngine", e)
        return NumpyDcfEngine(**kw)


# --------------------------------------------------------------------------- state estimator

class _Kalman:
    """NvDCF SIMPLE estimator: constant velocity on (cx, cy), random walk on (w, h)."""

    def __init__(self, bbox, cfg: NvDcfConfig):
        cx, cy, w, h = _to_cxcywh(bbox)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0])
        self.P = np.diag([cfg.measurementNoiseVar4Detector] * 4 + [cfg.processNoiseVar4Vel * 10] * 2)
        self.F = np.eye(6); self.F[0, 4] = 1.0; self.F[1, 5] = 1.0
        self.Q = np.diag([cfg.processNoiseVar4Loc] * 2 + [cfg.processNoiseVar4Size] * 2
                         + [cfg.processNoiseVar4Vel] * 2)
        self.H = np.zeros((4, 6)); self.H[:4, :4] = np.eye(4)
        self.r_det = cfg.measurementNoiseVar4Detector
        self.r_trk = cfg.measurementNoiseVar4Tracker

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, bbox, from_detector: bool) -> None:
        z = np.array(_to_cxcywh(bbox))
        R = np.eye(4) * (self.r_det if from_detector else self.r_trk)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def bbox(self):
        cx, cy, w, h = self.x[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _to_cxcywh(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0))


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    ab = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.clip(aa + ab - inter, 1e-6, None)


def _size_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    wa, ha = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
    wb, hb = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    rw = np.minimum(wa[:, None], wb[None, :]) / np.clip(np.maximum(wa[:, None], wb[None, :]), 1e-6, None)
    rh = np.minimum(ha[:, None], hb[None, :]) / np.clip(np.maximum(ha[:, None], hb[None, :]), 1e-6, None)
    return rw * rh


# --------------------------------------------------------------------------- targets

class _Target:
    def __init__(self, tid: int, det: ObjectMeta, slot: int, cfg: NvDcfConfig):
        self.id = tid
        self.slot = slot
        self.kf = _Kalman(det.bbox, cfg)
        self.bbox = tuple(det.bbox)
        self.class_id = det.class_id
        self.label_votes: dict[str, int] = {det.label: 1}
        self.age = 1
        self.hits = 1
        self.shadow_age = 0
        self.visual_only_age = 0
        self.state = TENTATIVE
        self.tracker_conf = 1.0
        self.resp: np.ndarray | None = None       # last localization response map
        self.search: tuple | None = None          # (cx, cy, w, h) of the last search window
        self.localized: tuple | None = None       # DCF-localized bbox this frame
        self.needs_init = True

    @property
    def label(self) -> str:
        return max(self.label_votes, key=self.label_votes.get)

    @property
    def confirmed(self) -> bool:
        return self.age > 0 and self.state != TENTATIVE


# --------------------------------------------------------------------------- tracker

class NvDcfTracker:
    """TrackerProtocol/VisualTrackerProtocol implementation, one per source.

    ``update(detections, frame=None)``: without a frame it degrades to a
    Kalman + IoU/size tracker with shadow tracking (no visual term)."""

    uses_frames = True

    def __init__(self, cfg: NvDcfConfig | None = None, backend: str = "auto",
                 device_ordinal: int = 0, engine=None):
        self.cfg = cfg or NvDcfConfig()
        self.engine = engine or make_engine(self.cfg, backend, device_ordinal)
        self._targets: list[_Target] = []
        self._free_slots = list(range(self.cfg.maxTargetsPerStream))
        self._next_id = 1
        self.frame_idx = 0
        self.stats = {"visual_bridged": 0, "created": 0, "terminated": 0}

    # ------------------------------------------------------------ helpers
    def _search_window(self, bbox) -> tuple[float, float, float, float]:
        cx, cy, w, h = _to_cxcywh(bbox)
        k = 1.0 + self.cfg.searchRegionPaddingScale
        return (cx, cy, w * k, h * k)

    def _response_at(self, t: _Target, bbox) -> float:
        """Correlation response at a box centre, relative to the peak (visual similarity)."""
        if t.resp is None or t.search is None:
            return 0.0
        S = self.engine.feature_size
        cx, cy, sw, sh = t.search
        bx, by = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        # response index 0 == zero displacement from the window centre (circular)
        ix = int(round((bx - cx) / sw * S)) % S
        iy = int(round((by - cy) / sh * S)) % S
        if abs(bx - cx) > sw / 2 or abs(by - cy) > sh / 2:
            return 0.0
        peak = float(t.resp.max())
        return float(t.resp[iy, ix]) / peak if peak > 1e-6 else 0.0

    # ------------------------------------------------------------ main step
    def update(self, detections: list[ObjectMeta], frame: VisualFrame | None = None
               ) -> list[ObjectMeta]:
        cfg = self.cfg
        self.frame_idx += 1
        for t in self._targets:
            t.kf.predict()
            t.age += 1
            t.localized = None
            t.resp = None

        # 1. visual localization of every target (predicted position -> DCF peak)
        if frame is not None and self._targets:
            self._localize_all(frame)

        # 2. data association
        dets = [d for d in detections if d.confidence >= cfg.minDetectorConfidence]
        full = [d for d in dets if d.confidence >= cfg.tentativeDetectorConfidence]
        tentative = [d for d in dets if d.confidence < cfg.tentativeDetectorConfidence]
        matched_t: set[int] = set()
        matched_d: dict[int, int] = {}
        if full and self._targets:
            matched_d = self._associate(full, frame is not None)
            matched_t = set(matched_d.values())
        if tentative and self._targets:
            rem = [i for i in range(len(self._targets)) if i not in matched_t]
            if rem:
                m = self._associate_iou_only(tentative, rem)
                for di, ti in m.items():
                    matched_t.add(ti)
                    matched_d[len(full) + di] = ti
        all_dets = full + tentative

        out: list[ObjectMeta] = []
        # 3. matched targets: detector measurement
        for di, ti in matched_d.items():
            t, det = self._targets[ti], all_dets[di]
            t.kf.update(det.bbox, from_detector=True)
            t.bbox = tuple(det.bbox)
            t.hits += 1
            t.shadow_age = 0
            t.visual_only_age = 0
            t.label_votes[det.label] = t.label_votes.get(det.label, 0) + 1
            if t.state != TENTATIVE or t.age > cfg.probationAge:
                t.state = ACTIVE
            self._emit(t, det, out)

        # 4. unmatched targets: DCF-only continuation or shadow mode
        for ti, t in enumerate(self._targets):
            if ti in matched_t:
                continue
            visual_ok = (t.localized is not None and t.tracker_conf >= cfg.minTrackerConfidence
                         and t.visual_only_age < cfg.maxVisualOnlyAge and t.state != TENTATIVE)
            if visual_ok:
                t.kf.update(t.localized, from_detector=False)
                t.bbox = t.kf.bbox if cfg.enableBboxUnClipping else t.localized
                t.visual_only_age += 1
                t.shadow_age = 0
                t.state = ACTIVE
                self.stats["visual_bridged"] += 1
                out.append(ObjectMeta(class_id=t.class_id, confidence=float(t.tracker_conf),
                                      bbox=tuple(float(v) for v in t.bbox), track_id=t.id,
                                      label=t.label, tracked_by="dcf"))
            else:
                t.shadow_age += 1
                if t.state != TENTATIVE:
                    t.state = SHADOW
                t.bbox = t.kf.bbox

        # 5. creation
        existing = np.array([t.bbox for t in self._targets], np.float64) if self._targets else None
        for di, det in enumerate(full):
            if di in matched_d:
                continue
            if existing is not None and len(existing):
                if _iou_matrix(np.array([det.bbox]), existing)[0].max() >= cfg.minIouDiff4NewTarget:
                    det.track_id = None
                    continue
            if not self._free_slots:
                det.track_id = None
                continue
            t = _Target(self._next_id, det, self._free_slots.pop(0), cfg)
            self._next_id += 1
            self._targets.append(t)
            self.stats["created"] += 1
            det.track_id = None                      # tentative until probation passes
            existing = np.array([x.bbox for x in self._targets], np.float64)
        # every input detection is returned (unmatched/tentative ones with track_id None),
        # like the motion-only trackers, so detection counts are tracker-independent
        emitted = {id(o) for o in out}
        for det in all_dets:
            if id(det) not in emitted:
                det.track_id = None
                out.append(det)

        # 6. termination
        keep = []
        for t in self._targets:
            dead = (t.shadow_age > cfg.maxShadowTrackingAge
                    or (t.state == TENTATIVE and t.shadow_age >= cfg.earlyTerminationAge))
            if dead:
                self.engine.clear_slot(t.slot)
                self._free_slots.append(t.slot)
                self.stats["terminated"] += 1
            else:
                keep.append(t)
        self._targets = keep

        # 7. filter learning at the final boxes
        if frame is not None:
            self._update_filters(frame)
        return out

    # ------------------------------------------------------------ steps
    def _localize_all(self, frame: VisualFrame) -> None:
        S = self.engine.feature_size
        windows = np.array([self._search_window(t.kf.bbox) for t in self._targets], np.float32)
        slots = np.array([t.slot for t in self._targets], np.int32)
        resp = self.engine.localize(frame, windows, slots)
        for t, win, r in zip(self._targets, windows, resp):
            t.search = tuple(float(v) for v in win)
            t.resp = r
            if t.needs_init:
                continue                                  # filter not learned yet
            iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
            dy = iy if iy <= S // 2 else iy - S            # circular displacement
            dx = ix if ix <= S // 2 else ix - S
            cx, cy, sw, sh = t.search
            ncx, ncy = cx + dx * sw / S, cy + dy * sh / S
            w, h = t.kf.x[2], t.kf.x[3]
            jump = np.hypot((ncx - cx) / max(w, 1.0), (ncy - cy) / max(h, 1.0))
            if jump > self.cfg.maxVisualDisplacementRatio:
                t.tracker_conf = 0.0                       # implausible jump: distrust
                continue
            t.localized = (ncx - w / 2, ncy - h / 2, ncx + w / 2, ncy + h / 2)
            peak = float(r.max())
            # PSR-flavoured confidence: peak height, penalised by side-lobe level
            side = np.delete(r.ravel(), np.argmax(r))
            psr = (peak - side.mean()) / (side.std() + 1e-6)
            t.tracker_conf = float(np.clip(peak, 0.0, 1.0) * np.clip(psr / 20.0, 0.0, 1.0))

    def _associate(self, dets: list[ObjectMeta], with_visual: bool) -> dict[int, int]:
        cfg = self.cfg
        T = self._targets
        db = np.array([d.bbox for d in dets], np.float64)
        tb = np.array([t.kf.bbox for t in T], np.float64)
        iou = _iou_matrix(db, tb)
        if with_visual:
            # a good DCF localization tightens IoU, a bad one must not break it
            tl = np.array([t.localized if t.localized is not None else t.kf.bbox for t in T], np.float64)
            iou = np.maximum(iou, _iou_matrix(db, tl))
        size = _size_similarity(db, tb)
        vis = np.zeros_like(iou)
        if with_visual:
            for ti, t in enumerate(T):
                if t.resp is None or t.needs_init:
                    continue
                for di, d in enumerate(dets):
                    vis[di, ti] = self._response_at(t, d.bbox)
        wv, ws, wi = (cfg.matchingScoreWeight4VisualSimilarity, cfg.matchingScoreWeight4SizeSimilarity,
                      cfg.matchingScoreWeight4Iou)
        if not with_visual:
            wv = 0.0
        score = (wv * vis + ws * size + wi * iou) / ((wv + ws + wi) or 1.0)
        valid = (iou >= cfg.minMatchingScore4Iou) & (size >= cfg.minMatchingScore4SizeSimilarity)
        if with_visual:
            # the visual gate only applies where the filter itself is trustworthy
            trusted = np.array([(not t.needs_init) and t.resp is not None
                                and t.tracker_conf >= cfg.minTrackerConfidence for t in T])
            valid &= (vis >= cfg.minMatchingScore4VisualSimilarity) | ~trusted[None, :]
        valid &= score >= cfg.minMatchingScore4Overall
        if cfg.checkClassMatch:
            cls_d = np.array([d.class_id for d in dets])[:, None]
            cls_t = np.array([t.class_id for t in T])[None, :]
            valid &= cls_d == cls_t
        score = np.where(valid, score, -1.0)
        # cascaded: active/shadow (confirmed) targets first, tentative ones after
        matches: dict[int, int] = {}
        used_d: set[int] = set()
        for group in ([i for i, t in enumerate(T) if t.state != TENTATIVE],
                      [i for i, t in enumerate(T) if t.state == TENTATIVE]):
            if not group:
                continue
            rows = [d for d in range(len(dets)) if d not in used_d]
            if not rows:
                break
            sub = score[np.ix_(rows, group)]
            for r, c in _hungarian(sub):
                if sub[r, c] >= 0:
                    matches[rows[r]] = group[c]
                    used_d.add(rows[r])
        return matches

    def _associate_iou_only(self, dets, track_idx) -> dict[int, int]:
        db = np.array([d.bbox for d in dets], np.float64)
        tb = np.array([self._targets[i].kf.bbox for i in track_idx], np.float64)
        iou = _iou_matrix(db, tb)
        out = {}
        for r, c in _hungarian(np.where(iou >= self.cfg.minMatchingScore4TentativeIou, iou, -1.0)):
            if iou[r, c] >= self.cfg.minMatchingScore4TentativeIou:
                out[r] = track_idx[c]
        return out

    def _emit(self, t: _Target, det: ObjectMeta, out: list[ObjectMeta]) -> None:
        if t.state == TENTATIVE:
            det.track_id = None
        else:
            det.track_id = t.id
            if self.cfg.classAgnosticLabels:
                det.label = t.label
        out.append(det)

    def _update_filters(self, frame: VisualFrame) -> None:
        learn = [t for t in self._targets if t.state == ACTIVE or t.needs_init]
        if not learn:
            return
        windows = np.array([self._search_window(t.bbox) for t in learn], np.float32)
        slots = np.array([t.slot for t in learn], np.int32)
        init = np.array([1 if t.needs_init else 0 for t in learn], np.uint8)
        self.engine.update(frame, windows, slots, init, self.cfg.filterLr)
        for t in learn:
            t.needs_init = False

    # ------------------------------------------------------------ introspection
    @property
    def targets(self) -> list[_Target]:
        return list(self._targets)


def _hungarian(score: np.ndarray) -> list[tuple[int, int]]:
    """Max-score assignment; rows/cols with all -1 are simply unmatched."""
    if score.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-score)
        return [(int(a), int(b)) for a, b in zip(r, c)]
    except ImportError:
        pairs, ur, uc = [], set(), set()
        for r, c in np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]:
            if r in ur or c in uc:
                continue
            ur.add(int(r)); uc.add(int(c)); pairs.append((int(r), int(c)))
        return pairs
