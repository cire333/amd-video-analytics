"""MuxedPipeline — the DeepStream shape: N decoders -> StreamMux -> canvas -> batched detector.

    decoders (one thread per source, VCN/VAAPI)  --push_frame-->  StreamMux
    StreamMuxRunner thread:  MuxBatch -> BatchCanvas ([N,3,H,W] on the GPU)
                             -> detector(batch) -> parse -> unproject -> tracker -> sink

Compared with ``avap.pipeline.Pipeline`` (per-frame, per-source inference)
this is the nvstreammux-equivalent path: one fixed-shape batch per inference
call, sources scaled/letterboxed onto a common canvas, DeepStream's
NvDsFrameMeta fields carried per frame (``FrameMeta.mux``).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .canvas import BatchCanvas, CanvasTransform
from .capabilities import probe_devices, require_decode_device
from .frame import ColorMatrix, ColorRange, CropRect, FrameMeta, ObjectMeta, PlaneLayout, RawFrame
from .nvdcf import VisualFrame
from .roi import RoiConfig
from .streammux import MuxBatch, MuxConfig, MuxEvent, MuxFrameMeta, StreamMux, StreamMuxRunner
from .tracker import TrackerBank

log = logging.getLogger(__name__)

# detector(batch [N,3,H,W] float32) -> list per slot of (x1,y1,x2,y2,score,cls) rows
BatchDetector = Callable[[np.ndarray], np.ndarray]
Sink = Callable[[FrameMeta], None]


@dataclass
class MuxedFrameMeta(FrameMeta):
    """FrameMeta + the nvstreammux fields."""
    mux: MuxFrameMeta | None = None
    batch_index: int = -1
    canvas: CanvasTransform | None = None
    objects: list[ObjectMeta] = field(default_factory=list)


class _SourceThread(threading.Thread):
    """Decode one URI and push frames into the mux pad (EOS at the end)."""

    def __init__(self, pipeline: "MuxedPipeline", pad: int, source_id: str, uri: str):
        super().__init__(name=f"decode-{source_id}", daemon=True)
        self.p = pipeline
        self.pad = pad
        self.source_id = source_id
        self.uri = uri
        self.frames = 0
        self.error: str | None = None

    def run(self) -> None:
        from . import _core
        p = self.p
        try:
            dec = _core.Decoder(self.uri, p.device.drm_render_node)
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            log.exception("[%s] open failed", self.source_id)
            p.mux.push_eos(self.pad)
            return
        try:
            while not p._stop.is_set():
                f = dec.next_frame()
                if f is None:
                    break
                raw = RawFrame(
                    source_id=self.source_id, pts=f.pts_us,
                    recv_us=time.monotonic_ns() // 1000, width=f.width, height=f.height,
                    crop=CropRect(f.crop_x, f.crop_y, f.crop_w, f.crop_h),
                    dmabuf_fd=f.dmabuf_fd,
                    planes=tuple(PlaneLayout(o, pp) for o, pp in f.planes),
                    drm_modifier=f.drm_modifier,
                    color_range=ColorRange.FULL if f.full_range else ColorRange.LIMITED,
                    color_matrix={"bt601": ColorMatrix.BT601,
                                  "bt709": ColorMatrix.BT709}.get(f.color_matrix,
                                                                  ColorMatrix.UNKNOWN),
                    device_ordinal=p.device.device_ordinal, host_data=f.host_data)
                p.mux.push_frame(self.pad, raw, f.pts_us * 1000, f.crop_w, f.crop_h)
                self.frames += 1
                if p.pace_fps:
                    time.sleep(1.0 / p.pace_fps)
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            log.exception("[%s] decode failed", self.source_id)
        finally:
            dec.close()
            p.mux.push_eos(self.pad)


class MuxedPipeline:
    def __init__(self, detector: BatchDetector, mux_cfg: MuxConfig, sink: Sink,
                 conf_threshold: float = 0.3, device_ordinal: int | None = None,
                 tracker_bank: TrackerBank | None = None, pace_fps: float | None = None,
                 labels: list[str] | None = None):
        if mux_cfg.width <= 0 or mux_cfg.height <= 0:
            raise ValueError("mux_cfg.width/height (canvas) must be set, like nvstreammux")
        devices = probe_devices()
        self.device = (devices[device_ordinal] if device_ordinal is not None
                       else require_decode_device(devices))
        self.mux_cfg = mux_cfg
        self.mux = StreamMux(mux_cfg)
        self.canvas = BatchCanvas(mux_cfg.batch_size, mux_cfg.width, mux_cfg.height,
                                  self.device.device_ordinal, mux_cfg.enable_padding,
                                  mux_cfg.interpolation)
        self.detector = detector
        self.sink = sink
        self.conf_threshold = conf_threshold
        self.tracker_bank = tracker_bank
        self.pace_fps = pace_fps
        self.labels = labels or []
        self.rois: dict[str, RoiConfig] = {}
        self.sources: dict[int, _SourceThread] = {}
        self._pad_to_source: dict[int, str] = {}
        self._stop = threading.Event()
        self._runner = StreamMuxRunner(self.mux, self._on_batch, self._on_event)
        self.events: list[MuxEvent] = []
        self.stats = {"batches": 0, "frames": 0, "infer_s": 0.0, "canvas_s": 0.0,
                      "start": 0.0, "end": 0.0}
        self.done = threading.Event()

    # ------------------------------------------------------------------ sources
    def add_source(self, source_id: str, uri: str, roi: RoiConfig | None = None) -> int:
        pad = len(self._pad_to_source)
        self._pad_to_source[pad] = source_id
        if roi is not None:
            self.rois[source_id] = roi
        self.mux.add_pad(pad)
        self.sources[pad] = _SourceThread(self, pad, source_id, uri)
        return pad

    def start(self) -> None:
        self.stats["start"] = time.monotonic()
        self.mux.start()
        self._runner.start()
        for s in self.sources.values():
            s.start()

    def stop(self) -> None:
        self._stop.set()
        for s in self.sources.values():
            s.join(timeout=10.0)
        self._runner.stop()
        self.canvas.close()
        self.stats["end"] = time.monotonic()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every source hit EOS and the mux emitted its EOS."""
        return self.done.wait(timeout)

    # ------------------------------------------------------------------ batch path
    def _on_event(self, ev: MuxEvent) -> None:
        self.events.append(ev)
        if ev.kind == "eos":
            self.done.set()

    def _on_batch(self, batch: MuxBatch) -> None:
        t0 = time.monotonic()
        transforms = self.canvas.render(batch, self.rois)
        tensor = self.canvas.to_host()          # v1: host round-trip into the model
        t1 = time.monotonic()
        out = self.detector(tensor)             # (N, K, 6)
        t2 = time.monotonic()
        self.stats["canvas_s"] += t1 - t0
        self.stats["infer_s"] += t2 - t1
        self.stats["batches"] += 1
        self.stats["frames"] += len(batch.frames)
        for slot, (meta, tf) in enumerate(zip(batch.frames, transforms)):
            dets = []
            for x1, y1, x2, y2, s, c in out[slot]:
                if s < self.conf_threshold:
                    continue
                bbox = tf.unproject_clipped((float(x1), float(y1), float(x2), float(y2)))
                cid = int(c)
                dets.append(ObjectMeta(class_id=cid, confidence=float(s), bbox=bbox,
                                       label=self.labels[cid] if cid < len(self.labels) else str(cid)))
            src = self._pad_to_source[meta.pad_index]
            if self.tracker_bank:
                # visual trackers (NvDCF) read pixels straight from the canvas slot
                sx, sy = tf.dst_w / tf.src_w, tf.dst_h / tf.src_h
                vframe = VisualFrame(self.canvas.slot_ptr(slot), self.canvas.width,
                                     self.canvas.height, sx, sy,
                                     tf.dst_x - tf.src_x * sx, tf.dst_y - tf.src_y * sy)
                objects = self.tracker_bank.update(src, dets, vframe)
            else:
                objects = dets
            fm = MuxedFrameMeta(source_id=src, pts=meta.buf_pts // 1000,
                                frame_width=tf.frame_width, frame_height=tf.frame_height,
                                objects=objects, mux=meta, batch_index=batch.batch_index,
                                canvas=tf)
            try:
                self.sink(fm)
            except Exception:
                log.exception("sink failed")
