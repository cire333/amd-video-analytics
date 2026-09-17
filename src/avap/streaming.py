"""User-facing streaming API: AMDStream + AMDGPUManager.

    stream = AMDStream(
        data_location="rtsp://cam1/live",        # local path, s3://, rtsp://, http(s)://
        region_of_interest=[(0.1, 0.2), (0.9, 0.2), (0.9, 0.9), (0.1, 0.9)],
        model="yolo26m",                          # zoo name or path to custom .onnx
        model_quant="fp16",                       # fp32 | fp16 | int8
        tracker_type="bytetrack",                 # iou | sort | bytetrack | TrackerProtocol instance
        output_location="kafka://broker:9092/detections",
        batch_size=1,                             # 1 = realtime
        output_format_template=None,              # optional str.format template
        output_format="json",                     # json | csv | parquet
        frame_sample_rate=5,                      # process at most N fps (None = all)
    )
    stream.start_stream()

    mgr = AMDGPUManager(device_id=0)
    mgr.add_stream(stream)
    mgr.start_streams()   # sequential; a stream that fails to start is
                          # logged and does not affect running streams

The ROI is applied close to the hardware: its bounding rect is fused into
the NV12->RGB conversion kernel, so out-of-region pixels never leave the
decode surface (architecture §8); the polygon mask is applied to the
model-input tensor.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from typing import Callable

import numpy as np

from .capabilities import DeviceCapabilities, probe_devices
from .events import EventCallback, RetryPolicy, StreamEvent
from .frame import ObjectMeta
from .bytetrack import ByteTracker
from .kalman_tracker import SortTracker
from .ocsort import OcSortTracker
from .model_zoo import QUANT_MODES, MigraphxModel, resolve_model
from .roi import RoiConfig
from .sinks import Sink, frame_record, make_sink
from .tracker import IouTracker, TrackerProtocol

log = logging.getLogger(__name__)

TRACKERS: dict[str, Callable[[], TrackerProtocol]] = {
    "iou": IouTracker,
    "sort": SortTracker,
    "bytetrack": ByteTracker,
    "ocsort": OcSortTracker,
}

COCO = None  # filled lazily from model zoo label list


def _coco_labels() -> list[str]:
    global COCO
    if COCO is None:
        from .streaming_labels import COCO_LABELS
        COCO = COCO_LABELS
    return COCO



def resolve_data_location(data_location: str) -> str:
    """Local path / rtsp / http(s) pass through to the decoder; s3://
    objects are downloaded to a temp file first."""
    if data_location.startswith(("rtsp://", "http://", "https://")):
        return data_location
    if data_location.startswith("s3://"):
        import boto3
        bucket, _, key = data_location[len("s3://"):].partition("/")
        fd, path = tempfile.mkstemp(suffix=os.path.splitext(key)[1] or ".mp4")
        os.close(fd)
        log.info("downloading %s -> %s", data_location, path)
        boto3.client("s3").download_file(bucket, key, path)
        return path
    if not os.path.exists(data_location):
        raise FileNotFoundError(f"data_location not found: {data_location}")
    return data_location


def _make_roi(region_of_interest) -> RoiConfig | None:
    """Accept a polygon [(x,y), ...] or a bbox (x1, y1, x2, y2), normalized 0-1."""
    if region_of_interest is None:
        return None
    if isinstance(region_of_interest, RoiConfig):
        return region_of_interest
    r = list(region_of_interest)
    if len(r) == 4 and all(isinstance(v, (int, float)) for v in r):
        x1, y1, x2, y2 = r
        r = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return RoiConfig(r)


def _make_tracker(tracker_type) -> TrackerProtocol:
    if isinstance(tracker_type, str):
        if tracker_type not in TRACKERS:
            raise ValueError(f"tracker_type must be one of {sorted(TRACKERS)} "
                             "or a TrackerProtocol instance")
        return TRACKERS[tracker_type]()
    if hasattr(tracker_type, "update"):
        return tracker_type  # bring-your-own
    raise TypeError("tracker_type: str name or object with .update(detections)")


class AMDStream:
    """One video source -> decode -> ROI-fused convert -> model -> tracker -> sink.

    Everything is validated/configured at construction; GPU/network
    resources are acquired in start_stream().
    """

    def __init__(self, data_location: str, region_of_interest=None,
                 model: str = "yolo26m", model_quant: str = "fp16",
                 tracker_type="sort", output_location: str = "detections.jsonl",
                 batch_size: int = 1, output_format_template: str | None = None,
                 output_format: str = "json", frame_sample_rate: float | None = None,
                 conf_threshold: float = 0.3, source_id: str | None = None,
                 device: DeviceCapabilities | None = None, imgsz: int = 640,
                 retry_policy: RetryPolicy | None = None,
                 on_event: EventCallback | None = None):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if frame_sample_rate is not None and frame_sample_rate <= 0:
            raise ValueError("frame_sample_rate must be > 0 fps")
        if model_quant not in QUANT_MODES:
            raise ValueError(f"model_quant must be one of {QUANT_MODES}")
        self.source_id = source_id or os.path.basename(data_location) or "stream"
        self.data_location = data_location
        self.roi = _make_roi(region_of_interest)
        self.model_name = model
        self.model_quant = model_quant
        self.tracker = _make_tracker(tracker_type)
        self.output_location = output_location
        self.batch_size = batch_size
        self.output_format = output_format
        self.output_format_template = output_format_template
        self.frame_sample_rate = frame_sample_rate
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device

        self.retry_policy = retry_policy  # None -> manager default or RetryPolicy()
        self.on_event = on_event

        self.state = "configured"   # -> starting -> running -> stopped/failed/eof
        self.frames_processed = 0
        self.restarts = 0           # successful reattachments
        self.sink_errors = 0        # records dropped due to output failures
        self.last_error: str | None = None
        self._model: MigraphxModel | None = None
        self._sink: Sink | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------

    def start_stream(self) -> None:
        """Acquire resources (raises on failure) and start processing."""
        self.state = "starting"
        try:
            if self.device is None:
                amd = [d for d in probe_devices() if d.has_decode_engine]
                if not amd:
                    raise RuntimeError("no AMD device with a decode engine found")
                self.device = amd[0]
            self._uri = resolve_data_location(self.data_location)
            onnx = resolve_model(self.model_name, self.batch_size, self.imgsz)
            self._model = MigraphxModel(onnx, self.model_quant,
                                        self.device.device_ordinal)
            self._sink = make_sink(self.output_location, self.output_format,
                                   self.output_format_template)
            from . import _core
            self._decoder_factory = lambda: _core.Decoder(
                self._uri, self.device.drm_render_node)
            # Files fail fast (an unopenable file never heals); live network
            # sources skip the probe — a camera that's down at startup goes
            # through the same retry cycle + events as one that drops later.
            if not self._uri.startswith(("rtsp://", "http://", "https://")):
                dec = self._decoder_factory()
                dec.close()
        except Exception:
            self.state = "failed"
            raise
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"avap-{self.source_id}")
        self._thread.start()
        self.state = "running"

    def stop_stream(self) -> None:
        self._stop.set()
        dec = getattr(self, "_active_decoder", None)
        if dec is not None:
            dec.abort()  # unblock a thread stuck in network I/O immediately
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self._sink is not None:
            self._sink.close()
            self._sink = None
        if self.state == "running":
            self.state = "stopped"

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
            if not self._thread.is_alive() and self._sink is not None:
                self._sink.close()
                self._sink = None

    # -- processing ----------------------------------------------------------

    def _emit_event(self, type_: str, **kw) -> None:
        """Report a lifecycle event; a broken reporter never kills the stream."""
        event = StreamEvent(type=type_, source_id=self.source_id,
                            frames_processed=self.frames_processed, **kw)
        log.info("[%s] %s%s", self.source_id, type_,
                 f" ({event.error})" if event.error else "")
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                log.exception("[%s] on_event callback raised; ignoring",
                              self.source_id)

    def _run(self) -> None:
        is_live = self._uri.startswith(("rtsp://", "http://", "https://"))
        policy = self.retry_policy or RetryPolicy()
        attempt = 0
        while not self._stop.is_set():
            connected_at = time.monotonic()
            try:
                self._emit_event("connecting", attempt=attempt)
                self._process_source()  # emits "connected" once frames flow
                if not is_live:
                    self.state = "eof"
                    self._emit_event("eof")
                    return
                self.last_error = None
                self._emit_event("disconnected")  # live source ended cleanly
            except Exception as e:
                self.last_error = repr(e)
                log.exception("[%s] stream error", self.source_id)
                self._emit_event("disconnected", error=self.last_error)
            if self._stop.is_set():
                return
            # a connection that held for reset_after_s starts a fresh cycle
            if time.monotonic() - connected_at >= policy.reset_after_s:
                attempt = 0
            attempt += 1
            if policy.max_retries is not None and attempt > policy.max_retries:
                self.state = "failed"
                self._emit_event("gave_up", attempt=attempt - 1,
                                 error=self.last_error)
                return
            backoff = policy.backoff(attempt)
            self._emit_event("reconnecting", attempt=attempt, backoff_s=backoff)
            if self._stop.wait(backoff):
                return
            self.restarts += 1

    def _process_source(self) -> None:
        from . import _core
        dec = self._decoder_factory()
        self._active_decoder = dec
        self._emit_event("connected")
        sample_period_us = (None if self.frame_sample_rate is None
                            else 1_000_000 / self.frame_sample_rate)
        next_sample_us = -1.0
        batch, meta = [], []
        try:
            while not self._stop.is_set():
                f = dec.next_frame()
                if f is None:
                    break
                if sample_period_us is not None:
                    if f.pts_us < next_sample_us:
                        if f.dmabuf_fd >= 0:
                            os.close(f.dmabuf_fd)
                        continue
                    # advance on a fixed grid (not pts + period) so the
                    # effective rate matches the requested rate
                    next_sample_us = max(next_sample_us + sample_period_us,
                                         f.pts_us + 1)

                tensor, transform = self._convert(f)
                if self._model.quant == "int8" and not self._model.ready:
                    self._model.calibrate(tensor[None])
                    continue
                batch.append(tensor)
                meta.append((self.frames_processed, f.pts_us, transform))
                self.frames_processed += 1
                if len(batch) >= self.batch_size:
                    self._infer_and_emit(batch, meta)
                    batch, meta = [], []
            if batch:  # partial final batch: pad, run, truncate
                pad = self.batch_size - len(batch)
                padded = batch + [np.zeros_like(batch[0])] * pad
                self._infer_and_emit(padded, meta)
        finally:
            self._active_decoder = None
            dec.close()

    def _convert(self, f):
        """ROI-fused NV12->RGB at model input size; returns (CHW, transform)."""
        fw, fh = f.crop_w, f.crop_h
        if self.roi is not None:
            rx, ry, rw, rh = self.roi.crop_rect_px(fw, fh)
            src = (f.crop_x + rx, f.crop_y + ry, rw, rh)
        else:
            rx = ry = 0
            src = (f.crop_x, f.crop_y, fw, fh)
        from . import _core
        planes = [tuple(p) for p in f.planes]
        args = (planes, src, (self.imgsz, self.imgsz), f.full_range,
                f.color_matrix != "bt601", self.device.device_ordinal)
        if f.dmabuf_fd >= 0:
            chw = _core.nv12_dmabuf_to_rgb(f.dmabuf_fd, f.width, f.height,
                                           planes, f.drm_modifier, *args[1:])
        else:
            chw = _core.nv12_host_to_rgb(f.host_data, *args)
        if self.roi is not None:
            mask = self.roi.mask(fw, fh)
            import cv2
            m = cv2.resize(mask, (self.imgsz, self.imgsz),
                           interpolation=cv2.INTER_NEAREST)
            chw = chw * m[None]
        # transform: model-input px -> full-frame px
        return chw, (rx, ry, src[2] / self.imgsz, src[3] / self.imgsz)

    def _infer_and_emit(self, batch: list[np.ndarray], meta: list) -> None:
        out = self._model(np.stack(batch))  # (B, 300, 6)
        labels = _coco_labels()
        for bi, (frame_idx, pts_us, (ox, oy, sx, sy)) in enumerate(meta):
            dets = [
                ObjectMeta(class_id=int(c), confidence=float(s),
                           bbox=(float(x1) * sx + ox, float(y1) * sy + oy,
                                 float(x2) * sx + ox, float(y2) * sy + oy),
                           label=(labels[int(c)] if int(c) < len(labels)
                                  else str(int(c))))
                for x1, y1, x2, y2, s, c in out[bi]
                if s >= self.conf_threshold
            ]
            tracked = self.tracker.update(dets)
            rec = frame_record(
                self.source_id, frame_idx, pts_us,
                [{"track_id": o.track_id, "label": o.label,
                  "class_id": o.class_id, "conf": round(o.confidence, 4),
                  "bbox": [round(v, 1) for v in o.bbox]} for o in tracked])
            try:
                self._sink.emit(rec)
            except Exception as e:
                # an output outage must not stall decode/inference: drop the
                # record, report it, keep the stream alive. The callback can
                # decide to stop_stream() if data loss is unacceptable.
                self.sink_errors += 1
                self._emit_event("sink_error", error=repr(e))


class AMDGPUManager:
    """Runs multiple AMDStreams on one device. Streams start sequentially;
    a stream that fails to start is logged and marked failed, later streams
    are left pending, and already-running streams are unaffected."""

    def __init__(self, device_id: int = 0, config: dict | None = None):
        # config keys: start_stagger_s, stop_on_failure, on_event (default
        # StreamEvent reporter for streams that don't set their own),
        # retry_policy (default RetryPolicy likewise)
        self.config = {"start_stagger_s": 0.5, "stop_on_failure": True,
                       "on_event": None, "retry_policy": None,
                       **(config or {})}
        amd = [d for d in probe_devices() if d.has_decode_engine]
        if device_id >= len(amd):
            raise ValueError(f"device_id {device_id} not found; "
                             f"{len(amd)} AMD decode-capable device(s) present")
        self.device = amd[device_id]
        self.streams: list[AMDStream] = []

    def add_stream(self, stream: AMDStream) -> None:
        stream.device = self.device
        if stream.on_event is None and self.config["on_event"] is not None:
            stream.on_event = self.config["on_event"]
        if stream.retry_policy is None:
            stream.retry_policy = self.config["retry_policy"]
        self.streams.append(stream)

    def start_streams(self) -> None:
        for stream in self.streams:
            if stream.state != "configured":
                continue
            try:
                stream.start_stream()
                log.info("[%s] started (%s, %s, %s)", stream.source_id,
                         stream.model_name, stream.model_quant,
                         type(stream.tracker).__name__)
            except Exception:
                log.exception(
                    "[%s] failed to start; device may be at capacity — "
                    "%d stream(s) keep running, later streams left pending",
                    stream.source_id,
                    sum(1 for s in self.streams if s.state == "running"))
                if self.config["stop_on_failure"]:
                    break
            time.sleep(self.config["start_stagger_s"])

    def stop_streams(self) -> None:
        for stream in self.streams:
            if stream.state == "running":
                stream.stop_stream()

    def status(self) -> dict[str, dict]:
        return {s.source_id: {"state": s.state,
                              "frames_processed": s.frames_processed,
                              "restarts": s.restarts,
                              "sink_errors": s.sink_errors,
                              "last_error": s.last_error}
                for s in self.streams}
