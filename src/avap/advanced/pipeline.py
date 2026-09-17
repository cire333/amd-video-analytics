"""AdvancedPipeline — the integrated executor (strategy + executor analog).

    pipe = AdvancedPipeline(
        primary=PrimaryStage(InferConfig(model="yolo26m", component_id=1,
                                         labels=COCO)),
        tracker="ocsort",
        stages=[plate_detect, plate_ocr, dinov2],   # SGIE chain, in order
        lpr=LPRStage(LPRConfig()),
        publisher=DataPublisher(FileTransport("./outbox")),
    )
    pipe.add_probe("tracker", my_probe)             # fn(frame_meta, frame_rgb)
    idx = pipe.add_source("rtsp://cam1/live", camera_id="IBG-C1011",
                          image_resolution=(1280, 720), fps=30,
                          metadata={"id": "IBG-C1011"})
    pipe.prepare()
    pipe.run()          # blocks until all file sources hit EOS / stop()

Contracts carried over from the DeepStream ingestion system:
- source identity is an integer index, recycled via a free list; the
  SourceContext carries camera_id / image_resolution / fps / metadata;
- published bboxes are rescaled from decode resolution to the source's
  declared image_resolution;
- one source's EOS or failure never stops the others;
- probes attach to named points ("primary", "tracker", each stage's name,
  "lpr") and receive (AdvFrameMeta, frame_rgb);
- records flow through the batched, lossy DataPublisher;
- process_image(index, rgb, pts_us) covers the image-directory use-case
  (no decoder involved).
Models are shared across sources; stage inference is serialized with a
lock (one MIGraphX context). Per-source trackers come from a factory so
track ids never cross cameras.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..capabilities import probe_devices, require_decode_device
from ..streaming import TRACKERS
from .lpr import LPRStage
from .meta import AdvFrameMeta
from .metrics import PerfData
from .publisher import DataPublisher
from .records import ClassMapper, DataRecord, build_object_record
from .stages import PrimaryStage, _Stage

log = logging.getLogger(__name__)

Probe = Callable[[AdvFrameMeta, np.ndarray], None]


@dataclass
class SourceContext:
    index: int
    uri: str
    camera_id: str
    image_resolution: tuple[int, int] | None = None   # publish coordinate space
    fps: float = 30.0
    batch_size_seconds: int = 30
    metadata: dict[str, Any] | None = None

    @property
    def batch_size(self) -> int:
        return int(round(self.batch_size_seconds * self.fps))


class AdvancedPipeline:
    def __init__(self, primary: PrimaryStage,
                 stages: list[_Stage] | None = None,
                 tracker: str | Callable = "ocsort",
                 lpr: LPRStage | None = None,
                 class_mapper: ClassMapper | None = None,
                 publisher: DataPublisher | None = None,
                 include_embedding: bool = False,
                 device_ordinal: int | None = None):
        self.primary = primary
        self.stages = list(stages or [])
        names = [self._stage_name(s) for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stage names/components: {names}")
        self._tracker_factory = (TRACKERS[tracker] if isinstance(tracker, str)
                                 else tracker)
        self.lpr = lpr
        self.class_mapper = class_mapper or ClassMapper()
        self.publisher = publisher
        self.include_embedding = include_embedding
        self.perf = PerfData()

        self._probes: dict[str, list[Probe]] = {}
        self._sources: dict[int, SourceContext] = {}
        self._trackers: dict[int, Any] = {}
        self._frame_no: dict[int, int] = {}
        self._free_indices: list[int] = []
        self._next_index = 0
        self._threads: dict[int, threading.Thread] = {}
        self._stop = threading.Event()
        self._infer_lock = threading.Lock()   # one MIGraphX context
        self._device = None
        self._device_ordinal = device_ordinal
        self._prepared = False

    # -- configuration ----------------------------------------------------------

    def _stage_name(self, stage: _Stage) -> str:
        return getattr(stage, "name", None) or f"stage_{stage.config.component_id}"

    def probe_points(self) -> list[str]:
        return ["primary", "tracker"] + [self._stage_name(s) for s in self.stages] \
            + (["lpr"] if self.lpr else [])

    def add_probe(self, point: str, fn: Probe) -> None:
        if point not in self.probe_points():
            raise ValueError(f"unknown probe point {point!r}; "
                             f"available: {self.probe_points()}")
        self._probes.setdefault(point, []).append(fn)

    def add_source(self, uri: str, camera_id: str,
                   image_resolution: tuple[int, int] | None = None,
                   fps: float = 30.0, batch_size_seconds: int = 30,
                   metadata: dict[str, Any] | None = None) -> int:
        index = (self._free_indices.pop() if self._free_indices
                 else self._next_index)
        if index == self._next_index:
            self._next_index += 1
        ctx = SourceContext(index, uri, camera_id, image_resolution, fps,
                            batch_size_seconds, metadata)
        self._sources[index] = ctx
        self._trackers[index] = self._tracker_factory()
        self._frame_no[index] = 0
        if self.publisher is not None:
            self.publisher.register_source(index, camera_id, ctx.batch_size)
        if self._prepared and not self._stop.is_set():
            self._start_source_thread(index)  # hot add on a running pipeline
        return index

    def remove_source(self, index: int) -> None:
        ctx = self._sources.pop(index, None)
        if ctx is None:
            return
        self._trackers.pop(index, None)
        self._frame_no.pop(index, None)
        thread = self._threads.pop(index, None)
        if self.publisher is not None:
            self.publisher.unregister_source(index)
        self._free_indices.append(index)
        log.info("[%s] source %d removed", ctx.camera_id, index)

    # -- lifecycle ---------------------------------------------------------------

    def prepare(self) -> None:
        """Compile all stage models. Call once before run()/process_image()."""
        if self._device_ordinal is None:
            self._device = require_decode_device(probe_devices())
            self._device_ordinal = self._device.device_ordinal
        else:
            self._device = next(d for d in probe_devices()
                                if d.device_ordinal == self._device_ordinal)
        for stage in [self.primary, *self.stages]:
            stage.device_ordinal = self._device_ordinal
            stage.prepare()
        if self.publisher is not None:
            self.publisher.enable()
        self._prepared = True

    def run(self) -> None:
        """Process all sources (a thread per source); blocks until every
        source finishes or stop() is called. One source failing or reaching
        EOS never affects the others."""
        if not self._prepared:
            self.prepare()
        self._stop.clear()
        for index in list(self._sources):
            self._start_source_thread(index)
        for thread in list(self._threads.values()):
            thread.join()
        if self.lpr is not None:
            self.lpr.save()
        if self.publisher is not None:
            self.publisher.disable()

    def stop(self) -> None:
        self._stop.set()

    # -- processing ----------------------------------------------------------------

    def _start_source_thread(self, index: int) -> None:
        t = threading.Thread(target=self._source_worker, args=(index,),
                             daemon=True,
                             name=f"avap-adv-{self._sources[index].camera_id}")
        self._threads[index] = t
        t.start()

    def _source_worker(self, index: int) -> None:
        from .. import _core
        ctx = self._sources[index]
        try:
            dec = _core.Decoder(ctx.uri, self._device.drm_render_node)
        except Exception:
            log.exception("[%s] failed to open source; other sources continue",
                          ctx.camera_id)
            return
        try:
            while not self._stop.is_set() and index in self._sources:
                f = dec.next_frame()
                if f is None:
                    log.info("[%s] EOS", ctx.camera_id)
                    return
                rgb = self._frame_to_rgb(f)
                self.process_image(index, rgb, f.pts_us)
        except Exception:
            log.exception("[%s] source failed; other sources continue",
                          ctx.camera_id)
        finally:
            dec.close()

    def _frame_to_rgb(self, f) -> np.ndarray:
        """Full-resolution HWC uint8 RGB from a decoded frame."""
        from .. import _core
        planes = [tuple(p) for p in f.planes]
        src = (f.crop_x, f.crop_y, f.crop_w, f.crop_h)
        args = (planes, src, (f.crop_w, f.crop_h), f.full_range,
                f.color_matrix != "bt601", self._device_ordinal)
        with self._infer_lock:
            chw = (_core.nv12_dmabuf_to_rgb(f.dmabuf_fd, f.width, f.height,
                                            planes, f.drm_modifier, *args[1:])
                   if f.dmabuf_fd >= 0
                   else _core.nv12_host_to_rgb(f.host_data, *args))
        return (np.clip(chw, 0.0, 1.0) * 255).astype(np.uint8).transpose(1, 2, 0)

    def process_image(self, index: int, rgb: np.ndarray, pts_us: int = 0) -> AdvFrameMeta:
        """Run the full stage graph over one RGB frame (HWC uint8) for the
        given source. Used by the decode workers and directly for
        image-directory workloads."""
        ctx = self._sources[index]
        frame_no = self._frame_no[index]
        self._frame_no[index] = frame_no + 1
        frame = AdvFrameMeta(source_id=ctx.camera_id, frame=frame_no,
                             pts_us=pts_us, frame_width=rgb.shape[1],
                             frame_height=rgb.shape[0])

        with self._infer_lock:
            self.primary.run(rgb, frame)
        self._fire_probes("primary", frame, rgb)

        self._trackers[index].update(frame.primary_objects())
        self._fire_probes("tracker", frame, rgb)

        for stage in self.stages:
            with self._infer_lock:
                stage.run(rgb, frame)
            self._fire_probes(self._stage_name(stage), frame, rgb)

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.lpr is not None:
            self.lpr.process(frame, timestamp)
            self._fire_probes("lpr", frame, rgb)

        self.perf.update_fps(ctx.camera_id)
        if self.publisher is not None:
            self._publish(index, ctx, frame)
        return frame

    def _fire_probes(self, point: str, frame: AdvFrameMeta,
                     rgb: np.ndarray) -> None:
        for fn in self._probes.get(point, ()):
            try:
                fn(frame, rgb)
            except Exception:
                log.exception("probe %r raised; ignoring", point)

    def _publish(self, index: int, ctx: SourceContext,
                 frame: AdvFrameMeta) -> None:
        # rescale from decode resolution to the source's declared resolution
        if ctx.image_resolution:
            sx = ctx.image_resolution[0] / frame.frame_width
            sy = ctx.image_resolution[1] / frame.frame_height
        else:
            sx = sy = 1.0
        objects = []
        for obj in frame.primary_objects():
            lpr = (self.lpr.get_result(obj.object_id, ctx.camera_id)
                   if self.lpr is not None and obj.object_id is not None
                   else None)
            rec = build_object_record(obj, float(frame.pts_us),
                                      self.class_mapper, lpr,
                                      self.include_embedding)
            if rec is None:
                continue
            rec.x1 = round(rec.x1 * sx, 2)
            rec.x2 = round(rec.x2 * sx, 2)
            rec.y1 = round(rec.y1 * sy, 2)
            rec.y2 = round(rec.y2 * sy, 2)
            objects.append(rec)
        self.publisher.publish(index, DataRecord(
            data=objects, frame=frame.frame, stream=index,
            time=str(int(time.time() * 1000)),
            fps=self.perf.get(ctx.camera_id).fps(),
            metadata=ctx.metadata))
