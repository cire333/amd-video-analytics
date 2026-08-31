"""V1 pipeline orchestrator: registry -> batcher -> bridge -> graph -> tracker -> sink.

Single device, sequential over batch items. Everything above the decoder
is keyed by source_id (architecture §1).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .batcher import DynamicBatcher
from .bridge import HipBridge
from .capabilities import probe_devices, require_decode_device
from .frame import FrameMeta, ObjectMeta, RawFrame
from .graph import GraphExecutor, ModelGraph
from .registry import StreamRegistry
from .roi import RoiConfig
from .tracker import TrackerBank

log = logging.getLogger(__name__)

# Parses the detector node's raw output into ObjectMeta (model-input coords).
DetectionParser = Callable[[object], list[ObjectMeta]]
Sink = Callable[[FrameMeta], None]


class Pipeline:
    def __init__(
        self,
        graph: ModelGraph,
        detector_node: str,
        parse_detections: DetectionParser,
        sink: Sink,
        model_input_hw: tuple[int, int] = (640, 640),
        window_ms: int = 33,
        device_ordinal: int | None = None,
    ):
        devices = probe_devices()
        dev = (devices[device_ordinal] if device_ordinal is not None
               else require_decode_device(devices))
        log.info("pipeline on %s (%s, decode=%s)",
                 dev.drm_render_node, dev.gfx_generation, dev.has_decode_engine)

        self.device = dev
        self.registry = StreamRegistry(dev.drm_render_node, dev.device_ordinal)
        self.batcher = DynamicBatcher(window_ms=window_ms)
        self.bridge = HipBridge(dev.device_ordinal)
        self.executor = GraphExecutor(graph)
        self.detector_node = detector_node
        self.parse_detections = parse_detections
        self.tracker_bank = TrackerBank()
        self.sink = sink
        self.model_input_hw = model_input_hw
        self.rois: dict[str, RoiConfig] = {}

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- stream management (delegates; hot add/remove) ----------------------

    def add_stream(self, source_id: str, uri: str, roi: RoiConfig | None = None) -> None:
        if roi is not None:
            self.rois[source_id] = roi
        self.registry.add_stream(source_id, uri)

    def remove_stream(self, source_id: str) -> None:
        self.registry.remove_stream(source_id)
        self.tracker_bank.remove(source_id)
        self.rois.pop(source_id, None)

    # -- run loop ------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self.registry.stop_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self.batcher.assemble_batch(self.registry)
            if not batch:
                time.sleep(0.005)
                continue
            for frame in batch:
                try:
                    self._process_frame(frame)
                except Exception:
                    log.exception("[%s] frame processing failed", frame.source_id)

    def _process_frame(self, frame: RawFrame) -> None:
        conv = self.bridge.convert(
            frame, self.model_input_hw, roi=self.rois.get(frame.source_id)
        )
        results = self.executor.run(conv.tensor[None, ...])  # add batch dim
        detections = self.parse_detections(results[self.detector_node])

        # ROI gates inference input, NOT tracking: un-project to full frame
        # before the tracker sees anything (architecture §8).
        for det in detections:
            det.bbox = conv.transform.unproject(det.bbox)
        tracked = self.tracker_bank.update(frame.source_id, detections)

        self.sink(FrameMeta(
            source_id=frame.source_id,
            pts=frame.pts,
            frame_width=conv.frame_width,
            frame_height=conv.frame_height,
            objects=tracked,
        ))
