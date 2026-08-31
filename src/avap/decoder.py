"""Per-stream decoder worker (architecture §4).

One thread per source, owning a C++ VAAPI decoder (FFmpeg demux/parse ->
VCN decode -> dmabuf export). Reconnect with exponential backoff lives
here so the registry never blocks on a bad stream.
"""
from __future__ import annotations

import logging
import threading
import time

from .frame import ColorMatrix, ColorRange, CropRect, PlaneLayout, RawFrame
from .ringbuffer import RingBuffer

log = logging.getLogger(__name__)

BACKOFF_INITIAL_S = 1.0
BACKOFF_MAX_S = 60.0


class DecoderWorker:
    def __init__(self, source_id: str, uri: str, render_node: str, device_ordinal: int,
                 queue_capacity: int = 4):
        self.source_id = source_id
        self.uri = uri
        self.render_node = render_node
        self.device_ordinal = device_ordinal
        self.frame_queue = RingBuffer(capacity=queue_capacity)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"decode-{self.source_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.frame_queue.clear()

    # -- internals ---------------------------------------------------------

    def _run(self) -> None:
        backoff = BACKOFF_INITIAL_S
        while not self._stop.is_set():
            try:
                self._decode_until_error()
                backoff = BACKOFF_INITIAL_S  # clean EOF/disconnect: quick retry
            except Exception:
                log.exception("[%s] decoder error; retrying in %.1fs", self.source_id, backoff)
            self.connected = False
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, BACKOFF_MAX_S)

    def _decode_until_error(self) -> None:
        from . import _core  # deferred: pure-Python parts importable without the ext

        dec = _core.Decoder(self.uri, self.render_node)
        self.connected = True
        log.info("[%s] connected: %s", self.source_id, self.uri)
        try:
            while not self._stop.is_set():
                f = dec.next_frame()  # C++ struct; None on EOF
                if f is None:
                    log.info("[%s] end of stream", self.source_id)
                    return
                self.frame_queue.push(self._to_raw_frame(f))
        finally:
            dec.close()

    def _to_raw_frame(self, f) -> RawFrame:
        return RawFrame(
            source_id=self.source_id,
            pts=f.pts_us,
            recv_us=time.monotonic_ns() // 1000,
            width=f.width,
            height=f.height,
            crop=CropRect(f.crop_x, f.crop_y, f.crop_w, f.crop_h),
            dmabuf_fd=f.dmabuf_fd,
            planes=tuple(PlaneLayout(o, p) for o, p in f.planes),
            drm_modifier=f.drm_modifier,
            color_range=ColorRange.FULL if f.full_range else ColorRange.LIMITED,
            color_matrix={"bt601": ColorMatrix.BT601, "bt709": ColorMatrix.BT709}.get(
                f.color_matrix, ColorMatrix.UNKNOWN
            ),
            device_ordinal=self.device_ordinal,
        )
