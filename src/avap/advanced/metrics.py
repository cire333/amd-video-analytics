"""Per-stream FPS metrics — functional port of the ingestion system's
GETFPS/PERF_DATA (cumulative average since first frame; lazily created so
dynamically added streams work)."""
from __future__ import annotations

import time
from threading import Lock


class StreamFps:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self._lock = Lock()
        self._start: float | None = None
        self._frames = 0
        self._last_frame_ts = 0.0

    def update(self) -> None:
        now = time.time()
        with self._lock:
            if self._start is None:
                self._start = now
            self._frames += 1
            self._last_frame_ts = now

    def fps(self) -> float:
        with self._lock:
            if self._start is None:
                return 0.0
            elapsed = time.time() - self._start
            if elapsed < 1.0:
                return 0.0
            return round(self._frames / elapsed, 2)

    def seconds_since_last_frame(self) -> float | None:
        with self._lock:
            if self._last_frame_ts == 0.0:
                return None
            return time.time() - self._last_frame_ts


class PerfData:
    def __init__(self):
        self._streams: dict[str, StreamFps] = {}
        self._lock = Lock()

    def update_fps(self, stream_id: str) -> None:
        self.get(stream_id).update()

    def get(self, stream_id: str) -> StreamFps:
        with self._lock:
            if stream_id not in self._streams:  # lazily created
                self._streams[stream_id] = StreamFps(stream_id)
            return self._streams[stream_id]

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            streams = dict(self._streams)
        return {sid: {"fps": s.fps(),
                      "idle_s": s.seconds_since_last_frame()}
                for sid, s in streams.items()}
