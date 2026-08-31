"""Stream registry (architecture §3): mutable and polled, not a static graph.

Adding/removing a source is a registry update the batcher picks up next
cycle — no pipeline renegotiation.
"""
from __future__ import annotations

import threading

from .decoder import DecoderWorker


class StreamHandle:
    def __init__(self, source_id: str, uri: str, render_node: str, device_ordinal: int):
        self.source_id = source_id
        self.uri = uri
        self.worker = DecoderWorker(source_id, uri, render_node, device_ordinal)

    @property
    def frame_queue(self):
        return self.worker.frame_queue

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()


class StreamRegistry:
    def __init__(self, render_node: str, device_ordinal: int):
        self._default_node = render_node
        self._default_ordinal = device_ordinal
        self._streams: dict[str, StreamHandle] = {}
        self._lock = threading.RLock()

    def add_stream(self, source_id: str, uri: str,
                   render_node: str | None = None, device_ordinal: int | None = None) -> None:
        with self._lock:
            if source_id in self._streams:
                raise ValueError(f"source_id already registered: {source_id}")
            handle = StreamHandle(
                source_id, uri,
                render_node or self._default_node,
                self._default_ordinal if device_ordinal is None else device_ordinal,
            )
            handle.start()
            self._streams[source_id] = handle

    def remove_stream(self, source_id: str) -> None:
        with self._lock:
            self._streams.pop(source_id).stop()

    def snapshot(self) -> list[StreamHandle]:
        """Cheap copy; the batcher iterates without holding the lock."""
        with self._lock:
            return list(self._streams.values())

    def stop_all(self) -> None:
        with self._lock:
            for handle in self._streams.values():
                handle.stop()
            self._streams.clear()
