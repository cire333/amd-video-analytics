"""Batched, decoupled record publishing — functional port of the
ingestion system's DataPublisher:

- bounded queue + one worker thread; publish() never blocks the pipeline
  (a full queue DROPS the record and counts queue_full — lossy by design);
- per-source batches flushed on batch_size records OR batch_timeout
  seconds (timeout evaluated when new records arrive, like upstream);
- payload = JSON array of DataRecord dicts;
- transports are callables (payload: str, camera_id: str) -> None, so any
  avap sink or user code can be the destination. FileTransport reproduces
  the upstream file publisher's json layout
  (<out>/source_<id>/batch_<ts>_<seq>.json).
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .records import DataRecord

Transport = Callable[[str, str], None]


class FileTransport:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self._seq: dict[str, int] = {}

    def __call__(self, payload: str, camera_id: str) -> None:
        d = self.output_dir / f"source_{camera_id}"
        d.mkdir(parents=True, exist_ok=True)
        seq = self._seq.get(camera_id, 0)
        self._seq[camera_id] = seq + 1
        (d / f"batch_{int(time.time() * 1000)}_{seq}.json").write_text(payload)


@dataclass
class _BatchSource:
    camera_id: str
    batch_size: int
    batch_timeout: float
    records: list[dict] = field(default_factory=list)
    last_flush: float = field(default_factory=time.time)
    frames_enqueued: int = 0
    batches_sent: int = 0
    failed_batches: int = 0

    def should_flush(self) -> bool:
        return (len(self.records) >= self.batch_size
                or (self.records
                    and time.time() - self.last_flush >= self.batch_timeout))


class DataPublisher:
    def __init__(self, transport: Transport, max_size: int = 100):
        self._transport = transport
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self._sources: dict[int, _BatchSource] = {}
        self._stats = {"sent": 0, "failed": 0, "queue_full": 0}
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    def register_source(self, source_index: int, camera_id: str,
                        batch_size: int, batch_timeout: float = 10.0) -> None:
        self._sources[source_index] = _BatchSource(camera_id, batch_size,
                                                   batch_timeout)

    def unregister_source(self, source_index: int) -> None:
        src = self._sources.get(source_index)
        if src is not None:
            self._flush(src)
            del self._sources[source_index]

    def enable(self) -> None:
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="avap-publisher")
        self._worker.start()

    def disable(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout)
        for src in self._sources.values():  # final flush
            self._flush(src)

    def publish(self, source_index: int, record: DataRecord) -> None:
        try:
            self._queue.put_nowait((source_index, record))
        except queue.Full:
            self._stats["queue_full"] += 1

    def stats(self) -> dict[str, Any]:
        per_source = {i: {"camera_id": s.camera_id,
                          "frames_enqueued": s.frames_enqueued,
                          "batches_sent": s.batches_sent,
                          "failed_batches": s.failed_batches}
                      for i, s in self._sources.items()}
        return {**self._stats, "sources": per_source}

    # -- worker ----------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                source_index, record = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            src = self._sources.get(source_index)
            if src is None:
                continue
            src.records.append(record.to_dict())
            src.frames_enqueued += 1
            if src.should_flush():
                self._flush(src)

    def _flush(self, src: _BatchSource) -> None:
        if not src.records:
            return
        payload = json.dumps(src.records)
        batch, src.records = src.records, []
        src.last_flush = time.time()
        try:
            self._transport(payload, src.camera_id)
            src.batches_sent += 1
            self._stats["sent"] += 1
        except Exception:
            src.failed_batches += 1
            self._stats["failed"] += 1
