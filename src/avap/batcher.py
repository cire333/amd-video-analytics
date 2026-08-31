"""Dynamic batcher (architecture §6).

Timestamp-windowed, variable-length, variable-shape batches. A stalled
source contributes nothing that cycle instead of holding up the batch.
Tradeoff vs nvstreammux's fixed shapes: robustness over per-batch latency
predictability — window_ms is deployment-tunable.
"""
from __future__ import annotations

import time

from .frame import RawFrame
from .registry import StreamRegistry


def current_time_us() -> int:
    return time.monotonic_ns() // 1000


class DynamicBatcher:
    def __init__(self, window_ms: int = 33, clock=current_time_us):
        self.window_us = window_ms * 1000
        self._clock = clock
        # A frame's fd is consumed by the bridge on import; never hand the
        # same frame out twice across cycles.
        self._last_pts: dict[str, int] = {}

    def assemble_batch(self, registry: StreamRegistry) -> list[RawFrame]:
        now = self._clock()
        items: list[RawFrame] = []
        for stream in registry.snapshot():
            frame = stream.frame_queue.peek_latest_within(now - self.window_us, now)
            if frame is None:
                continue  # stalled source sits out this cycle
            if frame.pts <= self._last_pts.get(frame.source_id, -1):
                continue  # already dispatched this frame in a prior cycle
            self._last_pts[frame.source_id] = frame.pts
            items.append(frame)  # per-item resolution may differ
        return items
