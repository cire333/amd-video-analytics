"""Bounded per-stream frame queue with drop-oldest backpressure.

Dropping closes the frame's dmabuf fd explicitly (architecture §11):
leaked fds hit the process ulimit and look like "crashes after ~10 min".
"""
from __future__ import annotations

import threading
from collections import deque

from .frame import RawFrame


class RingBuffer:
    def __init__(self, capacity: int = 4):
        self._buf: deque[RawFrame] = deque()
        self._capacity = capacity
        self._lock = threading.Lock()
        self.dropped = 0  # counter for observability

    def push(self, frame: RawFrame) -> None:
        """Insert; evict + close the oldest frame if full (keep freshest)."""
        with self._lock:
            if len(self._buf) >= self._capacity:
                self._buf.popleft().close()
                self.dropped += 1
            self._buf.append(frame)

    def peek_latest_within(self, t_min: int, t_max: int) -> RawFrame | None:
        """Newest frame with arrival time in [t_min, t_max], without removing it.

        Windows on recv_us (local monotonic clock), not PTS — PTS timebases
        differ per source. The batcher uses this: a stalled source returns
        None and simply sits out the cycle (architecture §6).
        """
        with self._lock:
            for frame in reversed(self._buf):
                if t_min <= frame.recv_us <= t_max:
                    return frame
            return None

    def pop_latest(self) -> RawFrame | None:
        """Remove and return the newest frame, closing anything staler."""
        with self._lock:
            if not self._buf:
                return None
            newest = self._buf.pop()
            while self._buf:
                self._buf.popleft().close()
                self.dropped += 1
            return newest

    def clear(self) -> None:
        with self._lock:
            while self._buf:
                self._buf.popleft().close()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
