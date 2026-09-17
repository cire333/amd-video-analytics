"""Stream lifecycle events and retry policy for the streaming API.

RetryPolicy controls the restart cycle for a stream's source (network
reattachment for rtsp/http, reopen for files):

    RetryPolicy(max_retries=10, initial_backoff_s=1, max_backoff_s=60,
                backoff_multiplier=2, reset_after_s=30)

- max_retries: consecutive failed reattachment attempts before the stream
  gives up (state "failed"). None = retry forever (typical for live cams).
- backoff grows initial * multiplier**(attempt-1), capped at max_backoff_s.
- reset_after_s: a connection that stayed up at least this long resets the
  attempt counter, so an overnight outage isn't charged against a camera
  that hiccups once a day.

Error reporting: pass on_event=callable to AMDStream (or a manager-wide
default via AMDGPUManager(config={"on_event": fn})). The callable receives
StreamEvent objects; exceptions raised inside it are swallowed and logged —
a broken reporter must never take down the stream.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

EVENT_TYPES = (
    "connecting",     # about to (re)open the source
    "connected",      # source open, frames flowing
    "disconnected",   # source lost or errored (error field says why)
    "reconnecting",   # scheduled reattachment (attempt, backoff_s set)
    "gave_up",        # retry cycle exhausted; stream state -> failed
    "eof",            # non-live source finished normally
    "sink_error",     # output emit failed; record dropped, stream continues
)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int | None = None      # None = retry forever
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0
    backoff_multiplier: float = 2.0
    reset_after_s: float = 30.0

    def __post_init__(self):
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError("max_retries must be >= 0 or None")
        if self.initial_backoff_s <= 0 or self.max_backoff_s <= 0:
            raise ValueError("backoff times must be > 0")
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1")

    def backoff(self, attempt: int) -> float:
        """Backoff before retry `attempt` (1-based)."""
        return min(self.initial_backoff_s
                   * self.backoff_multiplier ** max(0, attempt - 1),
                   self.max_backoff_s)


@dataclass(frozen=True)
class StreamEvent:
    type: str                     # one of EVENT_TYPES
    source_id: str
    ts: float = field(default_factory=time.time)
    attempt: int = 0              # reattachment attempt number (retry events)
    backoff_s: float = 0.0        # sleep before this attempt (reconnecting)
    error: str | None = None      # repr of the triggering exception
    frames_processed: int = 0     # stream progress at event time


EventCallback = Callable[[StreamEvent], None]
