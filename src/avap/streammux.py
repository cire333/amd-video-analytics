"""StreamMux — nvstreammux-equivalent batch former for AMD (no GPU code here).

Reverse-engineered from DeepStream 7.1 (docs/nvstreammux_reverse_engineering.md):

* ``policy="legacy"`` reproduces the closed-source default nvstreammux:
  - live_source=False: FIFO over all arrivals (a source may fill several
    slots); push when full or ``batched_push_timeout_us`` after the oldest
    queued buffer arrived; output pts = n * frame_duration(first source).
  - live_source=True: at most one frame per source per batch; push when every
    non-EOS source contributed or the timeout elapsed; pts = running time.
* ``policy="new"`` is a port of the open-source ``BatchPolicy``/``NvStreamMux``
  (gst-nvmultistream2): priority groups, round-robin with carried position,
  adaptive batch size (one slot per non-EOS source), min/max-fps windows since
  the last push, max-same-source-frames, source rate control, sync-inputs.

Metadata matches NvDsFrameMeta field-for-field (pad_index, source_id,
batch_id, frame_num, buf_pts, ntp_timestamp, source_frame_width/height,
num_surfaces_per_frame) plus the caller's payload (a RawFrame, typically).

The class is passive: decoder threads call ``push_frame``/``push_eos``, one
consumer thread calls ``poll(now_ns)`` (or uses ``StreamMuxRunner``). The
monotonic clock is injected so tests are deterministic.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Deque, Iterable

log = logging.getLogger(__name__)

NS = 1_000_000_000


def monotonic_ns() -> int:
    return time.monotonic_ns()


def wall_ns() -> int:
    return time.time_ns()


# --------------------------------------------------------------------------- config

@dataclass
class SourceConfig:
    """[source-config-N] group of the new mux's config file."""
    priority: int = 0
    max_fps: Fraction = Fraction(60, 1)
    min_fps: Fraction = Fraction(30, 1)
    max_frames_per_batch: int = 1


@dataclass
class MuxConfig:
    batch_size: int = 1
    # legacy: µs to wait after the oldest queued buffer; -1 = forever.
    # new:    overall-min-fps = 1e6 / timeout (0/-1 = no minimum rate).
    batched_push_timeout_us: int = 33000
    live_source: bool = False
    policy: str = "legacy"               # "legacy" (default nvstreammux) | "new"
    # --- new-mux policy knobs ([property] group) ---
    adaptive_batching: bool = True
    algorithm: str = "round_robin"       # "round_robin" | "priority"
    overall_max_fps: Fraction = Fraction(120, 1)
    max_fps_control: bool = False
    max_same_source_frames: int = 1
    enable_source_rate_control: bool = False
    sources: dict[int, SourceConfig] = field(default_factory=dict)
    # --- timestamps / sync ---
    sync_inputs: bool = False
    max_latency_ns: int = 0
    attach_sys_ts: bool = True
    frame_num_reset_on_eos: bool = False
    # --- canvas (consumed by avap.canvas; carried here like the old mux's props) ---
    width: int = 0
    height: int = 0
    enable_padding: bool = False
    interpolation: str = "bilinear"      # "bilinear" | "nearest"
    # --- per-pad input queue ---
    # DS blocks the upstream chain when a pad's queue is full. avap does the
    # same when block_when_full is True (default for file sources), and
    # drops the oldest queued frame instead when False (default for live
    # sources, where a blocked decoder would fall behind the camera anyway).
    queue_depth: int = 4
    block_when_full: bool | None = None
    # legacy non-live: old mux's 2-deep per-pad input (max repeats per batch)
    max_same_source_frames_legacy: int = 2

    @property
    def overall_min_fps(self) -> Fraction:
        if self.batched_push_timeout_us <= 0:
            return Fraction(0)
        return Fraction(1_000_000, self.batched_push_timeout_us)

    def timeout_ns(self) -> int | None:
        return None if self.batched_push_timeout_us < 0 else self.batched_push_timeout_us * 1000

    @property
    def blocks_when_full(self) -> bool:
        return (not self.live_source) if self.block_when_full is None else self.block_when_full


# --------------------------------------------------------------------------- outputs

@dataclass
class MuxFrameMeta:
    """NvDsFrameMeta analog; ``payload`` is whatever was pushed (RawFrame)."""
    pad_index: int
    source_id: int
    batch_id: int
    frame_num: int
    buf_pts: int                     # ns, as pushed
    ntp_timestamp: int               # ns since epoch (0 if unavailable)
    source_frame_width: int
    source_frame_height: int
    payload: Any
    num_surfaces_per_frame: int = 1
    recv_ns: int = 0                 # local monotonic arrival (avap extra)


@dataclass
class MuxBatch:
    """NvDsBatchMeta analog. ``max_frames_in_batch`` is the configured
    batch size even when the batch is partial, exactly like DeepStream."""
    frames: list[MuxFrameMeta]
    pts: int
    batch_index: int
    max_frames_in_batch: int
    push_ns: int

    @property
    def num_frames_in_batch(self) -> int:
        return len(self.frames)

    def pad_indices(self) -> tuple[int, ...]:
        return tuple(f.pad_index for f in self.frames)


@dataclass
class MuxEvent:
    """Downstream custom events: pad-added, stream-segment, stream-eos, eos, pad-deleted."""
    kind: str
    pad_index: int = -1
    source_id: int = -1
    data: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- pads

@dataclass
class _Queued:
    payload: Any
    pts: int
    width: int
    height: int
    ntp: int
    recv_ns: int
    seq: int          # per-pad receive index (old-mux frame_num)


class SinkPad:
    """Per-source input queue (SinkPad in nvstreammux_pads.cpp)."""

    def __init__(self, pad_index: int, source_id: int | None, cfg: SourceConfig,
                 queue_depth: int):
        self.pad_index = pad_index
        self.source_id = pad_index if source_id is None else source_id
        self.cfg = cfg
        self.queue: Deque[_Queued] = collections.deque()
        self.queue_depth = queue_depth
        self.frame_count = 0          # frames batched (new-mux frame_num)
        self.received = 0             # frames pushed (old-mux frame_num)
        self.dropped = 0
        self.eos_pending = False      # EOS pushed, queue still draining
        self.eos = False              # EOS forwarded downstream
        self.segment_sent = False
        self.frame_duration_ns: int = 0
        self.last_pts: int | None = None
        self.segment_start: int | None = None   # running time origin (sync-inputs)
        self.last_push_ns: int | None = None    # source rate control

    def available(self) -> int:
        return len(self.queue)

    def reset_frame_count(self) -> None:
        self.frame_count = 0
        self.received = 0

    @property
    def done(self) -> bool:
        """Cannot contribute any more frames."""
        return self.eos or (self.eos_pending and not self.queue)


# --------------------------------------------------------------------------- mux

class StreamMux:
    def __init__(self, cfg: MuxConfig | None = None,
                 clock: Callable[[], int] = monotonic_ns,
                 wall_clock: Callable[[], int] = wall_ns):
        self.cfg = cfg or MuxConfig()
        if self.cfg.policy not in ("legacy", "new"):
            raise ValueError(f"unknown policy {self.cfg.policy!r}")
        self._clock = clock
        self._wall = wall_clock
        self._lock = threading.Condition()
        self.pads: dict[int, SinkPad] = {}
        self._pad_order: list[int] = []          # insertion order == sink_%u order
        self.batch_index = 0
        self.base_ns: int | None = None          # running-time origin (start() or first push)
        self.first_push_ns: int | None = None
        self._first_pad: int | None = None
        self.first_frame_duration_ns = 0         # non-live pts step
        self._pending_events: list[MuxEvent] = []
        self.eos_sent = False
        self._prev_out_pts: int | None = None
        self._last_batch_ns: int | None = None   # BatchPolicy::last_batch_time
        self._rr_priority_pos = 0
        self._rr_source_pos = 0
        self._stopping = False
        self.stats = {"batches": 0, "frames": 0, "dropped_overflow": 0, "dropped_late": 0}

    # ------------------------------------------------------------------ lifecycle
    def start(self, now: int | None = None) -> None:
        """Set the running-time origin (pipeline PLAYING). Optional: the
        first pushed frame does it otherwise."""
        with self._lock:
            self.base_ns = self._clock() if now is None else now
            self._stopping = False

    def stop(self) -> None:
        """Unblock producers waiting in push_frame and refuse new frames."""
        with self._lock:
            self._stopping = True
            self._lock.notify_all()

    # ------------------------------------------------------------------ pads
    def add_pad(self, pad_index: int, source_id: int | None = None,
                source_cfg: SourceConfig | None = None) -> SinkPad:
        with self._lock:
            if pad_index in self.pads:
                raise ValueError(f"pad {pad_index} already exists")
            scfg = source_cfg or self.cfg.sources.get(pad_index) or SourceConfig()
            pad = SinkPad(pad_index, source_id, scfg, self.cfg.queue_depth)
            self.pads[pad_index] = pad
            self._pad_order.append(pad_index)
            self.eos_sent = False
            self._pending_events.append(MuxEvent("pad-added", pad_index, pad.source_id))
            self._lock.notify_all()
            return pad

    def remove_pad(self, pad_index: int) -> None:
        with self._lock:
            pad = self.pads.pop(pad_index)
            self._pad_order.remove(pad_index)
            self.stats["dropped_overflow"] += len(pad.queue)
            self._release(pad.queue)
            self._pending_events.append(MuxEvent("pad-deleted", pad_index, pad.source_id))
            self._lock.notify_all()

    @staticmethod
    def _release(items: Iterable[_Queued]) -> None:
        for q in items:
            close = getattr(q.payload, "close", None)
            if callable(close):
                close()

    # ------------------------------------------------------------------ inputs
    def push_frame(self, pad_index: int, payload: Any, pts_ns: int,
                   width: int, height: int, ntp_ns: int | None = None,
                   frame_duration_ns: int | None = None) -> bool:
        """Queue a decoded frame. Returns False if a frame was dropped to make
        room (bounded queue, drop-oldest — avap's stand-in for DS's blocking
        chain + leaky upstream queue). Payloads with ``close()`` are closed
        when dropped.
        """
        now = self._clock()
        with self._lock:
            pad = self.pads[pad_index]
            if pad.eos_pending or pad.eos:
                self._release([_Queued(payload, pts_ns, width, height, 0, now, 0)])
                return False
            if self.base_ns is None:
                self.base_ns = now
            if self.first_push_ns is None:
                self.first_push_ns = now
                self._first_pad = pad_index
                self._last_batch_ns = now      # "imaginary 0th batch" (add_buffer)
            if frame_duration_ns:
                pad.frame_duration_ns = frame_duration_ns
            elif (pad.last_pts is not None and pts_ns > pad.last_pts
                  and not pad.frame_duration_ns):
                pad.frame_duration_ns = pts_ns - pad.last_pts
            pad.last_pts = pts_ns
            if pad.segment_start is None:
                pad.segment_start = pts_ns
            if not pad.segment_sent:
                pad.segment_sent = True
                self._pending_events.append(MuxEvent(
                    "stream-segment", pad_index, pad.source_id, {"start_pts": pts_ns}))
            if ntp_ns is None:
                ntp_ns = self._wall() if self.cfg.attach_sys_ts else 0
            dropped = False
            if len(pad.queue) >= pad.queue_depth:
                if self.cfg.blocks_when_full:
                    # DS semantics: the chain blocks until a batch consumed a frame
                    while (len(pad.queue) >= pad.queue_depth and not self._stopping
                           and not pad.eos_pending and pad.pad_index in self.pads):
                        self._lock.wait(0.05)
                    if self._stopping or pad.pad_index not in self.pads:
                        self._release([_Queued(payload, pts_ns, width, height, 0, now, 0)])
                        return False
                    now = self._clock()
                else:
                    self._release([pad.queue.popleft()])
                    pad.dropped += 1
                    self.stats["dropped_overflow"] += 1
                    dropped = True
            seq = pad.received
            pad.received += 1
            pad.queue.append(_Queued(payload, pts_ns, width, height, ntp_ns, now, seq))
            self._lock.notify_all()
            return not dropped

    def push_eos(self, pad_index: int) -> None:
        with self._lock:
            pad = self.pads[pad_index]
            if not pad.eos and not pad.eos_pending:
                pad.eos_pending = True
                self._lock.notify_all()

    # ------------------------------------------------------------------ time helpers
    def running_time(self, now: int | None = None) -> int:
        now = self._clock() if now is None else now
        return 0 if self.base_ns is None else max(0, now - self.base_ns)

    def _active_pads(self) -> list[SinkPad]:
        return [self.pads[i] for i in self._pad_order if not self.pads[i].done]

    def _oldest_queued_ns(self) -> int | None:
        times = [p.queue[0].recv_ns for p in self.pads.values() if p.queue and not p.done]
        return min(times) if times else None

    def _legacy_timer_start(self) -> int | None:
        """Old mux: the timeout runs from the batch's first buffer, but never
        from before the previous push (S4_old: steady 34 ms cadence)."""
        oldest = self._oldest_queued_ns()
        if oldest is None:
            return None
        return oldest if self._last_batch_ns is None else max(oldest, self._last_batch_ns)

    def _frame_duration_for_pts(self) -> int:
        """Old mux: 1/framerate of the source whose buffer arrived first."""
        if not self.first_frame_duration_ns:
            first = self.pads.get(self._first_pad) if self._first_pad is not None else None
            if first is not None and first.frame_duration_ns:
                self.first_frame_duration_ns = first.frame_duration_ns
            else:
                for i in self._pad_order:
                    if self.pads[i].frame_duration_ns:
                        self.first_frame_duration_ns = self.pads[i].frame_duration_ns
                        break
        return self.first_frame_duration_ns

    def _min_fps_dur_ns(self) -> int | None:
        f = self.cfg.overall_min_fps
        return None if f == 0 else int(NS / f)

    # ------------------------------------------------------------------ polling
    def next_deadline_ns(self, now: int | None = None) -> int | None:
        """Absolute monotonic time at which poll() may have something new
        (None = only on new input)."""
        with self._lock:
            if self.cfg.policy == "legacy":
                t = self.cfg.timeout_ns()
                start = self._legacy_timer_start()
                if t is None or start is None:
                    return None
                return start + t
            d = self._min_fps_dur_ns()
            if d is None or self._last_batch_ns is None:
                return None
            if not any(p.queue for p in self.pads.values()):
                return None
            return self._last_batch_ns + d

    def poll(self, now: int | None = None) -> list[MuxBatch | MuxEvent]:
        """Drain pending events, form at most one batch, forward EOS.
        Cheap to call often; never blocks."""
        now = self._clock() if now is None else now
        out: list[MuxBatch | MuxEvent] = []
        with self._lock:
            out.extend(self._pending_events)
            self._pending_events.clear()
            self._forward_eos(out)
            batch = (self._form_legacy(now) if self.cfg.policy == "legacy"
                     else self._form_new(now))
            if batch is not None:
                out.append(batch)
                self._forward_eos(out)
            if self.pads and not self.eos_sent and all(p.eos for p in self.pads.values()):
                self.eos_sent = True
                out.append(MuxEvent("eos"))
        return out

    def drain(self, now: int | None = None) -> list[MuxBatch | MuxEvent]:
        """poll() until nothing more comes out at this instant."""
        out: list[MuxBatch | MuxEvent] = []
        while True:
            got = self.poll(now)
            if not got:
                return out
            out.extend(got)

    def _forward_eos(self, out: list) -> None:
        """A pad's EOS goes downstream once its queue is drained (push_events)."""
        for i in self._pad_order:
            pad = self.pads[i]
            if pad.eos_pending and not pad.queue and not pad.eos:
                pad.eos = True
                pad.eos_pending = False
                if self.cfg.frame_num_reset_on_eos:
                    pad.reset_frame_count()
                out.append(MuxEvent("stream-eos", pad.pad_index, pad.source_id))

    # ------------------------------------------------------------------ emit
    def _emit(self, picks: list[tuple[SinkPad, _Queued]], now: int) -> MuxBatch:
        legacy = self.cfg.policy == "legacy"
        frames = []
        for slot, (pad, q) in enumerate(picks):
            pad.queue.remove(q)
            frame_num = q.seq if legacy else pad.frame_count
            pad.frame_count += 1
            frames.append(MuxFrameMeta(
                pad_index=pad.pad_index, source_id=pad.source_id, batch_id=slot,
                frame_num=frame_num, buf_pts=q.pts, ntp_timestamp=q.ntp,
                source_frame_width=q.width, source_frame_height=q.height,
                payload=q.payload, recv_ns=q.recv_ns))
        for pad in {p for p, _ in picks}:
            pad.last_push_ns = now
        pts = self._output_pts(frames, now)
        batch = MuxBatch(frames=frames, pts=pts, batch_index=self.batch_index,
                         max_frames_in_batch=self.config_batch_size(), push_ns=now)
        self.batch_index += 1
        self._last_batch_ns = now
        self.stats["batches"] += 1
        self.stats["frames"] += len(frames)
        self._lock.notify_all()                  # wake producers blocked on a full pad
        return batch

    def config_batch_size(self) -> int:
        """NvDsBatchMeta.max_frames_in_batch."""
        if self.cfg.policy == "new" and self.cfg.adaptive_batching:
            return max(self.cfg.batch_size, len(self.pads))
        return self.cfg.batch_size

    def _output_pts(self, frames: list[MuxFrameMeta], now: int) -> int:
        if self.cfg.sync_inputs:
            pts = max(self._pad_running_time(self.pads[f.pad_index], f.buf_pts) for f in frames)
        elif self.cfg.policy == "legacy" and self.cfg.live_source:
            pts = self.running_time(now)
        else:
            pts = self.batch_index * self._frame_duration_for_pts()
            if self.cfg.policy == "new":
                # pts_offset = running time when the first buffer arrived
                pts += (self.first_push_ns or 0) - (self.base_ns or 0)
        if self._prev_out_pts is not None and pts == self._prev_out_pts:
            pts += 250_000  # GST_MSECOND >> 2, as the new mux does
        self._prev_out_pts = pts
        return pts

    @staticmethod
    def _pad_running_time(pad: SinkPad, pts: int) -> int:
        return pts - (pad.segment_start or 0)

    # ------------------------------------------------------------------ legacy policy
    def _form_legacy(self, now: int) -> MuxBatch | None:
        cfg = self.cfg
        active = self._active_pads()
        bs = cfg.batch_size
        if bs <= 0 or not active:
            return None
        timeout = cfg.timeout_ns()
        start = self._legacy_timer_start()
        due = start is not None and timeout is not None and now >= start + timeout
        picks: list[tuple[SinkPad, _Queued]] = []
        if cfg.live_source:
            for pad in active:                       # one slot per source, oldest frame
                avail = len(pad.queue)
                if cfg.sync_inputs:
                    avail = self._sync_filter(pad, avail, now)
                if avail and len(picks) < bs:
                    picks.append((pad, pad.queue[0]))
            full = len(picks) >= min(bs, len(active))
        else:                                        # FIFO over all arrivals, repeats allowed
            # The old mux holds at most 2 buffers per pad (upstream blocks on
            # the 3rd), so a source never fills more than 2 slots of a batch
            # (R1/R8/R7 traces: max same-source frames == 2). Excess frames
            # simply wait for the next batch.
            items = sorted(((q.recv_ns, pad.pad_index, q.seq, pad, q)
                            for pad in active for q in pad.queue),
                           key=lambda t: t[:3])
            per_src: collections.Counter = collections.Counter()
            for _, _, _, pad, q in items:
                if per_src[pad.pad_index] >= self.cfg.max_same_source_frames_legacy:
                    continue
                picks.append((pad, q))
                per_src[pad.pad_index] += 1
                if len(picks) >= bs:
                    break
            full = len(picks) >= bs
        if not picks or not (full or due):
            return None
        return self._emit(picks, now)

    # ------------------------------------------------------------------ new policy (BatchPolicy port)
    def _new_batch_size(self) -> int:
        if self.cfg.adaptive_batching:
            return sum(1 for p in self.pads.values() if not p.eos)
        return self.cfg.batch_size

    def _allowed_repeats(self, pad: SinkPad, bs: int) -> int:
        # BatchPolicy::update_with_source: min(max-same-source-frames,
        #   max(batch_size, source.max-num-frames-per-batch))
        return min(self.cfg.max_same_source_frames, max(bs, pad.cfg.max_frames_per_batch))

    def _form_new(self, now: int) -> MuxBatch | None:
        cfg = self.cfg
        if not self.pads:
            return None
        bs = self._new_batch_size()
        if bs <= 0:
            return None
        groups: dict[int, list[SinkPad]] = collections.defaultdict(list)
        for i in self._pad_order:
            groups[self.pads[i].cfg.priority].append(self.pads[i])
        priorities = sorted(groups)
        rr = cfg.algorithm == "round_robin"
        p_start = self._rr_priority_pos % len(priorities) if rr else 0
        picks: list[tuple[SinkPad, _Queued]] = []
        taken: collections.Counter = collections.Counter()
        first_group = True
        for gi in range(len(priorities)):
            prio = priorities[(p_start + gi) % len(priorities)]
            srcs = groups[prio]
            start = (self._rr_source_pos % len(srcs)) if (rr and first_group) else 0
            first_group = False
            for si in range(len(srcs)):
                pad = srcs[(start + si) % len(srcs)]
                if pad.done:
                    continue
                avail = len(pad.queue)
                if cfg.enable_source_rate_control and pad.last_push_ns is not None:
                    avail = min(avail, int((now - pad.last_push_ns) * pad.cfg.max_fps / NS))
                if cfg.sync_inputs:
                    avail = self._sync_filter(pad, avail, now)
                n = min(avail, self._allowed_repeats(pad, bs), bs - len(picks))
                for k in range(max(0, n)):
                    picks.append((pad, pad.queue[k]))
                taken[pad.pad_index] = n
                if len(picks) >= bs:
                    self._rr_source_pos = (start + si + 1) % len(srcs)
                    self._rr_priority_pos = (p_start + gi) % len(priorities)
                    break
            if len(picks) >= bs:
                break
        if not picks:
            return None
        ready = len(picks) >= bs
        max_dur = self._min_fps_dur_ns()
        due = (self._last_batch_ns is not None and max_dur is not None
               and now >= self._last_batch_ns + max_dur)
        if not (ready or due):
            return None
        if cfg.max_fps_control and self._last_batch_ns is not None:
            if now < self._last_batch_ns + int(NS / cfg.overall_max_fps):
                return None                          # throttled; re-poll at deadline
        return self._emit(picks, now)

    def _sync_filter(self, pad: SinkPad, avail: int, now: int) -> int:
        """NvTimeSync::get_synch_info over the first ``avail`` queued buffers:
        drop late ones, stop at the first early one, count on-time ones."""
        min_dur = self._min_fps_dur_ns() or 0
        cur = self.running_time(now)
        if cur < min_dur:
            return avail                             # pipeline just starting
        on_time = 0
        i = 0
        while i < len(pad.queue) and i < avail:
            q = pad.queue[i]
            rt = self._pad_running_time(pad, q.pts)
            if rt > cur - min_dur:
                break                                # early: the rest are early too
            if rt + self.cfg.max_latency_ns < cur - min_dur:
                del pad.queue[i]                     # late: drop and look at the next
                self._release([q])
                pad.dropped += 1
                self.stats["dropped_late"] += 1
                avail -= 1
                continue
            on_time += 1
            i += 1
        return on_time


# --------------------------------------------------------------------------- driver

class StreamMuxRunner:
    """Thread that turns the passive StreamMux into a push source, like the
    GStreamer src-pad task: sleeps until the next deadline or new input."""

    def __init__(self, mux: StreamMux, on_batch: Callable[[MuxBatch], None],
                 on_event: Callable[[MuxEvent], None] | None = None,
                 idle_poll_ms: float = 50.0):
        self.mux = mux
        self.on_batch = on_batch
        self.on_event = on_event or (lambda e: None)
        self.idle_poll_ns = int(idle_poll_ms * 1e6)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="streammux", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.mux.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        mux = self.mux
        while not self._stop.is_set():
            items = mux.poll()
            for item in items:
                try:
                    if isinstance(item, MuxBatch):
                        self.on_batch(item)
                    else:
                        self.on_event(item)
                except Exception:
                    log.exception("streammux consumer failed")
            if items:
                continue                              # more may be ready right now
            deadline = mux.next_deadline_ns()
            now = mux._clock()
            wait_ns = self.idle_poll_ns if deadline is None else max(deadline - now, 100_000)
            with mux._lock:
                mux._lock.wait(wait_ns / NS)
