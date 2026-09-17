"""StreamMux behaviour, pinned to what nvstreammux does on the 3090
(docs/nvstreammux_reverse_engineering.md). Deterministic clock, no GPU."""
from fractions import Fraction

import pytest

from avap.streammux import (MuxBatch, MuxConfig, MuxEvent, SourceConfig, StreamMux,
                            StreamMuxRunner)

MS = 1_000_000


class Clock:
    def __init__(self, t=0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += int(ms * MS)


class Payload:
    def __init__(self, tag):
        self.tag = tag
        self.closed = False

    def close(self):
        self.closed = True


def make(cfg, pads, t0=0):
    clk = Clock(t0)
    mux = StreamMux(cfg, clock=clk, wall_clock=lambda: 1_700_000_000 * 10**9 + clk.t)
    for p in pads:
        mux.add_pad(p)
    mux.drain()  # consume pad-added events
    return mux, clk


def push(mux, pad, pts_ms, w=1280, h=720, tag=None):
    return mux.push_frame(pad, Payload(tag or (pad, pts_ms)), int(pts_ms * MS), w, h)


def batches(items):
    return [i for i in items if isinstance(i, MuxBatch)]


def events(items, kind=None):
    return [i for i in items if isinstance(i, MuxEvent) and (kind is None or i.kind == kind)]


# ----------------------------------------------------------------- legacy, non-live (files)

def test_legacy_nonlive_fifo_repeats_and_arrival_order():
    """R8_old_file4_bs2: batch-size 2 with 4 sources -> pairs in arrival order,
    same source may fill both slots."""
    mux, clk = make(MuxConfig(batch_size=2, batched_push_timeout_us=40000), [0, 1, 2, 3])
    push(mux, 0, 0); clk.advance(1)
    push(mux, 0, 33); clk.advance(1)
    push(mux, 2, 0); clk.advance(1)
    push(mux, 1, 0)
    out = mux.drain()
    assert [b.pad_indices() for b in batches(out)] == [(0, 0), (2, 1)]
    b0 = batches(out)[0]
    assert [f.batch_id for f in b0.frames] == [0, 1]
    assert [f.frame_num for f in b0.frames] == [0, 1]
    assert b0.max_frames_in_batch == 2 and b0.num_frames_in_batch == 2


def test_legacy_nonlive_at_most_two_slots_per_source():
    """R1_old_file4: the fast 540p source never fills more than 2 of 4 slots."""
    mux, clk = make(MuxConfig(batch_size=4, batched_push_timeout_us=40000), [0, 1])
    for i in range(4):
        push(mux, 0, i * 33); clk.advance(1)
    push(mux, 1, 0); clk.advance(1)
    push(mux, 1, 33)
    b = batches(mux.drain())
    assert b[0].pad_indices() == (0, 0, 1, 1)
    assert [f.frame_num for f in b[0].frames] == [0, 1, 0, 1]
    clk.advance(45)                          # leftovers of source 0 go out on timeout
    assert batches(mux.drain())[0].pad_indices() == (0, 0)


def test_legacy_nonlive_waits_for_full_batch_then_timeout_pushes_partial():
    mux, clk = make(MuxConfig(batch_size=4, batched_push_timeout_us=40000), [0, 1])
    push(mux, 0, 0)
    push(mux, 1, 0)
    assert batches(mux.drain()) == []                 # 2 of 4, timer running
    clk.advance(39)
    assert batches(mux.drain()) == []
    assert mux.next_deadline_ns() == 40 * MS
    clk.advance(1)
    b = batches(mux.drain())
    assert len(b) == 1 and b[0].pad_indices() == (0, 1)


def test_legacy_nonlive_infinite_timeout_never_pushes_partial():
    mux, clk = make(MuxConfig(batch_size=3, batched_push_timeout_us=-1), [0, 1])
    push(mux, 0, 0); push(mux, 1, 0)
    clk.advance(10_000)
    assert batches(mux.drain()) == []
    assert mux.next_deadline_ns() is None
    push(mux, 0, 33)
    assert batches(mux.drain())[0].pad_indices() == (0, 1, 0)


def test_legacy_nonlive_output_pts_is_n_times_first_source_frame_duration():
    """R1_old_file4: 25 fps source arrived first -> 40 ms steps regardless of the others."""
    mux, clk = make(MuxConfig(batch_size=1, batched_push_timeout_us=40000), [0, 1])
    push(mux, 1, 0); push(mux, 1, 40)          # 25 fps source first
    push(mux, 0, 66); push(mux, 0, 99)         # 30 fps
    pts = [b.pts for b in batches(mux.drain())]
    assert pts == [0, 40 * MS, 80 * MS, 120 * MS]


def test_legacy_frame_num_counts_received_frames_so_drops_leave_gaps():
    mux, clk = make(MuxConfig(batch_size=1, batched_push_timeout_us=-1, queue_depth=2,
                              block_when_full=False), [0])
    p0 = Payload("a")
    mux.push_frame(0, p0, 0, 640, 480)
    mux.push_frame(0, Payload("b"), 33 * MS, 640, 480)
    assert mux.push_frame(0, Payload("c"), 66 * MS, 640, 480) is False   # overflow drops "a"
    assert p0.closed and mux.pads[0].dropped == 1
    fn = [b.frames[0].frame_num for b in batches(mux.drain())]
    assert fn == [1, 2]


def test_legacy_eos_shrinks_then_stream_eos_then_pipeline_eos():
    mux, clk = make(MuxConfig(batch_size=2, batched_push_timeout_us=40000), [0, 1])
    push(mux, 0, 0); push(mux, 1, 0)
    assert batches(mux.drain())[0].pad_indices() == (0, 1)
    mux.push_eos(1)
    out = mux.drain()
    assert [e.kind for e in events(out)] == ["stream-eos"]
    assert events(out)[0].pad_index == 1
    # remaining source alone fills both slots (R1 tail behaviour)
    push(mux, 0, 33); push(mux, 0, 66)
    assert batches(mux.drain())[0].pad_indices() == (0, 0)
    mux.push_eos(0)
    kinds = [e.kind for e in events(mux.drain())]
    assert kinds == ["stream-eos", "eos"]
    assert mux.eos_sent


# ----------------------------------------------------------------- legacy, live

def test_legacy_live_one_slot_per_source_and_timeout_cadence():
    """S3_old_live4: 30/30/15/10 fps, timeout 33 ms -> 2..4-frame batches every 33 ms."""
    cfg = MuxConfig(batch_size=4, batched_push_timeout_us=33333, live_source=True)
    mux, clk = make(cfg, [0, 1, 2, 3])
    push(mux, 0, 0); push(mux, 1, 0); push(mux, 2, 0); push(mux, 3, 0)
    b = batches(mux.drain())
    assert b[0].pad_indices() == (0, 1, 2, 3)            # all contributed -> immediate
    clk.advance(33.3)
    push(mux, 0, 33); push(mux, 1, 33); push(mux, 0, 66)   # source 0 twice, no 2/3
    assert batches(mux.drain()) == []                      # not all sources -> wait
    clk.advance(33.4)
    b = batches(mux.drain())
    assert len(b) == 1 and b[0].pad_indices() == (0, 1)    # one per source, oldest first
    assert b[0].frames[0].buf_pts == 33 * MS
    assert b[0].pts == mux.running_time()                  # live: pts = running time


def test_legacy_live_batch_size_bigger_than_sources_waits_every_time():
    """S4_old_live2_bs4: 2 sources, batch 4 -> 2-frame batches only after the timeout."""
    cfg = MuxConfig(batch_size=4, batched_push_timeout_us=33333, live_source=True)
    mux, clk = make(cfg, [0, 1])
    push(mux, 0, 0); push(mux, 1, 0)
    # min(batch_size, active sources) = 2 -> full as soon as both contributed
    assert batches(mux.drain())[0].pad_indices() == (0, 1)
    push(mux, 0, 33)
    assert batches(mux.drain()) == []
    clk.advance(34)
    assert batches(mux.drain())[0].pad_indices() == (0,)


def test_legacy_live_eos_source_no_longer_awaited():
    cfg = MuxConfig(batch_size=2, batched_push_timeout_us=100000, live_source=True)
    mux, clk = make(cfg, [0, 1])
    mux.push_eos(1)
    mux.drain()
    push(mux, 0, 0)
    assert batches(mux.drain())[0].pad_indices() == (0,)   # immediate, not after 100 ms


# ----------------------------------------------------------------- new policy (BatchPolicy port)

def new_cfg(**kw):
    base = dict(batch_size=4, batched_push_timeout_us=200000, policy="new")
    base.update(kw)
    return MuxConfig(**base)


def test_new_adaptive_round_robin_pad_order_and_shrink_after_eos():
    """R1_new_file4: always (0,1,2,3); 3-frame batches once source 3 is at EOS;
    max_frames_in_batch stays 4; frame_num contiguous."""
    mux, clk = make(new_cfg(), [0, 1, 2, 3])
    for f in range(2):
        for p in (3, 2, 1, 0):                          # arrival order != pad order
            push(mux, p, f * 33)
    out = batches(mux.drain())
    assert [b.pad_indices() for b in out] == [(0, 1, 2, 3), (0, 1, 2, 3)]
    assert all(b.max_frames_in_batch == 4 for b in out)
    mux.push_eos(3)
    assert [e.kind for e in events(mux.drain())] == ["stream-eos"]
    for p in (0, 1, 2):
        push(mux, p, 66)
    b = batches(mux.drain())
    assert b[0].pad_indices() == (0, 1, 2) and b[0].max_frames_in_batch == 4
    assert [f.frame_num for f in b[0].frames] == [2, 2, 2]


def test_new_frame_num_counts_batched_frames_not_received():
    mux, clk = make(new_cfg(batch_size=1, queue_depth=1, block_when_full=False), [0])
    push(mux, 0, 0); push(mux, 0, 33)                   # second push evicts the first
    b = batches(mux.drain())
    assert [f.frame_num for f in b[0].frames] == [0]    # no gap (new mux)


def test_new_waits_for_all_sources_until_min_fps_then_pushes_partial():
    """S3_new / R3_new: slowest source paces; a partial batch only after 1/min-fps."""
    mux, clk = make(new_cfg(batched_push_timeout_us=100000), [0, 1, 2])
    push(mux, 0, 0); push(mux, 1, 0)
    assert batches(mux.drain()) == []
    assert mux.next_deadline_ns() == 100 * MS
    clk.advance(99)
    assert batches(mux.drain()) == []
    clk.advance(1)
    b = batches(mux.drain())
    assert b[0].pad_indices() == (0, 1)


def test_new_non_adaptive_bs2_rotates_over_sources():
    """R8_new_file4_bs2: batch 2 over 4 sources, one frame per source, position carried."""
    mux, clk = make(new_cfg(batch_size=2, adaptive_batching=False), [0, 1, 2, 3])
    for p in range(4):
        push(mux, p, 0); push(mux, p, 33)
    out = batches(mux.drain())
    assert [b.pad_indices() for b in out] == [(0, 1), (2, 3), (0, 1), (2, 3)]
    assert all(b.max_frames_in_batch == 2 for b in out)


def test_new_same_source_repeats_when_configured():
    cfg = new_cfg(batch_size=4, adaptive_batching=False, max_same_source_frames=4,
                  sources={0: SourceConfig(max_frames_per_batch=4)})
    mux, clk = make(cfg, [0])
    for i in range(4):
        push(mux, 0, i * 16.7)
    b = batches(mux.drain())
    assert b[0].pad_indices() == (0, 0, 0, 0)
    assert [f.frame_num for f in b[0].frames] == [0, 1, 2, 3]


def test_new_priority_groups_lower_number_first():
    cfg = new_cfg(batch_size=2, adaptive_batching=False,
                  sources={0: SourceConfig(priority=1), 1: SourceConfig(priority=0)})
    mux, clk = make(cfg, [0, 1])
    push(mux, 0, 0); push(mux, 1, 0)
    assert batches(mux.drain())[0].pad_indices() == (1, 0)


def test_new_sync_inputs_drops_late_frames():
    """S6_new_sync: frames older than now - 1/min_fps - max_latency are dropped."""
    cfg = new_cfg(batch_size=2, batched_push_timeout_us=33333, sync_inputs=True,
                  max_latency_ns=100 * MS)
    mux, clk = make(cfg, [0, 1])
    mux.start()
    push(mux, 0, 0); push(mux, 1, 0)                   # running time 0 for both
    clk.advance(500)
    # on-time window is [now - 1/min_fps - max_latency, now - 1/min_fps] = [366, 466] ms
    push(mux, 0, 400); push(mux, 1, 400)
    push(mux, 0, 490); push(mux, 1, 490)               # early: held back
    b = batches(mux.drain())
    assert b and all(f.buf_pts == 400 * MS for f in b[0].frames)
    assert mux.stats["dropped_late"] == 2              # the pts-0 frames
    assert b[0].pts == 400 * MS                        # sync: max running time in batch
    assert all(len(p.queue) == 1 for p in mux.pads.values())


def test_new_output_pts_offset_plus_frame_duration():
    """S4_new: pts = running time of first buffer + n * frame_duration."""
    mux, clk = make(new_cfg(batch_size=1), [0])
    mux.start()
    clk.advance(223)
    push(mux, 0, 65); push(mux, 0, 98)
    pts = [b.pts for b in batches(mux.drain())]
    assert pts[0] == 223 * MS
    assert pts[1] - pts[0] == 33 * MS


# ----------------------------------------------------------------- driver

def test_runner_pushes_batches_with_real_clock():
    import time
    cfg = MuxConfig(batch_size=2, batched_push_timeout_us=20000, live_source=True)
    mux = StreamMux(cfg)
    mux.add_pad(0); mux.add_pad(1)
    got, evs = [], []
    runner = StreamMuxRunner(mux, got.append, evs.append)
    runner.start()
    try:
        mux.push_frame(0, Payload("a"), 0, 640, 480)
        mux.push_frame(1, Payload("b"), 0, 640, 480)
        mux.push_frame(0, Payload("c"), 33 * MS, 640, 480)   # partial -> after 20 ms timeout
        deadline = time.monotonic() + 2.0
        while len(got) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        runner.stop()
    assert [b.pad_indices() for b in got] == [(0, 1), (0,)]
    assert {e.kind for e in evs} >= {"pad-added", "stream-segment"}


def test_file_sources_block_producer_when_pad_queue_is_full():
    """DS blocks the upstream chain on a full pad; the consumer thread frees it."""
    import threading, time
    cfg = MuxConfig(batch_size=1, batched_push_timeout_us=-1, queue_depth=2)   # files: blocking
    assert cfg.blocks_when_full
    mux = StreamMux(cfg)
    mux.add_pad(0)
    mux.drain()
    mux.push_frame(0, Payload("a"), 0, 64, 64)
    mux.push_frame(0, Payload("b"), 1, 64, 64)
    done = threading.Event()

    def producer():
        mux.push_frame(0, Payload("c"), 2, 64, 64)   # must block until poll() consumes
        done.set()
    threading.Thread(target=producer, daemon=True).start()
    assert not done.wait(0.15)
    assert len(batches(mux.poll())) == 1
    assert done.wait(1.0)
    assert mux.pads[0].dropped == 0 and mux.pads[0].received == 3
    mux.stop()                                       # unblocks any further waiter


def test_live_sources_drop_oldest_by_default():
    cfg = MuxConfig(batch_size=1, live_source=True, queue_depth=1)
    assert not cfg.blocks_when_full


def test_rejects_unknown_policy():
    with pytest.raises(ValueError):
        StreamMux(MuxConfig(policy="bogus"))
