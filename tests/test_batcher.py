"""Batcher + ring buffer behavior — runs without GPU or the extension."""
import avap.batcher as batcher_mod
from avap.batcher import DynamicBatcher
from avap.frame import ColorMatrix, ColorRange, CropRect, PlaneLayout, RawFrame
from avap.ringbuffer import RingBuffer


def make_frame(source_id="cam0", pts=0, recv_us=0, fd=-1):
    return RawFrame(
        source_id=source_id, pts=pts, recv_us=recv_us, width=1920, height=1088,
        crop=CropRect(0, 0, 1920, 1080), dmabuf_fd=fd,
        planes=(PlaneLayout(0, 2048), PlaneLayout(2048 * 1088, 2048)),
        drm_modifier=0, color_range=ColorRange.LIMITED,
        color_matrix=ColorMatrix.BT709, device_ordinal=0,
    )


class FakeStream:
    def __init__(self, source_id, frames):
        self.source_id = source_id
        self.frame_queue = RingBuffer(capacity=8)
        for f in frames:
            self.frame_queue.push(f)


class FakeRegistry:
    def __init__(self, streams):
        self._streams = streams

    def snapshot(self):
        return self._streams


def test_ringbuffer_drop_oldest_keeps_freshest():
    rb = RingBuffer(capacity=2)
    frames = [make_frame(pts=i, recv_us=i) for i in range(4)]
    for f in frames:
        rb.push(f)
    assert rb.dropped == 2
    assert rb.peek_latest_within(0, 100).pts == 3
    assert len(rb) == 2


def test_stalled_source_sits_out_cycle():
    now = 100_000  # µs; window is 33 ms = 33_000 µs
    fresh = FakeStream("fresh", [make_frame("fresh", pts=1, recv_us=now - 1_000)])
    stale = FakeStream("stale", [make_frame("stale", pts=1, recv_us=now - 50_000)])
    b = DynamicBatcher(window_ms=33, clock=lambda: now)
    batch = b.assemble_batch(FakeRegistry([fresh, stale]))
    assert [f.source_id for f in batch] == ["fresh"]


def test_same_frame_never_dispatched_twice():
    stream = FakeStream("cam0", [make_frame("cam0", pts=5, recv_us=990)])
    b = DynamicBatcher(window_ms=33, clock=lambda: 1000)
    assert len(b.assemble_batch(FakeRegistry([stream]))) == 1
    assert len(b.assemble_batch(FakeRegistry([stream]))) == 0  # already dispatched


def test_variable_shape_batch():
    a = FakeStream("a", [make_frame("a", pts=1, recv_us=995)])
    hd = make_frame("b", pts=1, recv_us=996)
    hd.crop = CropRect(0, 0, 1280, 720)
    bstream = FakeStream("b", [hd])
    b = DynamicBatcher(window_ms=33, clock=lambda: 1000)
    batch = b.assemble_batch(FakeRegistry([a, bstream]))
    assert {f.crop.width for f in batch} == {1920, 1280}
