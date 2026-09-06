import pytest

from avap.events import RetryPolicy, StreamEvent
from avap.streaming import AMDStream


def make_stream(**kw) -> AMDStream:
    return AMDStream("rtsp://cam.test/live", source_id="cam-test", **kw)


def run_with_failing_source(stream, fail_times=None, error=ConnectionError("net down")):
    """Drive _run() synchronously with a stubbed source that always (or
    fail_times times) raises."""
    calls = {"n": 0}

    def fake_process():
        calls["n"] += 1
        if fail_times is None or calls["n"] <= fail_times:
            raise error
        stream._stop.set()  # succeed: end the loop

    stream._uri = "rtsp://cam.test/live"
    stream._process_source = fake_process
    stream._run()
    return calls["n"]


def test_policy_validation():
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
    with pytest.raises(ValueError):
        RetryPolicy(backoff_multiplier=0.5)
    assert RetryPolicy().backoff(1) == 1.0
    assert RetryPolicy(initial_backoff_s=1, backoff_multiplier=2,
                       max_backoff_s=5).backoff(10) == 5.0


def test_gives_up_after_max_retries_and_reports():
    events = []
    stream = make_stream(
        retry_policy=RetryPolicy(max_retries=3, initial_backoff_s=0.01,
                                 reset_after_s=9999),
        on_event=events.append)
    attempts = run_with_failing_source(stream)
    assert stream.state == "failed"
    assert attempts == 4  # initial + 3 retries
    types = [e.type for e in events]
    assert types.count("reconnecting") == 3
    assert types[-1] == "gave_up"
    gave_up = events[-1]
    assert gave_up.attempt == 3 and "net down" in gave_up.error
    assert stream.last_error and "net down" in stream.last_error


def test_recovery_before_limit():
    events = []
    stream = make_stream(
        retry_policy=RetryPolicy(max_retries=5, initial_backoff_s=0.01,
                                 reset_after_s=9999),
        on_event=events.append)
    run_with_failing_source(stream, fail_times=2)
    assert stream.state != "failed"
    types = [e.type for e in events]
    assert types.count("reconnecting") == 2
    assert "gave_up" not in types
    assert stream.restarts == 2


def test_backoff_schedule_reported():
    events = []
    stream = make_stream(
        retry_policy=RetryPolicy(max_retries=3, initial_backoff_s=0.01,
                                 backoff_multiplier=2.0, reset_after_s=9999),
        on_event=events.append)
    run_with_failing_source(stream)
    backoffs = [e.backoff_s for e in events if e.type == "reconnecting"]
    assert backoffs == [0.01, 0.02, 0.04]


def test_broken_callback_does_not_kill_stream():
    def bad_callback(event):
        raise RuntimeError("reporter exploded")

    stream = make_stream(
        retry_policy=RetryPolicy(max_retries=1, initial_backoff_s=0.01),
        on_event=bad_callback)
    run_with_failing_source(stream)
    assert stream.state == "failed"  # completed its cycle despite the callback


def test_stable_connection_resets_attempts(monkeypatch):
    """A long-lived connection resets the counter: with max_retries=1 the
    stream survives repeated single failures separated by stable periods."""
    import avap.streaming as st
    clock = {"t": 0.0}
    monkeypatch.setattr(st.time, "monotonic", lambda: clock["t"])

    stream = make_stream(
        retry_policy=RetryPolicy(max_retries=1, initial_backoff_s=0.01,
                                 reset_after_s=30))
    calls = {"n": 0}

    def fake_process():
        calls["n"] += 1
        clock["t"] += 60  # each connection holds for a "minute", then drops
        if calls["n"] >= 4:
            stream._stop.set()
            return
        raise ConnectionError("blip")

    stream._uri = "rtsp://cam.test/live"
    stream._process_source = fake_process
    stream._run()
    assert stream.state != "failed"
    assert calls["n"] == 4  # survived 3 blips despite max_retries=1


def test_manager_defaults_apply(monkeypatch):
    import avap.streaming as st
    monkeypatch.setattr(st, "probe_devices", lambda: [type("D", (), {
        "has_decode_engine": True, "device_ordinal": 0,
        "drm_render_node": "/dev/dri/renderD999"})()])
    seen = []

    def collector(event):
        seen.append(event)

    policy = RetryPolicy(max_retries=7)
    mgr = st.AMDGPUManager(config={"on_event": collector,
                                   "retry_policy": policy})
    s_default = make_stream()
    s_custom = make_stream(retry_policy=RetryPolicy(max_retries=1),
                           on_event=lambda e: None)
    mgr.add_stream(s_default)
    mgr.add_stream(s_custom)
    assert s_default.retry_policy is policy
    assert s_default.on_event is collector
    assert s_custom.retry_policy.max_retries == 1  # own settings win
    st_status = mgr.status()
    assert set(st_status["cam-test"]) == {"state", "frames_processed",
                                          "restarts", "sink_errors", "last_error"}


def test_sink_error_event_shape():
    e = StreamEvent(type="sink_error", source_id="cam0", error="KafkaDown()")
    assert e.type in ("sink_error",) and e.ts > 0
