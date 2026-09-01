import json

import pytest

from avap.bytetrack import ByteTracker
from avap.frame import ObjectMeta
from avap.roi import RoiConfig
from avap.sinks import Formatter, LocalFileSink, frame_record, make_sink
from avap.streaming import (AMDStream, _make_roi, _make_tracker,
                            resolve_data_location)


def det(x1, y1, x2, y2, conf=0.9, label="car"):
    return ObjectMeta(class_id=2, confidence=conf, bbox=(x1, y1, x2, y2),
                      label=label)


# -- config validation ---------------------------------------------------

def test_data_location_urls_pass_through():
    assert resolve_data_location("rtsp://cam/live") == "rtsp://cam/live"
    assert resolve_data_location("https://x/y.mp4") == "https://x/y.mp4"


def test_data_location_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        resolve_data_location("/no/such/file.mp4")


def test_roi_accepts_bbox_and_polygon():
    r = _make_roi((0.1, 0.2, 0.9, 0.8))
    assert isinstance(r, RoiConfig) and len(r.polygon_norm) == 4
    poly = _make_roi([(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])
    assert len(poly.polygon_norm) == 3
    assert _make_roi(None) is None


def test_tracker_registry_and_byo():
    assert type(_make_tracker("iou")).__name__ == "IouTracker"
    assert type(_make_tracker("sort")).__name__ == "SortTracker"
    assert type(_make_tracker("bytetrack")).__name__ == "ByteTracker"

    class Custom:
        def update(self, dets):
            return dets
    assert isinstance(_make_tracker(Custom()), Custom)
    with pytest.raises(ValueError):
        _make_tracker("deepsort-9000")


def test_stream_config_validation():
    with pytest.raises(ValueError):
        AMDStream("rtsp://x", batch_size=0)
    with pytest.raises(ValueError):
        AMDStream("rtsp://x", frame_sample_rate=-1)
    with pytest.raises(ValueError):
        AMDStream("rtsp://x", model_quant="fp8")


# -- formatters / sinks ----------------------------------------------------

def rec():
    return frame_record("cam0", 7, 466_000, [
        {"track_id": 3, "label": "car", "class_id": 2, "conf": 0.91,
         "bbox": [10.0, 20.0, 110.0, 90.0]}])


def test_json_and_template_formats():
    assert json.loads(Formatter("json").lines(rec())[0])["n_objects"] == 1
    line = Formatter("json", "{source_id}|{frame}|{n_objects}").lines(rec())[0]
    assert line == "cam0|7|1"


def test_csv_one_row_per_object(tmp_path):
    p = tmp_path / "out.csv"
    sink = LocalFileSink(str(p), Formatter("csv"))
    sink.emit(rec())
    sink.close()
    lines = p.read_text().strip().splitlines()
    assert lines[0].startswith("source_id,")
    assert lines[1].split(",")[:2] == ["cam0", "7"]


def test_make_sink_dispatch(tmp_path):
    s = make_sink(str(tmp_path / "d.jsonl"))
    assert type(s).__name__ == "LocalFileSink"
    s.close()


def test_kinesis_uri_parsing(monkeypatch):
    import avap.sinks as sinks
    sent = []

    class FakeClient:
        def put_record(self, **kw):
            sent.append(kw)

    class FakeBoto3:
        @staticmethod
        def client(service, **kw):
            assert service == "kinesis"
            FakeBoto3.region = kw.get("region_name")
            return FakeClient()

    monkeypatch.setitem(__import__("sys").modules, "boto3", FakeBoto3)
    s = sinks.KinesisSink("kinesis://us-west-2/detections", Formatter("json"))
    assert s.stream == "detections" and FakeBoto3.region == "us-west-2"
    s2 = sinks.KinesisSink("kinesis://detections", Formatter("json"))
    assert s2.stream == "detections" and FakeBoto3.region is None
    s2.emit(rec())
    assert sent[0]["StreamName"] == "detections"
    assert sent[0]["PartitionKey"] == "cam0"
    with pytest.raises(ValueError):
        sinks.KinesisSink("kinesis://", Formatter("json"))


def test_parquet_local(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    p = tmp_path / "out.parquet"
    sink = LocalFileSink(str(p), Formatter("parquet"))
    sink.emit(rec())
    sink.close()
    assert pq.read_table(str(p)).num_rows == 1


# -- bytetrack -------------------------------------------------------------

def test_bytetrack_low_conf_rescues_track():
    tr = ByteTracker(min_hits=1, high_conf=0.5, low_conf=0.1)
    [a] = tr.update([det(100, 100, 200, 200, conf=0.9)])
    # detector flickers low during partial occlusion: SORT would drop this
    [b] = tr.update([det(104, 102, 204, 202, conf=0.2)])
    assert b.track_id == a.track_id


def test_bytetrack_low_conf_never_spawns():
    tr = ByteTracker(min_hits=1)
    [o] = tr.update([det(100, 100, 200, 200, conf=0.2)])
    assert o.track_id is None
    assert tr.update([det(500, 500, 600, 600, conf=0.2)])[0].track_id is None


# -- gpu manager failure isolation ------------------------------------------

class StubStream:
    def __init__(self, name, fail=False):
        self.source_id = name
        self.model_name, self.model_quant = "stub", "fp16"
        self.tracker = object()
        self.state = "configured"
        self.frames_processed = 0
        self._fail = fail

    def start_stream(self):
        if self._fail:
            self.state = "failed"
            raise RuntimeError("out of VRAM")
        self.state = "running"


def test_manager_start_isolation(monkeypatch):
    import avap.streaming as st
    monkeypatch.setattr(st, "probe_devices", lambda: [type("D", (), {
        "has_decode_engine": True, "device_ordinal": 0,
        "drm_render_node": "/dev/dri/renderD999"})()])
    mgr = st.AMDGPUManager(device_id=0, config={"start_stagger_s": 0})
    ok1, bad, ok2 = StubStream("a"), StubStream("b", fail=True), StubStream("c")
    for s in (ok1, bad, ok2):
        mgr.add_stream(s)
    mgr.start_streams()
    assert ok1.state == "running"          # unaffected by b's failure
    assert bad.state == "failed"
    assert ok2.state == "configured"       # left pending, not failed
    st_status = mgr.status()
    assert st_status["a"]["state"] == "running"
