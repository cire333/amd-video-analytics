import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

import avap.advanced.pipeline as pl
from avap.advanced import (AdvancedPipeline, ClassMapper, DataPublisher,
                           FileTransport, LPRConfig, LPRStage)
from avap.advanced.meta import AdvObjectMeta, Classification


class FakePrimary:
    """Detects one 'car' covering a fixed region of every frame."""

    def __init__(self, component_id=1):
        self.config = SimpleNamespace(component_id=component_id)
        self.device_ordinal = 0
        self.prepared = False

    def prepare(self):
        self.prepared = True

    def run(self, rgb, frame, dev=None):
        h, w = rgb.shape[:2]
        frame.objects.append(AdvObjectMeta(
            object_id=None, component_id=self.config.component_id,
            class_id=2, label="car", confidence=0.95,
            bbox=(w * 0.25, h * 0.25, w * 0.75, h * 0.75)))


class FakePlateDetect:
    def __init__(self, component_id=2):
        self.config = SimpleNamespace(component_id=component_id)
        self.device_ordinal = 0
        self.name = "plate_detect"

    def prepare(self):
        pass

    def run(self, rgb, frame, dev=None):
        for parent in [o for o in frame.objects if o.component_id == 1]:
            child = AdvObjectMeta(object_id=None, component_id=2, class_id=0,
                                  label="plate", confidence=0.9,
                                  bbox=(parent.bbox[0] + 5, parent.bbox[1] + 5,
                                        parent.bbox[0] + 40, parent.bbox[1] + 20),
                                  parent=parent)
            parent.children.append(child)
            frame.objects.append(child)


class FakeOcr:
    def __init__(self, component_id=3):
        self.config = SimpleNamespace(component_id=component_id)
        self.device_ordinal = 0
        self.name = "plate_ocr"

    def prepare(self):
        pass

    def run(self, rgb, frame, dev=None):
        for obj in [o for o in frame.objects if o.component_id == 2]:
            obj.classifications.append(
                Classification(component_id=3, label="GMX999", confidence=0.9))


class FakeEmbed:
    def __init__(self, component_id=4, dim=768):
        self.config = SimpleNamespace(component_id=component_id)
        self.device_ordinal = 0
        self.name = "dinov2"
        self._vec = np.random.default_rng(0).normal(size=dim).astype(np.float32)
        self._vec /= np.linalg.norm(self._vec)

    def prepare(self):
        pass

    def run(self, rgb, frame, dev=None):
        for obj in [o for o in frame.objects if o.component_id == 1]:
            obj.tensors[4] = self._vec


@pytest.fixture(autouse=True)
def fake_device(monkeypatch):
    dev = SimpleNamespace(device_ordinal=0, drm_render_node="/dev/dri/renderD999",
                          has_decode_engine=True)
    monkeypatch.setattr(pl, "probe_devices", lambda: [dev])
    monkeypatch.setattr(pl, "require_decode_device", lambda devices: devices[0])


def rgb_frame(w=640, h=360):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_pipeline(tmp_path, **kw):
    sent = []
    pub = DataPublisher(lambda payload, cam: sent.append((cam, payload)))
    pipe = AdvancedPipeline(
        primary=FakePrimary(),
        stages=[FakePlateDetect(), FakeOcr(), FakeEmbed()],
        tracker="sort",
        lpr=LPRStage(LPRConfig(min_plate_reads=2)),
        publisher=pub,
        device_resident=False,  # stub stages, no GPU in unit tests
        **kw)
    return pipe, sent


def drive(pipe, index, n_frames, w=640, h=360):
    for i in range(n_frames):
        pipe.process_image(index, rgb_frame(w, h), pts_us=i * 40_000)


def test_full_cascade_and_lpr(tmp_path):
    pipe, sent = make_pipeline(tmp_path)
    idx = pipe.add_source("mem://", camera_id="cam0", fps=25,
                          batch_size_seconds=1)  # batch_size=25
    pipe.prepare()
    drive(pipe, idx, 40)
    frame = pipe.process_image(idx, rgb_frame())
    [vehicle] = frame.primary_objects()
    assert vehicle.object_id is not None                 # tracked
    assert vehicle.children and vehicle.children[0].label == "plate"
    assert vehicle.children[0].classification_from(3).label == "GMX999"
    assert vehicle.tensors[4].shape == (768,)
    res = pipe.lpr.get_result(vehicle.object_id, "cam0")
    assert res.plate_text == "GMX999"
    assert res.vehicle_id is not None                    # 30+ embeddings seen
    pipe.publisher.disable()
    assert sent, "publisher flushed at least one batch"
    cam, payload = sent[0]
    records = json.loads(payload)
    assert cam == "cam0" and len(records) == 25          # batch_size respected
    # first frames pre-confirmation publish empty Data; find a populated one
    obj = next(r["Data"][0] for r in records if r["Data"])
    assert obj["class"] == "light automobile"


def test_probe_points_and_order(tmp_path):
    pipe, _ = make_pipeline(tmp_path)
    order = []
    for point in pipe.probe_points():
        pipe.add_probe(point, lambda f, r, p=point: order.append(p))
    with pytest.raises(ValueError):
        pipe.add_probe("nonexistent", lambda f, r: None)
    idx = pipe.add_source("mem://", camera_id="cam0")
    pipe.prepare()
    pipe.process_image(idx, rgb_frame())
    assert order == ["primary", "tracker", "plate_detect", "plate_ocr",
                     "dinov2", "lpr"]


def test_broken_probe_does_not_kill_frame(tmp_path):
    pipe, _ = make_pipeline(tmp_path)

    def bad(frame, rgb):
        raise RuntimeError("probe exploded")
    pipe.add_probe("tracker", bad)
    idx = pipe.add_source("mem://", camera_id="cam0")
    pipe.prepare()
    frame = pipe.process_image(idx, rgb_frame())
    assert frame.objects  # processing completed


def test_bbox_rescaled_to_image_resolution(tmp_path):
    pipe, sent = make_pipeline(tmp_path)
    idx = pipe.add_source("mem://", camera_id="cam0",
                          image_resolution=(1280, 720),
                          fps=1, batch_size_seconds=1)   # flush every frame
    pipe.prepare()
    drive(pipe, idx, 5, w=640, h=360)                    # decode res 640x360
    pipe.publisher.disable()
    all_records = [r for _, payload in sent for r in json.loads(payload)]
    obj = next(r["Data"][0] for r in all_records if r["Data"])
    # detector box (160,90,480,270) in 640x360 -> x2 in 1280x720
    assert obj["x1"] == 320.0 and obj["x2"] == 960.0
    assert obj["y1"] == 180.0 and obj["y2"] == 540.0


def test_source_index_recycling_and_isolation(tmp_path):
    pipe, _ = make_pipeline(tmp_path)
    a = pipe.add_source("mem://a", camera_id="camA")
    b = pipe.add_source("mem://b", camera_id="camB")
    assert (a, b) == (0, 1)
    pipe.prepare()
    drive(pipe, a, 5)
    drive(pipe, b, 5)
    # per-source trackers: same detector box, but ids are independent
    fa = pipe.process_image(a, rgb_frame())
    fb = pipe.process_image(b, rgb_frame())
    assert fa.primary_objects()[0].object_id == fb.primary_objects()[0].object_id
    assert fa.source_id != fb.source_id
    pipe.remove_source(a)
    assert pipe.add_source("mem://c", camera_id="camC") == 0  # index recycled


def test_publisher_lossy_queue_and_stats():
    slow = DataPublisher(lambda p, c: time.sleep(10), max_size=2)
    slow.register_source(0, "cam0", batch_size=1)
    from avap.advanced.records import DataRecord
    for _ in range(10):  # worker not enabled: queue fills, records drop
        slow.publish(0, DataRecord(data=[], frame=0, stream=0, time="0", fps=0))
    stats = slow.stats()
    assert stats["queue_full"] == 8
    assert stats["sources"][0]["camera_id"] == "cam0"


def test_file_transport_layout(tmp_path):
    t = FileTransport(str(tmp_path))
    t('[{"Frame": 1}]', "camX")
    t('[{"Frame": 2}]', "camX")
    files = sorted((tmp_path / "source_camX").glob("batch_*.json"))
    assert len(files) == 2
    assert json.loads(files[0].read_text())[0]["Frame"] == 1


def test_perf_fps_lazily_created(tmp_path):
    pipe, _ = make_pipeline(tmp_path)
    idx = pipe.add_source("mem://", camera_id="late-cam")
    pipe.prepare()
    pipe.process_image(idx, rgb_frame())
    snap = pipe.perf.snapshot()
    assert "late-cam" in snap and snap["late-cam"]["idle_s"] is not None
