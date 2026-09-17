import numpy as np

from avap.advanced.lpr import LPRResult
from avap.advanced.meta import AdvFrameMeta, AdvObjectMeta, Classification
from avap.advanced.records import (ClassMapper, DataRecord,
                                   build_object_record)
from avap.advanced.stages import (InferConfig, _Stage,
                                  default_classifier_parser,
                                  default_detection_parser)


def obj(component=1, class_id=2, oid=1, bbox=(0, 0, 100, 100)):
    return AdvObjectMeta(object_id=oid, component_id=component,
                         class_id=class_id, label="car", confidence=0.9,
                         bbox=bbox)


# -- meta hierarchy ----------------------------------------------------------

def test_best_child_classification_picks_highest():
    parent = obj()
    for conf, text in [(0.6, "AAA111"), (0.9, "BBB222"), (0.7, "CCC333")]:
        child = AdvObjectMeta(object_id=None, component_id=2, class_id=0,
                              label="plate", confidence=0.8,
                              bbox=(0, 0, 10, 10), parent=parent)
        child.classifications.append(
            Classification(component_id=3, label=text, confidence=conf))
        parent.children.append(child)
    best = parent.best_child_classification(2, 3)
    assert best.label == "BBB222"


def test_primary_objects_excludes_children():
    f = AdvFrameMeta("cam0", 0, 0, 1920, 1080)
    parent = obj()
    child = AdvObjectMeta(object_id=None, component_id=2, class_id=0,
                          label="plate", confidence=0.8, bbox=(0, 0, 5, 5),
                          parent=parent)
    f.objects.extend([parent, child])
    assert f.primary_objects() == [parent]


# -- records: wire compatibility ------------------------------------------

def test_object_record_wire_format_matches_ingestion_schema():
    rec = build_object_record(
        obj(), 12.34, ClassMapper(),
        lpr=LPRResult(vehicle_id="v_abc123", plate_text="GMX100",
                      plate_confidence=0.91, plate_read_count=5))
    d = rec.to_dict()
    # exact keys + casing of the DeepStream ingestion system
    assert d["class"] == "light automobile"      # mapped taxonomy
    assert d["id"] == 1
    assert d["confidence_score"] == "0.9"        # stringified, as upstream
    assert d["frame_timestamp"] == 12.34
    assert d["plate_text"] == "GMX100" and d["plate_read_count"] == 5
    assert "embedding" not in d                  # excluded unless requested
    assert set(d) == {"class", "id", "x1", "x2", "y1", "y2",
                      "confidence_score", "frame_timestamp", "vehicle_id",
                      "plate_text", "plate_confidence", "plate_read_count"}


def test_data_record_envelope():
    rec = build_object_record(obj(), 1.0, ClassMapper())
    d = DataRecord(data=[rec], frame=7, stream="cam0", time="2026-09-02T00:00:00Z",
                   fps=24.5, metadata={"id": "cam0"}).to_dict()
    assert set(d) == {"Data", "Frame", "Stream", "Time", "FPS", "Metadata"}
    assert d["Data"][0]["class"] == "light automobile"


def test_class_mapper_modes():
    assert ClassMapper().map("unknown-thing") == "unknown-thing"
    assert ClassMapper(drop_unmapped=True).map("unknown-thing") is None
    rec = build_object_record(
        AdvObjectMeta(object_id=1, component_id=1, class_id=0,
                      label="unknown-thing", confidence=0.9, bbox=(0, 0, 1, 1)),
        0.0, ClassMapper(drop_unmapped=True))
    assert rec is None


def test_untracked_objects_not_published():
    untracked = obj(oid=None)
    assert build_object_record(untracked, 0.0, ClassMapper()) is None


# -- stage targeting / parsers ------------------------------------------------

def test_stage_target_selection():
    cfg = InferConfig(model="m.onnx", component_id=2, operate_on_component=1,
                      operate_on_class_ids=frozenset({2, 7}),
                      min_object_width=20, min_object_height=20)
    stage = _Stage(cfg)
    f = AdvFrameMeta("cam0", 0, 0, 1920, 1080)
    good = obj(component=1, class_id=2, bbox=(0, 0, 100, 100))
    wrong_class = obj(component=1, class_id=0)
    wrong_component = obj(component=9, class_id=2)
    too_small = obj(component=1, class_id=2, bbox=(0, 0, 10, 10))
    f.objects.extend([good, wrong_class, wrong_component, too_small])
    assert stage._targets(f) == [good]


def test_default_detection_parser_thresholds():
    out = np.array([[[0, 0, 10, 10, 0.9, 2], [0, 0, 5, 5, 0.1, 0]]],
                   dtype=np.float32)
    dets = list(default_detection_parser(out, 0.5))
    assert len(dets) == 1 and dets[0][2] == 2


def test_default_classifier_parser():
    out = np.array([[0.1, 3.0, 0.2]], dtype=np.float32)
    label, conf = default_classifier_parser(out, ["a", "b", "c"])
    assert label == "b" and conf > 0.8
