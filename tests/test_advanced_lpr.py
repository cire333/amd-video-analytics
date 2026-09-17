import json

import numpy as np
import pytest

from avap.advanced.lpr import (EMBEDDINGS_TO_AVERAGE, LPRConfig, LPRStage,
                               PlateVoter, VehicleReID)
from avap.advanced.meta import AdvFrameMeta, AdvObjectMeta, Classification

DIM = 768


def unit_vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def vehicle(track_id: int, class_id: int = 2, plate: str | None = None,
            plate_conf: float = 0.9, embedding: np.ndarray | None = None):
    obj = AdvObjectMeta(object_id=track_id, component_id=1, class_id=class_id,
                        label="car", confidence=0.9, bbox=(0, 0, 100, 100))
    if plate is not None:
        child = AdvObjectMeta(object_id=None, component_id=2, class_id=0,
                              label="plate", confidence=0.8,
                              bbox=(10, 10, 60, 30), parent=obj)
        child.classifications.append(
            Classification(component_id=3, label=plate, confidence=plate_conf))
        obj.children.append(child)
    if embedding is not None:
        obj.tensors[4] = embedding
    return obj


def frame_with(objs):
    f = AdvFrameMeta(source_id="cam0", frame=0, pts_us=0,
                     frame_width=1920, frame_height=1080)
    f.objects.extend(objs)
    for o in list(objs):
        f.objects.extend(o.children)
    return f


# -- voting ------------------------------------------------------------------

def test_voter_majority_and_min_reads():
    v = PlateVoter()
    assert v.vote("t1") is None
    for read in ["ABC123", "ABC123", "A8C123"]:
        v.accumulate("t1", read)
    assert v.vote("t1", min_reads=3) == ("ABC123", 3)
    assert v.vote("t1", min_reads=4) is None
    v.clear_track("t1")
    assert v.vote("t1") is None


# -- reid ---------------------------------------------------------------------

def test_reid_lookup_or_create_and_merge():
    r = VehicleReID(similarity_threshold=0.88, dim=DIM)
    a = unit_vec(1)
    vid = r.lookup_or_create(a, "t0")
    # a slightly perturbed embedding of the same vehicle matches
    noisy = a + np.random.default_rng(2).normal(scale=0.01, size=DIM).astype(np.float32)
    noisy /= np.linalg.norm(noisy)
    assert r.lookup_or_create(noisy, "t1") == vid
    assert r.get(vid).observation_count == 2
    # an orthogonal-ish embedding creates a new vehicle
    assert r.lookup_or_create(unit_vec(99), "t2") != vid
    assert len(r) == 2


def test_reid_plate_registry_and_persistence(tmp_path):
    r = VehicleReID(dim=DIM)
    vid = r.lookup_or_create(unit_vec(5), "t0")
    r.update_plate(vid, "XYZ789", "t1")
    r.update_plate(vid, "SHOULD-NOT-OVERWRITE", "t2")
    assert r.get(vid).plate_text == "XYZ789"
    r.save(str(tmp_path))
    data = json.loads((tmp_path / "embeddings.json").read_text())
    assert data[0]["vehicle_id"] == vid

    r2 = VehicleReID(dim=DIM)
    r2.load(str(tmp_path))
    assert len(r2) == 1
    assert r2.lookup_or_create(unit_vec(5), "t3") == vid  # index rebuilt


# -- stage ---------------------------------------------------------------------

def test_lpr_stage_votes_and_assigns_vehicle_id():
    stage = LPRStage(LPRConfig(min_plate_reads=3, similarity_threshold=0.88))
    emb = unit_vec(7)
    # 2 reads: below min_reads -> no plate yet
    for i in range(2):
        stage.process(frame_with([vehicle(11, plate="GMX100", embedding=emb)]), "t")
    assert stage.get_result(11, "cam0").plate_text is None
    # 3rd read crosses min_reads
    stage.process(frame_with([vehicle(11, plate="GMX100", embedding=emb)]), "t")
    assert stage.get_result(11, "cam0").plate_text == "GMX100"
    # vehicle id appears after enough embeddings are accumulated
    for _ in range(EMBEDDINGS_TO_AVERAGE):
        stage.process(frame_with([vehicle(11, plate="GMX100", embedding=emb)]), "t")
    res = stage.get_result(11, "cam0")
    assert res.vehicle_id is not None
    # plate propagated to the reid registry
    assert stage.reid.get(res.vehicle_id).plate_text == "GMX100"


def test_lpr_stage_low_confidence_reads_ignored():
    stage = LPRStage(LPRConfig(min_plate_reads=1,
                               plate_confidence_threshold=0.75))
    stage.process(frame_with([vehicle(5, plate="BAD999", plate_conf=0.4)]), "t")
    assert stage.get_result(5, "cam0").plate_text is None


def test_lpr_stage_same_vehicle_two_tracks_reidentified(tmp_path):
    """A vehicle leaves and returns as a new track id; re-ID gives it the
    same vehicle_id — the core LPR requirement."""
    cfg = LPRConfig(min_plate_reads=3, embeddings_dir=str(tmp_path))
    stage = LPRStage(cfg)
    emb = unit_vec(42)
    for _ in range(EMBEDDINGS_TO_AVERAGE):
        stage.process(frame_with([vehicle(1, plate="GMX200", embedding=emb)]), "t")
    first = stage.get_result(1, "cam0").vehicle_id
    stage.save()

    stage2 = LPRStage(cfg)  # fresh process, loads persisted embeddings
    for _ in range(EMBEDDINGS_TO_AVERAGE):
        stage2.process(frame_with([vehicle(77, embedding=emb)]), "t")
    assert stage2.get_result(77, "cam0").vehicle_id == first


def test_non_vehicle_classes_skipped():
    stage = LPRStage(LPRConfig())
    person = vehicle(9, class_id=0, plate="NOPE", embedding=unit_vec(3))
    stage.process(frame_with([person]), "t")
    assert stage.get_result(9, "cam0") is None
