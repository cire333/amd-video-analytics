from avap.frame import ObjectMeta
from avap.tracker import IouTracker, TrackerBank


def det(x1, y1, x2, y2, conf=0.9):
    return ObjectMeta(class_id=0, confidence=conf, bbox=(x1, y1, x2, y2))


def test_track_id_stable_across_frames():
    tr = IouTracker()
    [a] = tr.update([det(100, 100, 200, 200)])
    [b] = tr.update([det(105, 102, 205, 202)])  # small motion
    assert a.track_id == b.track_id


def test_new_object_gets_new_id():
    tr = IouTracker()
    [a] = tr.update([det(100, 100, 200, 200)])
    objs = tr.update([det(102, 101, 202, 201), det(500, 500, 600, 600)])
    ids = {o.track_id for o in objs}
    assert a.track_id in ids and len(ids) == 2


def test_bank_isolates_sources():
    bank = TrackerBank()
    [a] = bank.update("cam0", [det(100, 100, 200, 200)])
    # same bbox on a different source must not steal cam0's track
    [b] = bank.update("cam1", [det(100, 100, 200, 200)])
    [a2] = bank.update("cam0", [det(101, 101, 201, 201)])
    assert a.track_id == a2.track_id
    # removing a source doesn't disturb others
    bank.remove("cam1")
    [a3] = bank.update("cam0", [det(102, 102, 202, 202)])
    assert a3.track_id == a.track_id
