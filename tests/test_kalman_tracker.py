from avap.frame import ObjectMeta
from avap.kalman_tracker import SortTracker


def det(x1, y1, x2, y2, conf=0.9, label="car", cls=2):
    return ObjectMeta(class_id=cls, confidence=conf, bbox=(x1, y1, x2, y2),
                      label=label)


def test_confirmation_after_min_hits():
    tr = SortTracker(min_hits=3)
    assert tr.update([det(100, 100, 200, 200)])[0].track_id is None
    assert tr.update([det(102, 101, 202, 201)])[0].track_id is None
    assert tr.update([det(104, 102, 204, 202)])[0].track_id is not None


def test_id_stable_through_motion():
    tr = SortTracker(min_hits=1)
    ids = set()
    for i in range(10):
        [o] = tr.update([det(100 + 10 * i, 100, 200 + 10 * i, 200)])
        ids.add(o.track_id)
    assert len(ids) == 1  # constant-velocity prediction follows the motion


def test_survives_occlusion_gap():
    tr = SortTracker(min_hits=1, max_age=5)
    [a] = tr.update([det(100, 100, 200, 200)])
    tr.update([])  # 3 missed frames (occlusion)
    tr.update([])
    tr.update([])
    [b] = tr.update([det(112, 100, 212, 200)])
    assert b.track_id == a.track_id


def test_track_dies_after_max_age():
    tr = SortTracker(min_hits=1, max_age=2)
    [a] = tr.update([det(100, 100, 200, 200)])
    for _ in range(4):
        tr.update([])
    [b] = tr.update([det(100, 100, 200, 200)])
    assert b.track_id != a.track_id


def test_class_flicker_does_not_split_track():
    tr = SortTracker(min_hits=1)
    [a] = tr.update([det(100, 100, 300, 250, label="car")])
    [b] = tr.update([det(102, 101, 302, 251, label="truck")])  # flicker
    [c] = tr.update([det(104, 102, 304, 252, label="car")])
    assert a.track_id == b.track_id == c.track_id
    assert c.label == "car"  # majority vote


def test_two_crossing_objects_keep_ids():
    tr = SortTracker(min_hits=1)
    a0, b0 = det(0, 0, 50, 50), det(300, 0, 350, 50)
    ids = [o.track_id for o in tr.update([a0, b0])]
    for i in range(1, 6):
        objs = tr.update([det(0 + 20 * i, 0, 50 + 20 * i, 50),
                          det(300 - 20 * i, 0, 350 - 20 * i, 50)])
        assert [o.track_id for o in objs] == ids
