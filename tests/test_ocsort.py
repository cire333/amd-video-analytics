from avap.frame import ObjectMeta
from avap.ocsort import OcSortTracker


def det(x1, y1, x2, y2, conf=0.9, label="car"):
    return ObjectMeta(class_id=2, confidence=conf, bbox=(x1, y1, x2, y2),
                      label=label)


def test_id_stable_through_motion():
    tr = OcSortTracker(min_hits=1)
    ids = set()
    for i in range(10):
        [o] = tr.update([det(100 + 15 * i, 100, 200 + 15 * i, 200)])
        ids.add(o.track_id)
    assert len(ids) == 1


def test_oru_recovers_after_gap():
    tr = OcSortTracker(min_hits=1, max_age=20)
    [a] = tr.update([det(100, 100, 200, 200)])
    tr.update([det(115, 100, 215, 200)])
    for _ in range(6):          # occluded: no detections
        tr.update([])
    # reappears far along its trajectory
    [b] = tr.update([det(220, 100, 320, 200)])
    assert b.track_id == a.track_id
    # after ORU the filter tracks smoothly from the recovery point
    [c] = tr.update([det(235, 100, 335, 200)])
    assert c.track_id == a.track_id


def test_velocity_consistency_prefers_heading():
    """Two identical-size objects pass close by; each keeps the id whose
    observed heading matches its motion."""
    tr = OcSortTracker(min_hits=1, velocity_weight=0.4)
    # a moves right, b moves left (20 px/frame), vertically adjacent lanes;
    # they pass each other around x ~ 170
    seq = [([20 * i, 0, 60 + 20 * i, 60],
            [340 - 20 * i, 40, 400 - 20 * i, 100]) for i in range(10)]
    ids = None
    for a_box, b_box in seq:
        objs = tr.update([det(*a_box), det(*b_box)])
        if ids is None:
            ids = [o.track_id for o in objs]
        else:
            assert [o.track_id for o in objs] == ids


def test_low_conf_never_spawns():
    tr = OcSortTracker(min_hits=1)
    [o] = tr.update([det(0, 0, 50, 50, conf=0.2)])
    assert o.track_id is None
