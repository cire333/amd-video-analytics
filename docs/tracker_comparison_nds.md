# Tracker strategy comparison — NDS customer footage (2026-09-02)

Offline replay of saved detections (13 cameras x ~45k frames, nds13-yolo26m
@1280) through six strategies. Metrics aggregate all 13 videos; frag/min =
track deaths followed <=2 s later by a nearby (<=80 px) birth — breaks the
tracker failed to bridge. Visual check: results/nds/tracker_grid_160850.mp4
(2x2, same 45 s dense-crosswalk segment).

| strategy | tracks | >=5s | med len | mean len | frag/min | births/min |
|---|---|---|---|---|---|---|
| iou (greedy)            | 110,459 | 13,139 | 14.2 | 75.2  | 235.0 | 282.7 |
| sort (age 0.6s)         | 73,551  | 12,834 | 27.3 | 110.1 | 146.3 | 188.3 |
| bytetrack (age 1s)      | 42,728  | 11,012 | 41.7 | 183.2 | 70.1  | 109.4 |
| bytetrack-long (2.5s,h5)| 32,040  | 10,187 | 60.3 | 252.9 | 43.2  | 82.0  |
| ocsort (2.5s)           | 35,529  | 10,530 | 53.9 | 231.3 | 50.0  | 90.9  |
| **ocsort-long (3.6s,h5)**| **31,105** | 10,203 | **66.3** | **262.9** | **42.0** | **79.6** |

Reading:
- Total tracks drop 3.5x from greedy IoU to ocsort-long while >=5s track
  counts stay ~stable (13.1k -> 10.2k): the reduction is short-fragment
  elimination, not distinct objects being merged.
- Occlusion memory is the single biggest lever (bytetrack -> bytetrack-long).
  OC-SORT's velocity-consistency + re-update adds a further edge at equal
  memory and wins overall.
- Recommended for intersection deployments:
  `OcSortTracker(iou_threshold=0.25, max_age=~3.5*fps, min_hits=5)`;
  `ByteTracker(max_age=2.5*fps, min_hits=5)` is within a few percent and
  ~30% cheaper per frame if CPU-bound.
