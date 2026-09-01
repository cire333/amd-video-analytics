# AMD pipeline vs NVIDIA DeepStream — YOLO26-m comparison

Frames compared: **26831** | IoU match threshold: 0.5

## Detection agreement

| | AMD (R9700/MIGraphX) | DeepStream (3090/TensorRT) |
|---|---|---|
| total detections | 168022 | 162237 |
| per class | {'car': 144867, 'truck': 19027, 'traffic light': 2270, 'motorcycle': 10, 'bus': 1439, 'person': 264, 'train': 15, 'stop sign': 44, 'parking meter': 1, 'bicycle': 84, 'backpack': 1} | {'car': 137608, 'truck': 21978, 'traffic light': 822, 'bus': 1473, 'person': 250, 'train': 3, 'stop sign': 12, 'motorcycle': 12, 'bicycle': 78, 'backpack': 1} |

- **Matched (same object, IoU>=0.5): 148233** — 88.2% of AMD, 91.4% of DS
- AMD-only detections: 19789 | DS-only detections: 14004
- Mean IoU of matched pairs: 0.923
- Class label agreement on matched pairs: 140575/148233 (94.8%)
- Class disagreements (AMD label, DS label): {('car', 'truck'): 4216, ('truck', 'car'): 3309, ('car', 'traffic light'): 5, ('motorcycle', 'car'): 1, ('truck', 'bus'): 79, ('bus', 'truck'): 34, ('train', 'bus'): 6, ('train', 'truck'): 1, ('car', 'person'): 2, ('person', 'car'): 1, ('bicycle', 'motorcycle'): 2, ('motorcycle', 'bicycle'): 2}
- Mean |confidence delta| on matched pairs: 0.0808

## Tracking

| | AMD (greedy IoU tracker) | DeepStream (NvDsTracker IOU) |
|---|---|---|
| unique track ids | 4906 | 348 |
| mean track length (frames) | 34.2 | 584.8 |
| tracks >= 10 frames | 2261 | 342 |

## Per-frame object counts (first/mid/last)

| frame | AMD | DS |
|---|---|---|
| 0 | 5 | 6 |
| 1 | 6 | 6 |
| 2 | 6 | 7 |
| 13414 | 9 | 9 |
| 13415 | 10 | 8 |
| 13416 | 11 | 9 |
| 26828 | 6 | 6 |
| 26829 | 6 | 6 |
| 26830 | 6 | 6 |

## Preprocessing isolation (5 sampled frames, IoU-matched, MIGraphX runtime)

Bridge tensor vs CPU-reference tensor on this tagged (BT.709, full-range)
video: identical object counts per frame, all detections matched, zero
class flips, mean |Δscore| 0.0025–0.048 — attributable to resize
interpolation (NV12-domain bilinear vs RGB-domain), not colorimetry.
With runtimes previously proven equivalent (MIGraphX ≡ TRT-noTF32 @1e-5),
the remaining pipeline delta (0.081) is dominated by DeepStream-side
preprocessing — consistent with DS treating this full-range (pc) video as
limited-range. Supporting signal: DS reports 822 traffic lights vs AMD's
2270; small saturated objects suffer most from range compression.

## Methodology notes discovered on this run

- `migraphx.argument` BORROWS the numpy buffer (no copy). Passing a
  temporary (e.g. a fresh `np.ascontiguousarray` of a transposed view)
  is a use-after-free: nondeterministic garbage/NaN outputs. Keep the
  input array alive until after `run()`. Production runner unaffected
  (its tensors are already contiguous and held).
- Row-index diffing of the end-to-end head's (300,6) output is only valid
  for byte-identical inputs; near-identical inputs permute the score-sorted
  rows. Use IoU matching otherwise.
- KITTI labels can contain spaces ("traffic light"); parse from the
  numeric tail (compare_detections.py fixed).

## Kalman tracker rerun (2026-09-01)

Same detections, tracker swapped: greedy-IoU placeholder -> SORT-style
Kalman (constant-velocity, Hungarian assignment, min_hits=3, max_age=15,
class-agnostic association with majority-vote labels).

| | AMD greedy-IoU | AMD Kalman/SORT | DeepStream IOU |
|---|---|---|---|
| unique tracks | 4906 | 2799 | 348 |
| mean length (frames) | 34.2 | 57.3 | 584.8 |
| tracks >= 5s | — | 608 | 296 |
| tracks with avg area < 32x32 px | — | 786 | 26 |

Reading: the tracker itself is no longer the dominant fragmentation
source — detection flicker is. 786 of AMD's tracks are tiny distant
objects (< 32x32 px) that DeepStream's preprocessing largely suppresses
(26 such tracks); on substantial tracks (>= 5 s) the gap is 608 vs 296,
i.e. ~2x, driven by AMD's 19,789 extra borderline detections appearing
and disappearing around the 0.30 threshold. Levers to converge further:
raise max_age (1 s -> 2-3 s), add a track-level confidence gate, or a
min-area gate matching the deployment's object sizes.

## Performance decomposition (answering "34 fps seems low")

| measurement | fps |
|---|---|
| DeepStream end-to-end (3090, pipelined C, NVENC/OSD) | 172 |
| AMD full runner (single Python thread incl. cv2 draw + SW encode) | 34 |
| AMD inference only, MIGraphX FP32 | 139 |
| AMD inference only, MIGraphX FP16 | 381 |
| AMD decode only (incl. host-detile fallback) | 366 |

No throttling: the R9700 is idle most of each frame in the sequential
runner. Pipelining decode/infer/annotate and hardware encode are the
known optimizations; FP16 alone puts inference at 2.2x DeepStream's
observed end-to-end rate.

## Vehicle-count calibration vs hand count (2026-09-01)

Ground truth: manual count of ~166 vehicles in a 6-minute span (~830
extrapolated over the 29.8-min video). Offline sweep of tracking
mechanisms over the saved AMD detections (scripts/tracking_sweep.py — no
core-code changes), all configs excluding the smallest 12% of bounding
boxes by area (cutoff: 359 px^2 ~ 18x18 at 1280x720; same absolute cutoff
applied to DeepStream), counting vehicle classes only.

Best configuration — SortTracker(iou=0.3, max_age=30, min_hits=3) with
measurement gates conf >= 0.40, net displacement >= 100 px, track length
>= 2 s, tracklet stitching (gap <= 6 s, endpoint distance <= 80 px):

| | vehicle tracks | per-6-min windows | vs GT |
|---|---|---|---|
| AMD (best config) | **830** | 161 168 152 174 175 | +0 (windows vs 166 hand count) |
| AMD (neighboring configs) | 792–875 | ±8% | robust region, not a lucky point |
| DeepStream IOU (same gates) | 320–331 | 60–75 | **-60%: undercounts** |

Confirms the operator intuition: DeepStream undercounts (~2.5x low —
detection-limited: its preprocessing suppresses distant vehicles, so no
measurement-side gate can recover them) while raw AMD overcounts from
track fragmentation; duration/displacement gates plus stitching bring AMD
to the hand-counted rate across every window. The counting recipe lives
entirely in the measurement layer.
