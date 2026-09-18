# NvDCF-class visual tracker on AMD

Date: 2026-09-16. Branch `cire333/nvdcf-tracker`. Code: `src/avap/nvdcf.py`, `cpp/dcf.hip`,
`cpp/dcf.h`; evaluation `scripts/nvdcf_comparison.py`; tests `tests/test_nvdcf.py`.

## Why

DeepStream's NvDCF tracker is the last DeepStream component without an avap equivalent. Our
trackers (SORT, ByteTrack, OC-SORT) are motion-only: when the detector misses an object for a
few frames the track goes blind, and when two similar boxes cross, IoU + velocity cannot tell
them apart. NvDCF adds an appearance term — a discriminative correlation filter (DCF) per target
— and uses it both to localize the object when the detector fails and to score data association.
NVIDIA's optional optical-flow input (hardware OFA) has no AMD equivalent and is not part of the
core NvDCF algorithm; it is not reproduced.

## What NvDCF does (from the DeepStream config semantics and docs)

Per stream, per frame:

1. **State estimation** — Kalman filter per target (`StateEstimator`: SIMPLE = constant velocity
   on location, random walk on size; process/measurement noise variances from the yml, separate
   measurement noise for detector boxes and tracker (DCF) boxes).
2. **Visual localization** — for every target, extract a feature image (`featureImgSizeLevel`,
   ColorNames and/or HOG channels, Hann window with `featureFocusOffsetFactor_y`) from a search
   region around the predicted box, correlate with the target's filter, take the response peak as
   the new position and the peak value as `tracker_confidence`.
3. **Data association** (`DataAssociator`) — score(target, detection) = weighted sum of
   *visual similarity* ("correlation response ratio": response at the detection's position over
   the peak), *IoU* and *size similarity*, each gated by its `minMatchingScore4*`, the total by
   `minMatchingScore4Overall`; same-class only when `checkClassMatch`; cascaded matching.
   Low-confidence ("tentative") detections only match by IoU (`minMatchingScore4TentativeIou`)
   and never create targets.
4. **Target management** — matched targets take the detector box; unmatched targets with
   `tracker_confidence >= minTrackerConfidence` continue on the DCF box (this is what bridges
   detector misses); otherwise **shadow tracking** (Kalman only, nothing output) for up to
   `maxShadowTrackingAge` frames. New targets are tentative until `probationAge`, are dropped early
   after `earlyTerminationAge` shadow frames, and are not created on top of an existing target
   (`minIouDiff4NewTarget`). Filters learn every frame with `filterLr` (EMA).
5. Optional in DS, **not** reproduced here: ReID-based and trajectory-projection re-association
   (`TrajectoryManagement`, `ReID`, TAO resnet50 embedding), NvDCF's learned 32768-entry ColorNames
   table, per-channel adaptive weights (`filterChannelWeightsLr`).

## avap implementation

`NvDcfConfig` uses DeepStream's key names verbatim; `NvDcfConfig.from_yaml()` reads
`config_tracker_NvDCF_*.yml` files as they ship (the `%YAML:1.0` header is tolerated). avap-only
knobs: `searchRegionPaddingScale` (search window = box × 2.5 by default), `useGray`,
`filterLambda`, `maxVisualOnlyAge` (cap on consecutive DCF-only frames), `maxVisualDisplacementRatio`
(a DCF jump larger than half a box is distrusted), `classAgnosticLabels` (majority-vote label).

**Correlation filter (`cpp/dcf.hip`, MOSSE/DCF formulation, multi-channel):**

```
features   F_c = hann · (f_c − mean(f_c))          c ∈ {gray, 11 colour names, 9 gradient bins}
learn      A_c ← (1−lr) A_c + lr · Ĝ ⊙ conj(F̂_c)      B ← (1−lr) B + lr · Σ_c |F̂_c|²
respond    R = IFFT( Σ_c A_c ⊙ Ẑ_c / (B + λ) ) / S²      peak → displacement, peak value → confidence
```

One `DcfEngine` per stream owns a pool of `maxTargetsPerStream` filter slots and batched hipFFT
plans (R2C for `N·C` feature planes, C2R for `N` responses). Feature extraction, normalization,
the response/learn kernels and the peak readback run in one `localize()` and one `update()` call
per frame, batched over all targets; the frame is any device CHW float RGB tensor plus an affine
from source to frame pixels, so the tracker reads **directly from the BatchCanvas slot** the
detector just consumed (`MuxedPipeline` builds the `VisualFrame`; no extra copy). Colour names are
a soft assignment to 11 prototype colours (stand-in for the learned table); HOG is per-pixel
unsigned gradient orientation, 9 bins. `NumpyDcfEngine` is the bit-for-bit reference used by
the tests (`test_hip_engine_matches_numpy_reference`) and the CPU fallback.

**Where it plugs in:** `TrackerBank.update(source_id, dets, frame)` passes the frame to trackers
with `uses_frames`; `examples/multi_stream_mux.py --nvdcf` runs it inside the streammux pipeline.
`ObjectMeta.tracked_by == "dcf"` marks boxes the visual tracker produced during detector misses;
their `confidence` is the tracker confidence, as in `NvDsObjectMeta.tracker_confidence`.

## Verification

* `tests/test_nvdcf.py` (7): translation is recovered to within one feature pixel and a foreign
  texture scores <60% of the true peak; HIP features and responses match the numpy reference
  (features to 2e-3, responses to 2e-2, identical peak); a 5-frame detector gap is bridged by DCF
  boxes that follow the object with the id preserved; without frames the same gap goes to shadow
  mode and the id survives; tentative targets die early / confirm after probation; the response
  is high on the target's own box and zero outside its search window; yml loading.
* `examples/multi_stream_mux.py --nvdcf` on the R9700 (4 streams, yolo26m fp16 batch 4): 3420
  frames, 96.7 frames/s end-to-end with the tracker reading canvas slots (137 fps without any
  tracker), longest track 1798 frames.

## Results on NDS footage

Same replay methodology as `docs/tracker_comparison_nds.md`, but the visual tracker also decodes
the source video (854x480 @ 25 fps) and uploads each frame. `nvdcf` = accuracy-like profile
(feature level 3, colour names + HOG, lr 0.077, shadow age 90 frames = 3.6 s like ocsort-long);
`nvdcf-noframes` = identical association/lifecycle with no pixels (isolates the visual term).

Sanity segment, camera 160850, first 3000 frames (2 min):

| strategy | tracks | ≥5 s | med len | mean len | frag/min | births/min | DCF-bridged boxes | ms/frame |
|---|---|---|---|---|---|---|---|---|
| ocsort-long | 330 | 92 | 47.0 | 116.6 | 104.0 | 165.0 | – | 2.45 |
| nvdcf-noframes | 230 | 103 | 84.5 | 186.9 | 37.0 | 115.0 | – | 0.81 |
| nvdcf-perf | 286 | 95 | 60.0 | 154.7 | 71.0 | 143.0 | 1677 | 4.55 |
| **nvdcf** | **226** | **123** | **138.5** | **247.8** | **33.0** | **113.0** | 13830 | 4.84 |

Six cameras × first 15,000 frames (10 min each, 90k frames total; log and per-camera JSON in
`results/nds/nvdcf_comparison.{log,json}`):

| strategy | tracks | ≥5 s | med len | mean len | frag/min | births/min | DCF-bridged boxes | ms/frame |
|---|---|---|---|---|---|---|---|---|
| ocsort-long (baseline) | 5570 | 1809 | 75.7 | 318.5 | 54.8 | 92.8 | – | 1.74 |
| bytetrack-long | 5734 | 1780 | 64.9 | 299.8 | 56.8 | 95.6 | – | 0.50 |
| nvdcf-noframes | 3794 | 1774 | 124.2 | 541.7 | 23.8 | 63.2 | – | 0.68 |
| **nvdcf (HIP DCF)** | **3478** | **1966** | **174.6** | **702.9** | **19.7** | **58.0** | 252,899 | 3.71 |

Reading:

* Against the previous best (ocsort-long), NvDCF cuts track count 38% and fragments/min 64%
  while the number of ≥5 s tracks *rises* 9% — the reduction is fragment elimination, not distinct
  objects being merged. Median track length goes from 3.0 s to 7.0 s.
* Isolating the appearance term (`nvdcf` vs `nvdcf-noframes`, identical association/lifecycle):
  −8% tracks, −17% fragments/min, +11% ≥5 s tracks, +41% median length. That is the 10–20%
  the correlation filter was expected to buy on top of NvDCF's lifecycle; the remaining gain over
  OC-SORT comes from the lifecycle itself (probation, shadow tracking, size gating, tentative
  detections). A quarter of a million boxes (2.8 per frame) were produced by the DCF during detector
  misses.
* Per camera the ordering is the same everywhere except 165331 (a quiet camera with long-parked
  vehicles), where both NvDCF variants report fewer ≥5 s tracks (51–54 vs 91): the long shadow
  memory keeps a parked object as one track where OC-SORT restarts it; inspect before deploying on
  parking-heavy views (`maxShadowTrackingAge`, `maxVisualOnlyAge` are the levers).
* Cost: 3.7 ms/frame on the R9700 including video decode upload and the Python association, vs
  1.7 ms for OC-SORT; the HIP part (extract + FFT + response + learn for all targets) is ~1 ms.

## Gaps vs NvDCF / follow-ups

1. Learned ColorNames table (w2c) instead of the 11-prototype soft assignment.
2. Per-channel adaptive weights (`filterChannelWeightsLr`) and scale adaptation.
3. Re-association: trajectory projection (`TrajectoryManagement`) and a ReID embedding (the LPR
   cascade's DINOv2 features are a candidate) — this is where DS's accuracy profile gets its last
   fragmentation reduction.
4. Async localization: run `localize()` for frame t+1 while the detector runs, as NvDCF does.
