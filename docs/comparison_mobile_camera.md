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
