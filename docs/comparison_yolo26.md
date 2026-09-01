# AMD pipeline vs NVIDIA DeepStream — YOLO26-m comparison

Frames compared: **89** | IoU match threshold: 0.5

## Detection agreement

| | AMD (R9700/MIGraphX) | DeepStream (3090/TensorRT) |
|---|---|---|
| total detections | 546 | 496 |
| per class | {'car': 408, 'truck': 138} | {'car': 303, 'truck': 192, 'person': 1} |

- **Matched (same object, IoU>=0.5): 436** — 79.9% of AMD, 87.9% of DS
- AMD-only detections: 110 | DS-only detections: 60
- Mean IoU of matched pairs: 0.943
- Class label agreement on matched pairs: 374/436 (85.8%)
- Class disagreements (AMD label, DS label): {('truck', 'car'): 8, ('car', 'truck'): 54}
- Mean |confidence delta| on matched pairs: 0.1227

## Tracking

| | AMD (greedy IoU tracker) | DeepStream (NvDsTracker IOU) |
|---|---|---|
| unique track ids | 63 | 11 |
| mean track length (frames) | 8.7 | 69.6 |
| tracks >= 10 frames | 16 | 11 |

## Per-frame object counts (first/mid/last)

| frame | AMD | DS |
|---|---|---|
| 0 | 6 | 7 |
| 1 | 5 | 2 |
| 2 | 6 | 3 |
| 43 | 10 | 6 |
| 44 | 7 | 5 |
| 45 | 5 | 4 |
| 86 | 6 | 5 |
| 87 | 6 | 6 |
| 88 | 5 | 5 |

## Runtime-parity experiment (identical input tensors)

Ten CPU-decoded reference tensors fed to all three runtime configs; raw
(1,300,6) outputs diffed row-wise (rows with score > 0.25):

| comparison | max \|Δscore\| | mean \|Δscore\| | max \|Δbox\| px | class flips |
|---|---|---|---|---|
| MIGraphX FP32 vs TRT noTF32 | 0.00001 | 0.00000 | 0.001 | 0 |
| MIGraphX FP32 vs TRT TF32 (default) | 0.00322 | 0.00077 | 0.090 | 0 |
| bridge preprocessing vs CPU reference (same runtime) | 0.21910 | 0.03040 | — | 15 |

**Conclusions**
1. The model runtimes are numerically equivalent: MIGraphX FP32 and
   TensorRT with TF32 disabled agree to 1e-5. The ONNX is byte-identical
   on both stacks.
2. TensorRT's Ampere TF32 default contributes at most 0.003 — negligible.
3. The pipeline-level 0.12 confidence delta is ~entirely preprocessing.
   The bridge's pixel delta vs reference (mean 0.0057) alone moves scores
   by 0.03 mean / 0.22 max and flips ambiguous car/truck classes.
4. Root cause: the video is fully untagged (color_space=unknown). The
   bridge defaults UNKNOWN -> BT.709; ffmpeg/swscale and DeepStream
   default untagged content to BT.601. Fix: default the bridge to BT.601
   for untagged sources (match ecosystem convention), keep honoring tags.
