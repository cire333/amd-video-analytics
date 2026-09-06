# Advanced API (`avap.advanced`)

The integrated pipeline layer, reverse-engineered from the DeepStream
ingestion system (`ingestion_processing`). Where the light API
(`avap.AMDStream`) is detector + tracker + sink, the advanced API
reproduces the composed-pipeline contracts: secondary inference on object
crops, hierarchical metadata, probes, LPR, and batched publishing.

## Concept mapping

| DeepStream / ingestion_processing | avap.advanced |
|---|---|
| nvinfer PGIE (`gie-unique-id=1`) | `PrimaryStage(InferConfig(component_id=1, ...))` |
| SGIE detector (`process-mode=2`, `operate-on-gie-id`, `operate-on-class-ids`) | `DetectionStage(InferConfig(operate_on_component=, operate_on_class_ids=))` — creates child objects with `parent` links |
| SGIE classifier (`network-type=1`, `parse-classifier-func-name`) | `ClassifierStage(..., parse=my_parser)` — attaches `Classification`; parser pluggable (e.g. CTC decode for plate OCR) |
| SGIE raw tensor (`network-type=100`, `output-tensor-meta=1`) | `EmbeddingStage` — attaches the output vector to `obj.tensors[component_id]` |
| NvDsBatch/Frame/ObjectMeta + parent + `unique_component_id` + classifier/tensor meta | `AdvFrameMeta` / `AdvObjectMeta` (`parent`, `children`, `component_id`, `classifications`, `tensors`) |
| pad probes (`attach_data_extraction_probe`, LPR probe) | `pipe.add_probe(point, fn)` — points: `primary`, `tracker`, each stage name, `lpr`; broken probes are isolated |
| nvtracker (NvDCF) | per-source tracker from the avap registry (`sort`/`bytetrack`/`ocsort`/BYO) |
| `LPRStage`, `PlateVoter`, `VehicleReID` (FAISS) | ported functionally (`avap.advanced.lpr`): same voting/thresholds/30-embedding averaging; re-ID is exact inner-product top-1 in numpy (no FAISS dep), JSON persistence. **Improvement**: state keyed `(source_id, track_id)` — upstream collides track ids across cameras |
| `DataRecord`/`ObjectRecord` JSON (`class`, `id`, stringified `confidence_score`, conditional LPR keys) | `avap.advanced.records` — byte-compatible `to_dict()` (verified by test) |
| `CLASS_MAPPING` (label -> customer taxonomy, doubles as filter) | `ClassMapper(mapping, drop_unmapped=)` |
| streammux canvas -> `image_resolution` bbox rescale | records rescaled from decode resolution to the source's declared `image_resolution` |
| source index + free list + `StreamResource` context | `add_source() -> int` (recycled indices), `SourceContext` |
| per-source EOS isolation (EOS dropped at source pad) | per-source worker threads; one source's EOS/failure never affects others |
| `DataPublisher` (bounded queue, worker thread, per-source size/time batches, lossy `put_nowait`) | `avap.advanced.publisher.DataPublisher` — same semantics + stats; transports are callables (`FileTransport` matches the upstream file layout; wire any avap sink or custom code) |
| `GETFPS`/`PERF_DATA` (cumulative avg, lazily created) | `avap.advanced.metrics.PerfData` |
| ImageBasedStrategy (appsrc images) | `pipe.process_image(index, rgb, pts_us)` — the same stage graph without a decoder |

## LPR usage

```python
from avap.advanced import (AdvancedPipeline, PrimaryStage, DetectionStage,
                           ClassifierStage, EmbeddingStage, InferConfig,
                           LPRStage, LPRConfig, DataPublisher, FileTransport)

pipe = AdvancedPipeline(
    primary=PrimaryStage(InferConfig(model="yolo26m", component_id=1,
                                     labels=COCO)),
    stages=[
        DetectionStage(InferConfig(model="plate_detect.onnx", component_id=2,
                                   operate_on_component=1,
                                   operate_on_class_ids=frozenset({2,3,5,7}),
                                   labels=["plate"])),
        ClassifierStage(InferConfig(model="plate_ocr.onnx", component_id=3,
                                    operate_on_component=2),
                        parse=my_ctc_decoder),      # plug the OCR decoder here
        EmbeddingStage(InferConfig(model="dinov2_vitb14.onnx", component_id=4,
                                   operate_on_component=1,
                                   operate_on_class_ids=frozenset({2,3,5,7}),
                                   normalize_output=True)),
    ],
    tracker="ocsort",
    lpr=LPRStage(LPRConfig(similarity_threshold=0.88, min_plate_reads=3,
                           plate_confidence_threshold=0.75,
                           embeddings_dir="./embeddings")),
    publisher=DataPublisher(FileTransport("./outbox")),
)
idx = pipe.add_source("rtsp://cam/live", camera_id="IBG-C1011",
                      image_resolution=(1920, 1080), fps=30,
                      metadata=camera_json)
pipe.run()
```

Per-tracked-vehicle output (merged into published ObjectRecords, same as
upstream): `vehicle_id` (stable across track breaks via embedding re-ID),
`plate_text` (majority vote over confident reads), `plate_confidence`,
`plate_read_count`, `embedding[768]`. The embedding registry persists to
`embeddings_dir/embeddings.json` and reloads across runs.

## Known gaps vs the DeepStream system (deliberate, roadmap)

- Display/RTMP restream strategy and mkv recording branches — not ported
  (annotation exists in scripts; a display sink is a separate feature).
- Distributed executor (DynamoDB ledger, locks, watchdog top-up, ECS
  identity) — orchestration layer, out of library scope for now; the
  building blocks (dynamic add/remove, per-source isolation, stats) exist.
- The plate OCR CTC decoder is a pluggable `parse=` hook rather than
  built-in (upstream keeps it in a custom C++ lib; port it as a ~20-line
  Python function against your OCR model's vocab).
- Stage inference is serialized on one GPU context (a lock), not batched
  across sources like nvstreammux; throughput work is tracked separately.

Tested: 24 unit tests (metadata hierarchy, voting, re-ID persistence &
cross-track re-identification, record wire-format byte-compat, stage
targeting, publisher batching/lossiness, probes, index recycling, bbox
rescale) + a GPU end-to-end running the full cascade with real MIGraphX
inference over real decoded video (tests/test_advanced_gpu_e2e.py).
