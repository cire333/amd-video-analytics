"""GPU end-to-end for the advanced API: real VCN decode + real MIGraphX
inference through the full PGIE -> SGIE cascade -> LPR -> publisher, using
tiny synthetic ONNX models with known outputs (the proprietary plate/OCR
weights are not in this repo; the cascade CONTRACTS are what's under test).

Run on the GPU box:
    sg render -c "PYTHONPATH=/opt/rocm/lib AVAP_GPU_E2E=1 \
        .venv/bin/python -m pytest tests/test_advanced_gpu_e2e.py -v"
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AVAP_GPU_E2E") != "1",
    reason="GPU e2e: set AVAP_GPU_E2E=1 on a ROCm box (run via sg render)")

VIDEO = "/home/eric-valasek/Downloads/load-to-server/1933_A22.mp4"
IN = 64  # tiny model input size


def export_models(d: Path) -> dict[str, str]:
    import torch

    class ConstDet(torch.nn.Module):
        def __init__(self, box):
            super().__init__()
            out = torch.zeros(1, 300, 6)
            out[0, 0] = torch.tensor(box)
            self.register_buffer("out", out)

        def forward(self, x):
            return self.out + 0.0 * x.mean()

    class Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.fc = torch.nn.Linear(3 * IN * IN, 768)

        def forward(self, x):
            return self.fc(x.flatten(1))

    class Logits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("out", torch.tensor([[0.0, 6.0, 0.0, 0.0]]))

        def forward(self, x):
            return self.out + 0.0 * x.mean()

    models = {
        # 'vehicle' (class 2) covering the frame center at conf 0.9
        "primary": ConstDet([16, 16, 48, 48, 0.9, 2]),
        # plate box inside any vehicle crop
        "plate": ConstDet([8, 20, 56, 40, 0.8, 0]),
        "ocr": Logits(),
        "embed": Embed(),
    }
    paths = {}
    x = torch.zeros(1, 3, IN, IN)
    for name, m in models.items():
        p = str(d / f"{name}.onnx")
        torch.onnx.export(m.eval(), (x,), p, input_names=["images"],
                          opset_version=17, dynamo=False)
        paths[name] = p
    return paths


def test_full_cascade_on_gpu(tmp_path):
    from avap.advanced import (AdvancedPipeline, ClassifierStage,
                               DataPublisher, DetectionStage, EmbeddingStage,
                               FileTransport, InferConfig, LPRConfig,
                               LPRStage, PrimaryStage)

    paths = export_models(tmp_path)
    labels_coco = ["person", "bicycle", "car"]
    pipe = AdvancedPipeline(
        primary=PrimaryStage(InferConfig(model=paths["primary"],
                                         component_id=1, labels=labels_coco,
                                         conf_threshold=0.3)),
        stages=[
            DetectionStage(InferConfig(model=paths["plate"], component_id=2,
                                       operate_on_component=1,
                                       operate_on_class_ids=frozenset({2}),
                                       labels=["plate"], conf_threshold=0.3)),
            ClassifierStage(InferConfig(model=paths["ocr"], component_id=3,
                                        operate_on_component=2,
                                        labels=["AAA111", "GMX777", "B", "C"],
                                        min_object_width=4,
                                        min_object_height=4)),
            EmbeddingStage(InferConfig(model=paths["embed"], component_id=4,
                                       operate_on_component=1,
                                       operate_on_class_ids=frozenset({2}),
                                       normalize_output=True)),
        ],
        tracker="sort",
        lpr=LPRStage(LPRConfig(min_plate_reads=2, vehicle_class_ids={2},
                               plate_confidence_threshold=0.5)),
        publisher=DataPublisher(FileTransport(str(tmp_path / "outbox"))),
    )
    seen = {"frames": 0}
    pipe.add_probe("lpr", lambda f, r: seen.__setitem__("frames", f.frame + 1))
    idx = pipe.add_source(VIDEO, camera_id="e2e-cam",
                          image_resolution=(1280, 704), fps=25,
                          batch_size_seconds=1,
                          metadata={"id": "e2e-cam"})
    pipe.run()   # 89 frames, blocks to EOS

    assert seen["frames"] == 89

    # LPR: plate voted, vehicle re-identified via real embeddings
    results = [pipe.lpr.get_result(t, "e2e-cam") for t in range(1, 10)]
    results = [r for r in results if r is not None]
    assert results, "no LPR results produced"
    assert any(r.plate_text == "GMX777" for r in results)
    assert any(r.vehicle_id is not None for r in results)
    assert any(r.embedding is not None and r.embedding.shape == (768,)
               for r in results)

    # published records: schema + bbox rescale to declared resolution
    batches = sorted((tmp_path / "outbox" / "source_e2e-cam").glob("*.json"))
    assert batches, "publisher wrote no batches"
    records = [r for b in batches for r in json.loads(b.read_text())]
    assert len(records) == 89
    objs = [r["Data"][0] for r in records if r["Data"]]
    assert objs, "no objects published"
    o = objs[-1]
    assert o["class"] == "light automobile"
    assert 0 <= o["x1"] < o["x2"] <= 1280 and 0 <= o["y1"] < o["y2"] <= 704
    assert o.get("plate_text") == "GMX777"
    assert o.get("vehicle_id", "").startswith("v_")
    fps_vals = [r["FPS"] for r in records[-5:]]
    assert all(isinstance(v, float) for v in fps_vals)
