"""Benchmark: SGIE crop-based cascade, host crops vs device-resident crops.

Simulates a realistic frame: K tracked vehicles, each passing through 3
secondary models (plate detect / OCR / embedding at 224 input).

    sg render -c "PYTHONPATH=/opt/rocm/lib:src .venv/bin/python scripts/bench_sgie.py [K] [frame_w] [frame_h]"
"""
import sys
import time
from pathlib import Path

import numpy as np


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    fw = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    fh = int(sys.argv[3]) if len(sys.argv) > 3 else 720
    reps = 30

    import torch

    class Conv(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.c = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
            self.h = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)

        def forward(self, x):
            return self.h(torch.relu(self.c(x)))

    d = Path("/tmp/claude-1000") if Path("/tmp/claude-1000").exists() else Path("/tmp")
    paths = []
    for i in range(3):
        p = str(d / f"sgie_bench_{i}.onnx")
        torch.onnx.export(Conv(i).eval(), (torch.zeros(1, 3, 224, 224),), p,
                          input_names=["images"], opset_version=17, dynamo=False)
        paths.append(p)

    import cv2
    from avap.advanced.device import DeviceFrame, DeviceModel
    from avap.model_zoo import MigraphxModel

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (fh, fw, 3), dtype=np.uint8)
    boxes = [(int(x), int(y), int(x) + 160, int(y) + 120)
             for x, y in zip(rng.integers(0, fw - 160, k),
                             rng.integers(0, fh - 120, k))]

    host_models = [MigraphxModel(p, "fp16") for p in paths]
    dev_models = [DeviceModel(p, "fp16") for p in paths]

    def host_pass():
        for x1, y1, x2, y2 in boxes:
            crop = frame[y1:y2, x1:x2]
            chw = np.ascontiguousarray(
                cv2.resize(crop, (224, 224)).astype(np.float32)
                .transpose(2, 0, 1)[None] / 255.0)
            for m in host_models:
                m(chw)

    def device_pass(dev):
        for x1, y1, x2, y2 in boxes:
            rect = (x1, y1, x2 - x1, y2 - y1)
            for m in dev_models:
                m.run_crop(dev, rect)

    host_pass()
    with DeviceFrame.from_host(frame, 0) as dev:
        device_pass(dev)

        t0 = time.perf_counter()
        for _ in range(reps):
            host_pass()
        host_ms = (time.perf_counter() - t0) / reps * 1000

        t0 = time.perf_counter()
        for _ in range(reps):
            device_pass(dev)
        dev_ms = (time.perf_counter() - t0) / reps * 1000

    n_inf = k * len(paths)
    print(f"SGIE cascade: {k} objects x {len(paths)} stages "
          f"({n_inf} crop-inferences/frame), frame {fw}x{fh}, input 224, fp16")
    print(f"  host crops (numpy+cv2+H2D):   {host_ms:6.2f} ms/frame "
          f"({host_ms/n_inf:.2f} ms/inference)")
    print(f"  device crops (HIP kernel):    {dev_ms:6.2f} ms/frame "
          f"({dev_ms/n_inf:.2f} ms/inference)")
    print(f"  saved: {host_ms - dev_ms:.2f} ms/frame ({1 - dev_ms/host_ms:.0%})")


if __name__ == "__main__":
    main()
