"""Benchmark: daisy-chained models, host-roundtrip vs device-resident.

    sg render -c "PYTHONPATH=/opt/rocm/lib:src .venv/bin/python scripts/bench_chain.py [size] [n_models]"
"""
import sys
import time
from pathlib import Path

import numpy as np


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 640
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    reps = 50

    import torch

    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.c = torch.nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, x):
            return torch.relu(self.c(x))

    d = Path("/tmp/claude-1000") if Path("/tmp/claude-1000").exists() else Path("/tmp")
    onnx = str(d / f"chain_hop_{size}.onnx")
    torch.onnx.export(Conv().eval(), (torch.zeros(1, 3, size, size),), onnx,
                      input_names=["images"], opset_version=17, dynamo=False)

    from avap.chain import ModelChain
    from avap.model_zoo import MigraphxModel

    hosts = [MigraphxModel(onnx, "fp16") for _ in range(n)]
    chain = ModelChain([onnx] * n, quant="fp16")
    x = np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32)

    for _ in range(5):
        cur = x
        for m in hosts:
            cur = m(cur)
        chain(x)

    t0 = time.perf_counter()
    for _ in range(reps):
        cur = x
        for m in hosts:
            cur = m(cur)
    host_ms = (time.perf_counter() - t0) / reps * 1000

    t0 = time.perf_counter()
    for _ in range(reps):
        chain(x)
    dev_ms = (time.perf_counter() - t0) / reps * 1000

    mb = 3 * size * size * 4 / 1e6
    print(f"chain of {n} models @ {size}x{size} fp16 ({mb:.1f} MB/frame):")
    print(f"  host-roundtrip:  {host_ms:6.2f} ms/frame  ({host_ms/n:.2f} ms/hop)")
    print(f"  ModelChain:      {dev_ms:6.2f} ms/frame  ({dev_ms/n:.2f} ms/hop)")
    print(f"  saved: {host_ms - dev_ms:.2f} ms/frame ({(1 - dev_ms/host_ms):.0%})")


if __name__ == "__main__":
    main()
