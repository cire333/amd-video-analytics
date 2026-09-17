"""ModelChain GPU tests: device-resident daisy chaining.

    sg render -c "PYTHONPATH=/opt/rocm/lib AVAP_GPU_E2E=1 \
        .venv/bin/python -m pytest tests/test_model_chain.py -v"
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AVAP_GPU_E2E") != "1",
    reason="GPU test: set AVAP_GPU_E2E=1 on a ROCm box (run via sg render)")

SIZE = 64


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> dict[str, str]:
    import torch
    d = tmp_path_factory.mktemp("chain_models")

    class Conv(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.c = torch.nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, x):
            return torch.relu(self.c(x))

    class Head(torch.nn.Module):  # frame -> two outputs (vector + scalar-ish)
        def __init__(self):
            super().__init__()
            torch.manual_seed(9)
            self.fc = torch.nn.Linear(3 * SIZE * SIZE, 16)

        def forward(self, x):
            v = self.fc(x.flatten(1))
            return v, v.sum(dim=1, keepdim=True)

    class Small(torch.nn.Module):  # incompatible input size
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, x):
            return self.c(x)

    paths = {}
    x = torch.zeros(1, 3, SIZE, SIZE)
    for name, m, inp in [("a", Conv(1), x), ("b", Conv(2), x),
                         ("head", Head(), x),
                         ("small", Small(), torch.zeros(1, 3, 32, 32))]:
        p = str(d / f"{name}.onnx")
        torch.onnx.export(m.eval(), (inp,), p, input_names=["images"],
                          opset_version=17, dynamo=False)
        paths[name] = p
    return paths


def test_chain_matches_sequential_host_execution(models):
    from avap.chain import ModelChain
    from avap.model_zoo import MigraphxModel

    chain = ModelChain([models["a"], models["b"]], quant="fp32")
    x = np.random.default_rng(0).random((1, 3, SIZE, SIZE), dtype=np.float32)
    chained = chain(x)

    ma = MigraphxModel(models["a"], "fp32")
    mb = MigraphxModel(models["b"], "fp32")
    sequential = mb(ma(x))
    assert np.abs(chained - sequential).max() < 1e-4


def test_shape_mismatch_rejected_at_build(models):
    from avap.chain import ModelChain
    with pytest.raises(ValueError, match="shape mismatch at hop 0"):
        ModelChain([models["a"], models["small"]], quant="fp32")


def test_multi_output_final_model(models):
    from avap.chain import ModelChain
    chain = ModelChain([models["a"], models["head"]], quant="fp32")
    outs = chain(np.ones((1, 3, SIZE, SIZE), dtype=np.float32))
    assert isinstance(outs, list) and len(outs) == 2
    assert outs[0].shape == (1, 16) and outs[1].shape == (1, 1)
    # multi-output models are only allowed at the END of the chain
    with pytest.raises(ValueError, match="interior chain models"):
        ModelChain([models["head"], models["a"]], quant="fp32")


def test_single_model_chain_and_ducktype(models):
    from avap.chain import ModelChain
    chain = ModelChain([models["a"]], quant="fp16")
    assert len(chain) == 1 and chain.ready and chain.quant == "fp16"
    assert chain.input_shape == (1, 3, SIZE, SIZE)
    out = chain(np.zeros((1, 3, SIZE, SIZE), dtype=np.float32))
    assert out.shape == (1, 3, SIZE, SIZE)


def test_repeated_calls_reuse_buffers(models):
    from avap.chain import ModelChain
    chain = ModelChain([models["a"], models["b"]], quant="fp32")
    x = np.random.default_rng(1).random((1, 3, SIZE, SIZE), dtype=np.float32)
    first = chain(x)
    for _ in range(20):
        again = chain(x)
    assert np.array_equal(first, again)  # deterministic across buffer reuse
