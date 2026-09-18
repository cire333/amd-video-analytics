"""MIGraphX compiled-program cache (GPU).

    sg render -c "PYTHONPATH=/opt/rocm/lib:src AVAP_GPU_E2E=1 \
        .venv/bin/python -m pytest tests/test_compile_cache.py -v"
"""
import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AVAP_GPU_E2E") != "1",
    reason="GPU test: set AVAP_GPU_E2E=1 (run via sg render)")


@pytest.fixture()
def small_onnx(tmp_path):
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.c = torch.nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            return torch.relu(self.c(x))

    p = str(tmp_path / "net.onnx")
    torch.onnx.export(Net().eval(), (torch.zeros(1, 3, 128, 128),), p,
                      input_names=["images"], opset_version=17, dynamo=False)
    return p


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    import avap.model_zoo as mz
    cache = tmp_path / "mxr"
    monkeypatch.setattr(mz, "_MXR_CACHE", cache)
    return cache


def test_cache_roundtrip_identical_outputs(small_onnx, isolated_cache):
    from avap.model_zoo import MigraphxModel
    x = np.random.default_rng(0).random((1, 3, 128, 128), dtype=np.float32)

    cold = MigraphxModel(small_onnx, "fp16")
    assert not cold.loaded_from_cache
    assert len(list(isolated_cache.glob("*.mxr"))) == 1

    warm = MigraphxModel(small_onnx, "fp16")
    assert warm.loaded_from_cache
    assert np.array_equal(cold(x), warm(x))


def test_warm_load_is_fast(small_onnx, isolated_cache):
    from avap.model_zoo import MigraphxModel
    t0 = time.perf_counter()
    MigraphxModel(small_onnx, "fp16")
    cold_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    MigraphxModel(small_onnx, "fp16")
    warm_s = time.perf_counter() - t0
    assert warm_s < cold_s, (cold_s, warm_s)
    assert warm_s < 2.0, f"warm load took {warm_s:.2f}s"


def test_cache_key_separates_variants(small_onnx, isolated_cache):
    from avap.model_zoo import MigraphxModel, compile_or_load
    MigraphxModel(small_onnx, "fp16")
    MigraphxModel(small_onnx, "fp32")
    compile_or_load(small_onnx, "fp16", offload_copy=False)
    assert len(list(isolated_cache.glob("*.mxr"))) == 3


def test_model_change_invalidates(small_onnx, isolated_cache):
    from avap.model_zoo import MigraphxModel
    MigraphxModel(small_onnx, "fp16")
    # touch the file with new mtime + size -> different key
    time.sleep(0.02)
    with open(small_onnx, "ab") as f:
        f.write(b"\x00")  # (invalid onnx tail is fine; parse reads protobuf)
    os.utime(small_onnx, (time.time() + 5, time.time() + 5))
    from avap.model_zoo import _cache_key
    keys = {_cache_key(small_onnx, "fp16", True)}
    assert not (isolated_cache / f"{keys.pop()}.mxr").exists()


def test_corrupt_cache_falls_back(small_onnx, isolated_cache):
    from avap.model_zoo import MigraphxModel, _cache_key
    MigraphxModel(small_onnx, "fp16")
    mxr = isolated_cache / f"{_cache_key(small_onnx, 'fp16', True)}.mxr"
    mxr.write_bytes(b"garbage")
    m = MigraphxModel(small_onnx, "fp16")   # must not raise
    assert not m.loaded_from_cache


def test_disable_env(small_onnx, isolated_cache, monkeypatch):
    monkeypatch.setenv("AVAP_NO_MXR_CACHE", "1")
    from avap.model_zoo import MigraphxModel
    MigraphxModel(small_onnx, "fp16")
    assert not list(isolated_cache.glob("*.mxr"))


def test_device_model_and_chain_cached(small_onnx, isolated_cache):
    from avap.advanced.device import DeviceModel
    from avap.chain import ModelChain
    d1 = DeviceModel(small_onnx, "fp16")
    assert not d1.loaded_from_cache
    d2 = DeviceModel(small_onnx, "fp16")
    assert d2.loaded_from_cache
    # chain reuses the same offload_copy=False cache entry
    n_before = len(list(isolated_cache.glob("*.mxr")))
    ModelChain([small_onnx], quant="fp16")
    assert len(list(isolated_cache.glob("*.mxr"))) == n_before
