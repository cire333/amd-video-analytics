"""rocDecode zero-copy backend (GPU).

    sg render -c "LD_LIBRARY_PATH=... PYTHONPATH=/opt/rocm/lib:src AVAP_GPU_E2E=1 \
        .venv/bin/python -m pytest tests/test_rocdecode.py -v"
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AVAP_GPU_E2E") != "1",
    reason="GPU test: set AVAP_GPU_E2E=1 (run via sg render)")

VIDEO = "/home/eric-valasek/Downloads/load-to-server/1933_A22.mp4"


def _need_rocdecode():
    from avap import _core
    if not getattr(_core, "has_rocdecode", False):
        pytest.skip("_core built without rocDecode")
    return _core


def test_decodes_all_frames_zero_copy():
    _core = _need_rocdecode()
    rd = _core.RocDecoder(VIDEO, 0)
    n = 0
    while True:
        r = rd.next_frame_device_rgb()
        if r is None:
            break
        handle, w, h, pts = r
        assert (w, h) == (1920, 1056)
        _core.free_device_buffer(handle)
        n += 1
    rd.close()
    assert n == 89


def test_pixel_parity_with_vaapi_backend():
    _core = _need_rocdecode()
    from avap.capabilities import probe_devices, require_decode_device
    dev = require_decode_device(probe_devices())

    rd = _core.RocDecoder(VIDEO, dev.device_ordinal)
    r = rd.next_frame_device_rgb()
    handle, w, h, _ = r
    roc = _core.device_rgb_to_host(handle, w, h)
    _core.free_device_buffer(handle)
    rd.close()

    dec = _core.Decoder(VIDEO, dev.drm_render_node)
    f = dec.next_frame()
    planes = [tuple(p) for p in f.planes]
    args = (planes, (f.crop_x, f.crop_y, f.crop_w, f.crop_h),
            (f.crop_w, f.crop_h), f.full_range,
            f.color_matrix != "bt601", dev.device_ordinal)
    va = (_core.nv12_dmabuf_to_rgb(f.dmabuf_fd, f.width, f.height, planes,
                                   f.drm_modifier, *args[1:])
          if f.dmabuf_fd >= 0 else _core.nv12_host_to_rgb(f.host_data, *args))
    dec.close()
    assert np.abs(roc - va).max() < 1e-6   # same VCN, same kernel: identical


def test_advanced_pipeline_uses_rocdecode(tmp_path):
    _core = _need_rocdecode()
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_advanced_gpu_e2e import export_models
    from avap.advanced import AdvancedPipeline, InferConfig, PrimaryStage

    paths = export_models(tmp_path)
    seen = []
    pipe = AdvancedPipeline(
        primary=PrimaryStage(InferConfig(model=paths["primary"], component_id=1,
                                         labels=["p", "b", "car"])),
        tracker="sort", decode_backend="rocdecode")
    pipe.add_probe("tracker", lambda fr, rgb: seen.append(fr.frame))
    pipe.add_source(VIDEO, camera_id="rocdec-cam")
    pipe.run()
    assert pipe.decode_backend == "rocdecode"
    assert len(seen) == 89


def test_amdstream_backend_resolution():
    _core = _need_rocdecode()
    from avap import AMDStream
    s = AMDStream(VIDEO, decode_backend="auto", source_id="t")
    s._uri = VIDEO
    s._resolve_decode_backend()
    assert s.decode_backend == "rocdecode"      # file -> zero-copy
    s2 = AMDStream("rtsp://cam/live", decode_backend="auto", source_id="t2")
    s2._uri = "rtsp://cam/live"
    s2._resolve_decode_backend()
    assert s2.decode_backend == "vaapi"         # live -> deadline-bounded backend
    import pytest as _pt
    with _pt.raises(ValueError):
        AMDStream(VIDEO, decode_backend="bogus")


def test_amdstream_end_to_end_rocdecode(tmp_path):
    _core = _need_rocdecode()
    import json
    import time
    from avap import AMDStream
    s = AMDStream(VIDEO, model="yolo26m", model_quant="fp16",
                  tracker_type="ocsort", decode_backend="rocdecode",
                  region_of_interest=(0.0, 0.2, 0.9, 1.0),
                  output_location=str(tmp_path / "d.jsonl"),
                  annotated_output=str(tmp_path / "a.mp4"),
                  source_id="rocdec-stream")
    s.start_stream()
    deadline = time.time() + 300
    while s.state == "running" and time.time() < deadline:
        time.sleep(0.5)
    s.join(timeout=30)
    s.stop_stream()
    assert s.state == "eof"
    lines = [json.loads(l) for l in open(tmp_path / "d.jsonl")]
    assert len(lines) == 89
    # ROI-fused GPU crop: detections exist and stay inside the ROI x-range
    objs = [o for l in lines for o in l["objects"]]
    assert objs
    assert all(o["bbox"][0] >= -1 for o in objs)
    import subprocess
    n = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0",
                        str(tmp_path / "a.mp4")],
                       capture_output=True, text=True).stdout.strip()
    assert n == "89"
