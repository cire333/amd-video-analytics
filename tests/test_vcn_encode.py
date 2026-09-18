"""VCN encoder GPU tests: hardware encode -> container -> decode round-trip.

    sg render -c "PYTHONPATH=/opt/rocm/lib:src AVAP_GPU_E2E=1 \
        .venv/bin/python -m pytest tests/test_vcn_encode.py -v"
"""
import json
import os
import subprocess

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AVAP_GPU_E2E") != "1",
    reason="GPU test: set AVAP_GPU_E2E=1 on a ROCm box (run via sg render)")

W, H, FPS, N = 640, 360, 25.0, 50


def make_frame(i: int) -> np.ndarray:
    """Moving gradient + block: content that exercises motion estimation."""
    y, x = np.mgrid[0:H, 0:W]
    r = ((x + 3 * i) % 256).astype(np.uint8)
    g = ((y + 2 * i) % 256).astype(np.uint8)
    b = np.full((H, W), 64, np.uint8)
    img = np.stack([r, g, b], axis=-1)
    bx = (10 + 5 * i) % (W - 80)
    img[100:180, bx:bx + 80] = (255, 255, 255)
    return img


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,width,height,nb_read_frames",
         "-of", "json", path], capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]


def test_h264_encode_and_decode_roundtrip(tmp_path):
    from avap import VideoEncoder
    from avap import _core
    from avap.capabilities import probe_devices, require_decode_device

    path = str(tmp_path / "out.mp4")
    with VideoEncoder(path, W, H, fps=FPS, bitrate=4_000_000) as enc:
        for i in range(N):
            enc.write(make_frame(i))
    assert enc.frames_written == N

    st = probe(path)
    assert st["codec_name"] == "h264"
    assert (int(st["width"]), int(st["height"])) == (W, H)
    assert int(st["nb_read_frames"]) == N

    # decode back with OUR decoder and compare a mid frame to the source
    dev = require_decode_device(probe_devices())
    dec = _core.Decoder(path, dev.drm_render_node)
    for i in range(N // 2 + 1):
        f = dec.next_frame()
    planes = [tuple(p) for p in f.planes]
    args = (planes, (f.crop_x, f.crop_y, f.crop_w, f.crop_h), (W, H),
            f.full_range, f.color_matrix != "bt601", dev.device_ordinal)
    chw = (_core.nv12_dmabuf_to_rgb(f.dmabuf_fd, f.width, f.height, planes,
                                    f.drm_modifier, *args[1:])
           if f.dmabuf_fd >= 0 else _core.nv12_host_to_rgb(f.host_data, *args))
    dec.close()
    got = (np.clip(chw, 0, 1) * 255).transpose(1, 2, 0)
    ref = make_frame(N // 2).astype(np.float64)
    mad = float(np.abs(got - ref).mean())
    assert mad < 12.0, f"round-trip mean abs diff {mad:.1f} (lossy budget 12)"


def test_hevc_codec(tmp_path):
    from avap import VideoEncoder
    path = str(tmp_path / "out_hevc.mkv")
    with VideoEncoder(path, W, H, fps=FPS, codec="hevc") as enc:
        for i in range(20):
            enc.write(make_frame(i))
    assert probe(path)["codec_name"] == "hevc"


def test_write_device_matches_host_path(tmp_path):
    """Device-side RGB->NV12 kernel vs the CPU conversion: same video content
    within colorimetry tolerance (device honors BT.709; cv2 path is BT.601)."""
    from avap import VideoEncoder
    from avap.advanced.device import DeviceFrame

    p_host = str(tmp_path / "host.mp4")
    p_dev = str(tmp_path / "dev.mp4")
    frames = [make_frame(i) for i in range(20)]
    with VideoEncoder(p_host, W, H, fps=FPS) as enc:
        for fr in frames:
            enc.write(fr)
    with VideoEncoder(p_dev, W, H, fps=FPS, bt709=False) as enc:  # match cv2's 601
        for fr in frames:
            with DeviceFrame.from_host(fr, enc.device_ordinal) as dev:
                enc.write_device(dev)
    assert int(probe(p_dev)["nb_read_frames"]) == 20

    import cv2
    def mid_frame(path):
        cap = cv2.VideoCapture(path)
        for _ in range(10):
            ok, img = cap.read()
        cap.release()
        return img.astype(np.float64)
    mad = float(np.abs(mid_frame(p_dev) - mid_frame(p_host)).mean())
    assert mad < 6.0, f"device vs host encode diff {mad:.1f}"


def test_dimension_validation(tmp_path):
    from avap import VideoEncoder
    with pytest.raises(ValueError, match="even"):
        VideoEncoder(str(tmp_path / "x.mp4"), 641, 360)
    enc = VideoEncoder(str(tmp_path / "y.mp4"), W, H)
    with pytest.raises(ValueError, match="encoder is"):
        enc.write(np.zeros((H + 2, W, 3), np.uint8))
    enc.close()
    enc.close()  # idempotent
