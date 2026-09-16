"""BatchCanvas on real HIP hardware (skipped without the extension / a GPU).

Uses synthetic NV12 host frames (no decoder), so it checks geometry and the
colour path independently of VAAPI: letterbox placement must match the
nvstreammux measurements (docs §2) and the scaled content must equal the
existing single-frame bridge kernel."""
import numpy as np
import pytest

from avap.frame import ColorMatrix, ColorRange, CropRect, PlaneLayout, RawFrame

pytest.importorskip("avap._core")
from avap import _core  # noqa: E402

if _core.hip_device_count() == 0:
    pytest.skip("no HIP device", allow_module_level=True)

from avap.canvas import BatchCanvas, letterbox_geometry  # noqa: E402
from avap.streammux import MuxBatch, MuxFrameMeta  # noqa: E402


def synth_nv12(w, h, pitch=None):
    """Y = x ramp, U = y ramp, V = 128 (+ white 16px square top-left)."""
    pitch = pitch or ((w + 63) // 64 * 64)
    y = np.zeros((h, pitch), np.uint8)
    y[:, :w] = np.linspace(16, 235, w).astype(np.uint8)[None, :]
    y[:16, :16] = 235
    uv = np.zeros((h // 2, pitch), np.uint8)
    uv[:, 0:w:2] = np.linspace(16, 240, h // 2).astype(np.uint8)[:, None]   # U
    uv[:, 1:w:2] = 128                                                       # V
    data = y.tobytes() + uv.tobytes()
    planes = (PlaneLayout(0, pitch), PlaneLayout(pitch * h, pitch))
    return data, planes


def host_frame(source_id, w, h, pts=0):
    data, planes = synth_nv12(w, h)
    return RawFrame(source_id=source_id, pts=pts, recv_us=0, width=w, height=h,
                    crop=CropRect(0, 0, w, h), dmabuf_fd=-1, planes=planes, drm_modifier=0,
                    color_range=ColorRange.LIMITED, color_matrix=ColorMatrix.BT709,
                    device_ordinal=0, host_data=data)


def test_letterbox_geometry_matches_nvstreammux_measurements():
    assert letterbox_geometry(1280, 720, 640, 640, True) == (0, 140, 640, 360)
    assert letterbox_geometry(1920, 1080, 640, 640, True) == (0, 140, 640, 360)
    assert letterbox_geometry(640, 480, 1920, 1080, True) == (240, 0, 1440, 1080)
    assert letterbox_geometry(1000, 700, 1920, 1056, True) == (206, 0, 1508, 1056)
    assert letterbox_geometry(1280, 720, 1920, 1080, False) == (0, 0, 1920, 1080)


def test_canvas_letterbox_content_and_padding():
    W = H = 320
    canvas = BatchCanvas(batch_size=2, width=W, height=H, enable_padding=True)
    try:
        f0 = host_frame("a", 640, 360)      # 16:9 -> 320x180 at (0, 70)
        f1 = host_frame("b", 300, 400)      # 3:4  -> 240x320 at (40, 0)
        batch = MuxBatch(frames=[
            MuxFrameMeta(0, 0, 0, 0, 0, 0, 640, 360, f0),
            MuxFrameMeta(1, 1, 1, 0, 0, 0, 300, 400, f1)], pts=0, batch_index=0,
            max_frames_in_batch=2, push_ns=0)
        tfs = canvas.render(batch)
        out = canvas.to_host()
        assert out.shape == (2, 3, H, W)
        assert (tfs[0].dst_x, tfs[0].dst_y, tfs[0].dst_w, tfs[0].dst_h) == (0, 70, 320, 180)
        assert (tfs[1].dst_x, tfs[1].dst_y, tfs[1].dst_w, tfs[1].dst_h) == (40, 0, 240, 320)
        # padding is black
        assert np.all(out[0, :, :70, :] == 0) and np.all(out[0, :, 250:, :] == 0)
        assert np.all(out[1, :, :, :40] == 0) and np.all(out[1, :, :, 280:] == 0)
        # content equals the single-frame bridge kernel (same bilinear sampler)
        data, planes = synth_nv12(640, 360)
        ref = _core.nv12_host_to_rgb(data, [(p.offset, p.pitch) for p in planes],
                                     (0, 0, 640, 360), (320, 180), False, True, 0)
        np.testing.assert_allclose(out[0, :, 70:250, :], ref, atol=1e-5)
        # bright square (Y=235, V=128 -> R == Y) lands top-left of the content
        # region, scaled 0.5x -> 8px; the x-ramp right of it is still dark
        assert out[0, 0, 70:78, :8].min() > 0.9
        assert out[0, 0, 70:78, 8:16].max() < 0.5
        # unproject maps canvas content corners back to the source frame
        assert tfs[0].unproject((0, 70, 320, 250)) == (0.0, 0.0, 640.0, 360.0)
        assert tfs[1].unproject((40, 0, 280, 320)) == (0.0, 0.0, 300.0, 400.0)
    finally:
        canvas.close()


def test_canvas_partial_batch_zeroes_unused_slots_and_stretch_mode():
    canvas = BatchCanvas(batch_size=3, width=256, height=128, enable_padding=False)
    try:
        canvas._core.device_memset(canvas.ptr, canvas.nbytes, 0x7f)   # dirty
        f0 = host_frame("a", 640, 360)
        batch = MuxBatch(frames=[MuxFrameMeta(0, 0, 0, 0, 0, 0, 640, 360, f0)],
                         pts=0, batch_index=0, max_frames_in_batch=3, push_ns=0)
        tfs = canvas.render(batch)
        out = canvas.to_host()
        assert (tfs[0].dst_w, tfs[0].dst_h) == (256, 128)        # anisotropic stretch
        assert np.all(out[1:] == 0)
        assert out[0].max() > 0.9
        assert abs(tfs[0].scale_x - 2.5) < 1e-9 and abs(tfs[0].scale_y - 2.8125) < 1e-9
    finally:
        canvas.close()


def test_canvas_nearest_interpolation_has_no_intermediate_values():
    canvas = BatchCanvas(batch_size=1, width=64, height=64, enable_padding=True,
                         interpolation="nearest")
    try:
        f0 = host_frame("a", 640, 360)
        # replace the ramp by a hard 2-level checker to make blur measurable
        y = np.zeros((360, 640), np.uint8)
        y[:] = np.where(((np.arange(640)[None, :] // 40) + (np.arange(360)[:, None] // 40)) % 2, 235, 16)
        uv = np.full((180, 640), 128, np.uint8)
        f0.host_data = y.tobytes() + uv.tobytes()
        f0.planes = (PlaneLayout(0, 640), PlaneLayout(640 * 360, 640))
        batch = MuxBatch(frames=[MuxFrameMeta(0, 0, 0, 0, 0, 0, 640, 360, f0)],
                         pts=0, batch_index=0, max_frames_in_batch=1, push_ns=0)
        canvas.render(batch)
        out = canvas.to_host()[0, 0]
        content = out[canvas.height // 2 - 10: canvas.height // 2 + 10]
        vals = np.unique(np.round(content, 2))
        assert len(vals) <= 3, vals                              # black/white only
    finally:
        canvas.close()
