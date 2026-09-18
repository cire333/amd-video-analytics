"""NvDCF-class tracker: DCF engine math (numpy reference), HIP parity, lifecycle."""
import numpy as np
import pytest

from avap.frame import ObjectMeta
from avap.nvdcf import (NumpyDcfEngine, NvDcfConfig, NvDcfTracker, VisualFrame)


def synth_frame(w=320, h=240, seed=0):
    """Gray background with a textured 40x30 patch (object) at a known place."""
    rng = np.random.default_rng(seed)
    img = np.full((3, h, w), 0.45, np.float32)
    tex = rng.random((3, 30, 40)).astype(np.float32)
    return img, tex


def place(img, tex, x, y):
    out = img.copy()
    h, w = tex.shape[1:]
    out[:, y:y + h, x:x + w] = tex
    return out


def host_frame(chw):
    return VisualFrame(ptr=0, width=chw.shape[2], height=chw.shape[1], host=chw)


def small_cfg(**kw):
    base = dict(maxTargetsPerStream=8, featureImgSizeLevel=2, useHog=0, gaussianSigma=2.0,
                probationAge=2, maxShadowTrackingAge=10, earlyTerminationAge=2,
                minTrackerConfidence=0.15, checkClassMatch=1)
    base.update(kw)
    return NvDcfConfig(**base)


# ------------------------------------------------------------------ engine math

def test_numpy_engine_localizes_translation():
    img, tex = synth_frame()
    f0 = host_frame(place(img, tex, 100, 80))
    eng = NumpyDcfEngine(max_targets=2, feature_size=32, use_hog=False, gaussian_sigma=2.0)
    win = np.array([[120.0, 95.0, 100.0, 75.0]], np.float32)     # centre of the patch, 2.5x
    eng.update(f0, win, np.array([0]), np.array([1], np.uint8), lr=0.1)
    # same place -> peak ~1 at zero displacement
    r0 = eng.localize(f0, win, np.array([0]))[0]
    assert np.unravel_index(r0.argmax(), r0.shape) == (0, 0)
    assert 0.8 < r0.max() <= 1.05
    # object moves +12 px right, +6 px down; window stays -> peak at that displacement
    f1 = host_frame(place(img, tex, 112, 86))
    r1 = eng.localize(f1, win, np.array([0]))[0]
    iy, ix = np.unravel_index(r1.argmax(), r1.shape)
    S = 32
    dx = (ix if ix <= S // 2 else ix - S) * 100.0 / S
    dy = (iy if iy <= S // 2 else iy - S) * 75.0 / S
    assert abs(dx - 12) <= 3.2 and abs(dy - 6) <= 2.4          # within one feature pixel
    # a different texture correlates much worse
    other = np.random.default_rng(7).random((3, 30, 40)).astype(np.float32)
    r2 = eng.localize(host_frame(place(img, other, 100, 80)), win, np.array([0]))[0]
    assert r2.max() < 0.6 * r0.max()


def test_hip_engine_matches_numpy_reference():
    pytest.importorskip("avap._core")
    from avap import _core
    if _core.hip_device_count() == 0:
        pytest.skip("no HIP device")
    from avap.nvdcf import FrameUploader, HipDcfEngine
    img, tex = synth_frame()
    chw = place(img, tex, 100, 80)
    up = FrameUploader()
    try:
        fr = up.upload(chw, keep_host=True)
        kw = dict(max_targets=4, feature_size=32, use_hog=True, gaussian_sigma=2.0,
                  focus_offset_y=-0.1)
        ref, gpu = NumpyDcfEngine(**kw), HipDcfEngine(**kw)
        win = np.array([[120.0, 95.0, 100.0, 75.0], [60.0, 40.0, 50.0, 50.0]], np.float32)
        fr_np = VisualFrame(0, fr.width, fr.height, host=chw)
        f_ref, f_gpu = ref.extract_features(fr_np, win), gpu.extract_features(fr, win)
        assert f_gpu.shape == f_ref.shape == (2, 21, 32, 32)
        np.testing.assert_allclose(f_gpu, f_ref, atol=2e-3, rtol=1e-3)
        slots, init = np.array([0, 1]), np.array([1, 1], np.uint8)
        ref.update(fr_np, win, slots, init, 0.1); gpu.update(fr, win, slots, init, 0.1)
        moved = place(img, tex, 112, 86)
        fr2 = up.upload(moved, keep_host=True)
        r_ref = ref.localize(VisualFrame(0, fr2.width, fr2.height, host=moved), win, slots)
        r_gpu = gpu.localize(fr2, win, slots)
        np.testing.assert_allclose(r_gpu, r_ref, atol=2e-2, rtol=2e-2)
        assert np.unravel_index(r_gpu[0].argmax(), (32, 32)) == np.unravel_index(r_ref[0].argmax(), (32, 32))
    finally:
        up.close()


# ------------------------------------------------------------------ tracker lifecycle

def run_sequence(tracker, frames, dets_per_frame):
    outs = []
    for chw, dets in zip(frames, dets_per_frame):
        objs = [ObjectMeta(class_id=2, confidence=0.9, bbox=b, label="car") for b in dets]
        outs.append(tracker.update(objs, host_frame(chw) if chw is not None else None))
    return outs


def moving_scene(n=20, gap=range(8, 13), step=4):
    img, tex = synth_frame()
    frames, dets = [], []
    for i in range(n):
        x, y = 40 + step * i, 60 + i
        frames.append(place(img, tex, x, y))
        dets.append([] if i in gap else [(x, y, x + 40, y + 30)])
    return frames, dets


def test_dcf_bridges_detector_gap_and_keeps_id():
    frames, dets = moving_scene()
    trk = NvDcfTracker(small_cfg(), engine=NumpyDcfEngine(8, 32, use_hog=False, gaussian_sigma=2.0))
    outs = run_sequence(trk, frames, dets)
    ids = [{o.track_id for o in out if o.track_id is not None} for out in outs]
    confirmed = [s for s in ids if s]
    assert confirmed and all(s == confirmed[0] for s in confirmed), ids   # one id throughout
    bridged = [o for out in outs[8:13] for o in out if o.tracked_by == "dcf"]
    assert len(bridged) == 5, "DCF must emit a box on every frame the detector missed"
    # the DCF-localized boxes follow the object (within a few px)
    for i, out in enumerate(outs[8:13], start=8):
        b = out[0].bbox
        assert abs((b[0] + b[2]) / 2 - (40 + 4 * i + 20)) < 6
        assert abs((b[1] + b[3]) / 2 - (60 + i + 15)) < 6
    assert trk.stats["visual_bridged"] == 5 and trk.stats["created"] == 1


def test_without_frames_gap_goes_to_shadow_and_recovers():
    frames, dets = moving_scene()
    trk = NvDcfTracker(small_cfg(), engine=NumpyDcfEngine(8, 32, use_hog=False))
    outs = run_sequence(trk, [None] * len(frames), dets)
    assert all(len(out) == 0 for out in outs[8:13])              # shadow: nothing emitted
    ids = {o.track_id for out in outs for o in out if o.track_id is not None}
    assert len(ids) == 1                                          # same id after the gap


def test_tentative_targets_terminate_early_and_probation():
    cfg = small_cfg(probationAge=3, earlyTerminationAge=1)
    trk = NvDcfTracker(cfg, engine=NumpyDcfEngine(8, 32, use_hog=False))
    # one spurious detection, then nothing
    out = trk.update([ObjectMeta(2, 0.9, (10, 10, 30, 30), label="car")])
    assert out[0].track_id is None and len(trk.targets) == 1
    trk.update([]); trk.update([])
    assert len(trk.targets) == 0                                  # early termination
    # steady detections: confirmed only after probationAge frames
    seen = []
    for i in range(5):
        out = trk.update([ObjectMeta(2, 0.9, (10 + i, 10, 30 + i, 30), label="car")])
        seen.append(out[0].track_id)
    assert seen[:3] == [None, None, None] and seen[3] is not None and seen[4] == seen[3]


def test_visual_similarity_prefers_matching_appearance():
    img, tex = synth_frame()
    other = np.random.default_rng(3).random((3, 30, 40)).astype(np.float32)
    f0 = place(place(img, tex, 60, 80), other, 200, 80)
    trk = NvDcfTracker(small_cfg(), engine=NumpyDcfEngine(8, 32, use_hog=False, gaussian_sigma=2.0))
    for i in range(4):
        fr = place(place(img, tex, 60 + 2 * i, 80), other, 200 - 2 * i, 80)
        trk.update([ObjectMeta(2, 0.9, (60 + 2 * i, 80, 100 + 2 * i, 110), label="car"),
                    ObjectMeta(2, 0.9, (200 - 2 * i, 80, 240 - 2 * i, 110), label="car")], host_frame(fr))
    t_a, t_b = trk.targets
    trk._localize_all(host_frame(f0))
    # each target's response is high at its own box and low at the other one
    assert trk._response_at(t_a, (60, 80, 100, 110)) > 0.8
    assert trk._response_at(t_b, (200, 80, 240, 110)) > 0.8
    assert trk._response_at(t_a, (200, 80, 240, 110)) == 0.0       # outside search window
    # inside the window but on the wrong texture: clearly lower than the true peak
    fr_swap = place(place(img, other, 60, 80), tex, 200, 80)
    trk._localize_all(host_frame(fr_swap))
    assert trk._response_at(t_a, (60, 80, 100, 110)) < 0.7 or t_a.tracker_conf < 0.5


def test_config_from_deepstream_yaml(tmp_path):
    y = tmp_path / "dcf.yml"
    y.write_text("%YAML:1.0\nBaseConfig:\n  minDetectorConfidence: 0.2\nTargetManagement:\n"
                 "  maxShadowTrackingAge: 30\nVisualTracker:\n  featureImgSizeLevel: 4\n"
                 "  useHog: 0\nReID:\n  reidType: 0\n")
    cfg = NvDcfConfig.from_yaml(str(y))
    assert cfg.minDetectorConfidence == 0.2 and cfg.maxShadowTrackingAge == 30
    assert cfg.feature_size == 48 and cfg.useHog == 0
