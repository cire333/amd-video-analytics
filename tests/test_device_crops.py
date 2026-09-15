"""GPU tests for the device-resident SGIE crop path.

    sg render -c "PYTHONPATH=/opt/rocm/lib:src AVAP_GPU_E2E=1 \
        .venv/bin/python -m pytest tests/test_device_crops.py -v"
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("AVAP_GPU_E2E") != "1",
    reason="GPU test: set AVAP_GPU_E2E=1 on a ROCm box (run via sg render)")


def rand_image(w=1280, h=720, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3),
                                                dtype=np.uint8)


def test_device_frame_roundtrip():
    from avap.advanced.device import DeviceFrame
    img = rand_image()
    with DeviceFrame.from_host(img, 0) as dev:
        back = dev.to_host()
    # float32 quantization through /255 and *255: at most 1 count off
    assert int(np.abs(back.astype(int) - img.astype(int)).max()) <= 1


def test_crop_kernel_matches_cv2_resize():
    import cv2
    from avap import _core
    from avap.advanced.device import DeviceFrame
    import migraphx

    img = rand_image(seed=3)
    rect = (200, 100, 400, 300)   # x, y, w, h
    dst_w = dst_h = 224

    with DeviceFrame.from_host(img, 0) as dev:
        out_arg = migraphx.allocate_gpu(
            migraphx.shape(lens=[1, 3, dst_h, dst_w], type="float_type"))
        _core.rgb_crop_resize_device(dev.handle, dev.width, dev.height,
                                     rect, out_arg.data_ptr(), dst_w, dst_h, 0)
        kernel = np.array(migraphx.from_gpu(out_arg)).reshape(3, dst_h, dst_w)

    crop = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    ref = cv2.resize(crop, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR
                     ).astype(np.float32).transpose(2, 0, 1) / 255.0
    # both bilinear; implementations may differ by a few counts at edges
    diff = np.abs(kernel - ref)
    assert diff.mean() < 2 / 255
    assert np.quantile(diff, 0.99) < 8 / 255


def test_device_and_host_cascades_agree(tmp_path):
    """Same video + models through the pipeline with device_resident on and
    off: identical plate reads, near-identical embeddings."""
    from test_advanced_gpu_e2e import VIDEO, export_models
    from avap.advanced import (AdvancedPipeline, ClassifierStage,
                               DetectionStage, EmbeddingStage, InferConfig,
                               PrimaryStage)

    paths = export_models(tmp_path)

    def build(device_resident):
        return AdvancedPipeline(
            primary=PrimaryStage(InferConfig(model=paths["primary"],
                                             component_id=1,
                                             labels=["p", "b", "car"],
                                             conf_threshold=0.3)),
            stages=[
                DetectionStage(InferConfig(model=paths["plate"], component_id=2,
                                           operate_on_component=1,
                                           operate_on_class_ids=frozenset({2}),
                                           labels=["plate"],
                                           conf_threshold=0.3)),
                ClassifierStage(InferConfig(model=paths["ocr"], component_id=3,
                                            operate_on_component=2,
                                            labels=["A", "GMX777", "B", "C"],
                                            min_object_width=4,
                                            min_object_height=4)),
                EmbeddingStage(InferConfig(model=paths["embed"], component_id=4,
                                           operate_on_component=1,
                                           operate_on_class_ids=frozenset({2}),
                                           normalize_output=True)),
            ],
            tracker="sort", device_resident=device_resident)

    results = {}
    for mode in (True, False):
        pipe = build(mode)
        idx = pipe.add_source(VIDEO, camera_id="cmp")
        pipe.prepare()
        assert pipe.device_resident is mode
        frames = []
        from avap import _core
        dec = _core.Decoder(VIDEO, pipe._device.drm_render_node)
        for _ in range(10):
            f = dec.next_frame()
            if mode:
                from avap.advanced.device import DeviceFrame
                with DeviceFrame.from_decoded(f, pipe._device_ordinal) as dev:
                    frames.append(pipe.process_image(idx, dev.to_host(),
                                                     f.pts_us, dev=dev))
            else:
                rgb = pipe._frame_to_rgb(f)
                frames.append(pipe.process_image(idx, rgb, f.pts_us))
        dec.close()
        results[mode] = frames

    for fd, fh in zip(results[True], results[False]):
        vd = [o for o in fd.primary_objects()]
        vh = [o for o in fh.primary_objects()]
        assert len(vd) == len(vh) == 1
        # identical plate OCR reads through both paths
        pd = vd[0].best_child_classification(2, 3)
        ph = vh[0].best_child_classification(2, 3)
        assert pd.label == ph.label == "GMX777"
        # embeddings computed from kernel-resized vs cv2-resized crops:
        # near-identical direction
        ed, eh = vd[0].tensors[4], vh[0].tensors[4]
        cos = float(ed @ eh)
        assert cos > 0.999, f"embedding cosine {cos}"
