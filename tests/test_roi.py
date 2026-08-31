import numpy as np
import pytest

from avap.roi import RoiConfig, RoiTransform


def test_unproject_roundtrip():
    # ROI crop at (100, 200), model input downscaled 2x from the crop
    t = RoiTransform(offset_x=100, offset_y=200, scale_x=2.0, scale_y=2.0)
    assert t.unproject((10, 20, 30, 40)) == (120, 240, 160, 280)


def test_crop_rect_even_aligned_and_normalized():
    roi = RoiConfig([(0.25, 0.25), (0.75, 0.25), (0.5, 0.75)])
    x, y, w, h = roi.crop_rect_px(1919, 1081)  # odd frame dims
    assert x % 2 == 0 and y % 2 == 0 and w % 2 == 0 and h % 2 == 0
    assert x + w <= 1919 + 1 and y + h <= 1081 + 1
    assert w > 0 and h > 0


def test_mask_covers_polygon_interior():
    # unit square ROI covering the middle of the frame
    roi = RoiConfig([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
    mask = roi.mask(100, 100)
    cx, cy, cw, ch = roi.crop_rect_px(100, 100)
    assert mask.shape == (ch, cw)
    assert mask[ch // 2, cw // 2] == 1        # center inside
    assert mask.mean() > 0.8                  # rect ROI ~ fills its bounding rect


def test_mask_cached_per_resolution():
    roi = RoiConfig([(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])
    m1 = roi.mask(640, 480)
    assert roi.mask(640, 480) is m1           # cache hit
    assert roi.mask(1280, 720) is not m1      # resolution change re-rasterizes


def test_rejects_pixel_coordinates():
    with pytest.raises(ValueError, match="normalized"):
        RoiConfig([(0, 0), (1920, 0), (1920, 1080)])
