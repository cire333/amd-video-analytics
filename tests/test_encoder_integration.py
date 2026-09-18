"""Encoder integration: draw_objects (unit) + AMDStream annotated_output (GPU).

GPU part:
    sg render -c "PYTHONPATH=/opt/rocm/lib:src AVAP_GPU_E2E=1 \
        .venv/bin/python -m pytest tests/test_encoder_integration.py -v"
"""
import json
import os
import subprocess
import time

import numpy as np
import pytest

from avap.frame import ObjectMeta


def det(track_id=7, label="car"):
    return ObjectMeta(class_id=2, confidence=0.91, bbox=(50, 60, 200, 180),
                      track_id=track_id, label=label)


def test_draw_objects_marks_frame():
    from avap.annotate import draw_objects
    img = np.zeros((360, 640, 3), np.uint8)
    out = draw_objects(img, [det()])
    assert out is img                       # in place
    assert img.sum() > 0                    # something was drawn
    assert img[60, 50:200].sum() > 0        # along the box's top edge


def test_draw_objects_handles_untracked():
    from avap.annotate import draw_objects
    img = np.zeros((100, 100, 3), np.uint8)
    o = det(track_id=None)
    o.bbox = (10, 10, 60, 60)
    draw_objects(img, [o])
    assert img.sum() > 0


@pytest.mark.skipif(os.environ.get("AVAP_GPU_E2E") != "1",
                    reason="GPU: set AVAP_GPU_E2E=1 (run via sg render)")
def test_amdstream_annotated_output(tmp_path):
    from avap import AMDStream

    video = "/home/eric-valasek/Downloads/load-to-server/1933_A22.mp4"
    out_mp4 = str(tmp_path / "annot.mp4")
    s = AMDStream(
        data_location=video, model="yolo26m", model_quant="fp16",
        tracker_type="ocsort",
        output_location=str(tmp_path / "dets.jsonl"),
        annotated_output=out_mp4, annotated_fps=25,
        source_id="annot-test")
    s.start_stream()
    deadline = time.time() + 240
    while s.state == "running" and time.time() < deadline:
        time.sleep(0.5)
    s.join(timeout=30)
    s.stop_stream()
    assert s.state == "eof"
    assert s.frames_processed == 89

    st = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,nb_read_frames,width",
         "-of", "json", out_mp4], capture_output=True, text=True).stdout
    )["streams"][0]
    assert st["codec_name"] == "h264"
    assert int(st["nb_read_frames"]) == 89   # every processed frame, incl. last
    assert int(st["width"]) == 1920
    # detections were produced too (both outputs from one pass)
    lines = open(tmp_path / "dets.jsonl").read().splitlines()
    assert len(lines) == 89
