"""Process a video file through the AMD pipeline with a YOLO26 ONNX model.

    PYTHONPATH=/opt/rocm/lib .venv/bin/python scripts/run_video_yolo.py \
        <video> <model.onnx> <out_dir> [conf_threshold]

Every frame is processed (sequential file mode, not the real-time batcher):
VCN decode -> bridge (fused NV12->RGB 640x640 stretch) -> MIGraphX -> IoU
tracker. Writes per-frame detections to detections.jsonl and an annotated
video with track ids to annotated.mp4.

YOLO26 is end-to-end (NMS-free): output (1, 300, 6) = x1,y1,x2,y2,score,cls
in model-input pixel coordinates.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

from avap import _core
from avap.capabilities import probe_devices, require_decode_device
from avap.frame import ObjectMeta
from avap.kalman_tracker import SortTracker

INPUT_SIZE = 640

COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class MigraphxYolo:
    def __init__(self, onnx_path: str):
        import migraphx
        self._mgx = migraphx
        print(f"parsing + compiling {onnx_path} for gpu (first run takes a bit)...")
        t0 = time.time()
        self.prog = migraphx.parse_onnx(onnx_path)
        self.input_name = self.prog.get_parameter_names()[0]
        self.prog.compile(migraphx.get_target("gpu"))
        print(f"compiled in {time.time() - t0:.1f}s")

    def __call__(self, chw: np.ndarray) -> np.ndarray:
        arg = self._mgx.argument(np.ascontiguousarray(chw[None]))
        out = self.prog.run({self.input_name: arg})
        return np.array(out[0])  # (1, 300, 6)


def nv12_host_to_bgr(frame) -> np.ndarray:
    """Full-res BGR for annotation from the driver-detiled host NV12."""
    y_off, y_pitch = frame.planes[0].offset, frame.planes[0].pitch
    uv_off, uv_pitch = frame.planes[1].offset, frame.planes[1].pitch
    assert y_pitch == uv_pitch, "cv2 two-plane path assumes equal pitches"
    buf = np.frombuffer(frame.host_data, dtype=np.uint8)
    y = buf[y_off:y_off + y_pitch * frame.height].reshape(frame.height, y_pitch)
    uv = buf[uv_off:uv_off + uv_pitch * (frame.height // 2)].reshape(
        frame.height // 2, uv_pitch)
    bgr = cv2.cvtColorTwoPlane(y, uv.reshape(frame.height // 2, uv_pitch // 2, 2),
                               cv2.COLOR_YUV2BGR_NV12)
    return bgr[frame.crop.y:frame.crop.y + frame.crop.height,
               frame.crop.x:frame.crop.x + frame.crop.width]


def color_for(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def main():
    video, model_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    conf_thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
    out_fps = float(sys.argv[5]) if len(sys.argv) > 5 else 25.0
    os.makedirs(out_dir, exist_ok=True)

    dev = require_decode_device(probe_devices())
    print(f"device: {dev.drm_render_node} {dev.gcn_arch}")
    model = MigraphxYolo(model_path)
    tracker = SortTracker(iou_threshold=0.3, max_age=15, min_hits=3)

    dec = _core.Decoder(video, dev.drm_render_node)
    writer = None
    n_frames = 0
    t_start = time.time()

    with open(f"{out_dir}/detections.jsonl", "w") as jf:
        while True:
            f = dec.next_frame()
            if f is None:
                break
            planes = [tuple(p) for p in f.planes]
            src = (f.crop_x, f.crop_y, f.crop_w, f.crop_h)
            common = (planes, src, (INPUT_SIZE, INPUT_SIZE),
                      f.full_range, f.color_matrix != "bt601", dev.device_ordinal)
            if f.dmabuf_fd >= 0:
                chw = _core.nv12_dmabuf_to_rgb(f.dmabuf_fd, f.width, f.height,
                                               common[0], f.drm_modifier, *common[1:])
            else:
                chw = _core.nv12_host_to_rgb(f.host_data, *common)

            dets_raw = model(chw)[0]  # (300, 6)
            sx, sy = f.crop_w / INPUT_SIZE, f.crop_h / INPUT_SIZE
            detections = [
                ObjectMeta(class_id=int(c), confidence=float(s),
                           bbox=(float(x1) * sx, float(y1) * sy,
                                 float(x2) * sx, float(y2) * sy),
                           label=COCO[int(c)] if int(c) < len(COCO) else str(int(c)))
                for x1, y1, x2, y2, s, c in dets_raw if s >= conf_thr
            ]
            tracked = tracker.update(detections)

            jf.write(json.dumps({
                "frame": n_frames, "pts_us": f.pts_us,
                "objects": [
                    {"track_id": o.track_id, "class_id": o.class_id,
                     "label": o.label, "conf": round(o.confidence, 4),
                     "bbox": [round(v, 1) for v in o.bbox]}
                    for o in tracked
                ],
            }) + "\n")

            # annotate on full-res frame
            class DummyF:  # noqa: N801 - adapter so nv12_host_to_bgr reads dataclass-ish fields
                pass
            if f.host_data is not None:
                fd = DummyF()
                fd.planes = [type("P", (), {"offset": p[0], "pitch": p[1]})()
                             for p in planes]
                fd.host_data, fd.height = f.host_data, f.height
                fd.crop = type("C", (), {"x": f.crop_x, "y": f.crop_y,
                                         "width": f.crop_w, "height": f.crop_h})()
                bgr = nv12_host_to_bgr(fd)
            else:  # dmabuf path: reconvert at full res via the kernel
                full = _core.nv12_dmabuf_to_rgb(  # note: fd already consumed; guard
                    f.dmabuf_fd, f.width, f.height, planes, f.drm_modifier,
                    src, (f.crop_w, f.crop_h), f.full_range,
                    f.color_matrix != "bt601", dev.device_ordinal)
                bgr = (np.clip(full, 0, 1) * 255).astype(np.uint8
                        ).transpose(1, 2, 0)[:, :, ::-1].copy()

            for o in tracked:
                x1, y1, x2, y2 = (int(v) for v in o.bbox)
                col = color_for(o.track_id or 0)
                cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 2)
                cv2.putText(bgr, f"#{o.track_id} {o.label} {o.confidence:.2f}",
                            (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, col, 2)
            if writer is None:
                writer = cv2.VideoWriter(f"{out_dir}/annotated_raw.mp4",
                                         cv2.VideoWriter_fourcc(*"mp4v"), out_fps,
                                         (bgr.shape[1], bgr.shape[0]))
            writer.write(bgr)
            n_frames += 1
            if n_frames % 25 == 0:
                print(f"  {n_frames} frames...")

    if writer:
        writer.release()
    dec.close()
    dt = time.time() - t_start
    print(f"processed {n_frames} frames in {dt:.1f}s ({n_frames / dt:.1f} fps)")
    # re-encode for broad playback compatibility
    os.system(f"ffmpeg -y -loglevel error -i {out_dir}/annotated_raw.mp4 "
              f"-c:v libx264 -pix_fmt yuv420p {out_dir}/annotated.mp4 "
              f"&& rm {out_dir}/annotated_raw.mp4")
    print(f"outputs: {out_dir}/detections.jsonl, {out_dir}/annotated.mp4")


if __name__ == "__main__":
    main()
