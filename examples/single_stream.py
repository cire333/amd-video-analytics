"""V1 milestone: one source end-to-end on the R9700.

    python examples/single_stream.py <video-or-rtsp-uri> <detector.onnx>

Decode (VCN/VAAPI) -> dmabuf->HIP -> NV12->RGB -> ONNX detector -> tracker,
printing tracked objects per frame. The detection parser below assumes a
standard (1, N, 6) [x1, y1, x2, y2, score, class] output — adjust for the
actual model (RT-DETR heads differ).
"""
import logging
import sys
import time

from avap import ModelGraph, ObjectMeta, OnnxModel, Pipeline

CONF_THRESHOLD = 0.4


def parse_detections(outputs) -> list[ObjectMeta]:
    dets = outputs[0][0]  # (N, 6)
    return [
        ObjectMeta(class_id=int(cls), confidence=float(score),
                   bbox=(float(x1), float(y1), float(x2), float(y2)))
        for x1, y1, x2, y2, score, cls in dets
        if score >= CONF_THRESHOLD
    ]


def print_sink(meta):
    objs = ", ".join(
        f"#{o.track_id} cls={o.class_id} ({o.bbox[0]:.0f},{o.bbox[1]:.0f})"
        for o in meta.objects
    )
    print(f"[{meta.source_id} pts={meta.pts}] {len(meta.objects)} objects: {objs}")


def main():
    logging.basicConfig(level=logging.INFO)
    uri, model_path = sys.argv[1], sys.argv[2]

    graph = ModelGraph()
    graph.add_node("detector", OnnxModel(model_path))

    pipe = Pipeline(
        graph=graph,
        detector_node="detector",
        parse_detections=parse_detections,
        sink=print_sink,
        model_input_hw=(640, 640),
    )
    pipe.add_stream("cam0", uri)
    pipe.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipe.stop()


if __name__ == "__main__":
    main()
