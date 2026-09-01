"""AMDStream / AMDGPUManager usage — the library-facing API.

    sg render -c "PYTHONPATH=/opt/rocm/lib .venv/bin/python examples/streaming_api.py"
"""
import logging
import time

from avap import AMDGPUManager, AMDStream

logging.basicConfig(level=logging.INFO)

stream1 = AMDStream(
    data_location="rtsp://camera-1.local/stream",   # or /path/file.mp4, s3://, https://
    region_of_interest=[(0.05, 0.30), (0.95, 0.30), (0.95, 0.95), (0.05, 0.95)],
    model="yolo26m",              # zoo: yolo26n/s/m/l/x — or a path to your .onnx
    model_quant="fp16",           # fp32 | fp16 | int8 (int8 self-calibrates on the stream)
    tracker_type="bytetrack",     # iou | sort | bytetrack | your TrackerProtocol object
    output_location="kafka://broker:9092/detections",
    batch_size=1,                 # 1 = realtime; >1 compiles a batched model
    output_format="json",
    output_format_template=None,  # e.g. "{source_id},{pts_us},{n_objects}"
    frame_sample_rate=10,         # process at most 10 fps from the source
    source_id="intersection-7-north",
)

stream2 = AMDStream(
    data_location="s3://gm-videos/curbside/1933_A22.mp4",
    region_of_interest=(0.0, 0.25, 0.75, 0.9),      # bbox form also accepted
    model="yolo26s",
    model_quant="int8",
    tracker_type="sort",
    output_location="s3://gm-results/curbside/1933_A22.parquet",
    batch_size=8,                 # offline file: batch for throughput
    output_format="parquet",
)

manager = AMDGPUManager(device_id=0)
manager.add_stream(stream1)
manager.add_stream(stream2)
manager.start_streams()   # sequential; a failed start is logged, running
                          # streams are unaffected, later ones stay pending
try:
    while True:
        time.sleep(10)
        print(manager.status())
except KeyboardInterrupt:
    manager.stop_streams()
