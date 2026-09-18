"""Benchmark: VCN hardware encode vs the software paths.

    sg render -c "PYTHONPATH=/opt/rocm/lib:src .venv/bin/python scripts/bench_encode.py [w] [h] [n]"
"""
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


_POOL: list = []


def frames(n, w, h):
    """Cycled pre-generated frames so generation cost stays out of timings."""
    if not _POOL:
        y, x = np.mgrid[0:h, 0:w]
        for i in range(20):
            r = ((x + 3 * i) % 256).astype(np.uint8)
            g = ((y + 2 * i) % 256).astype(np.uint8)
            _POOL.append(np.ascontiguousarray(
                np.dstack([r, g, np.full((h, w), 64, np.uint8)])))
    for i in range(n):
        yield _POOL[i % len(_POOL)]


def main():
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 1920
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 1080
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    d = Path("/tmp/claude-1000") if Path("/tmp/claude-1000").exists() else Path("/tmp")
    results = {}

    # cv2 mp4v (what the batch scripts used)
    vw = cv2.VideoWriter(str(d / "b_mp4v.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    t0 = time.perf_counter()
    for fr in frames(n, w, h):
        vw.write(fr[:, :, ::-1])
    vw.release()
    results["cv2 mp4v (SW)"] = time.perf_counter() - t0

    # libx264 via ffmpeg pipe (the re-encode-quality path)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         str(d / "b_x264.mp4")], stdin=subprocess.PIPE)
    t0 = time.perf_counter()
    for fr in frames(n, w, h):
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    results["libx264 veryfast (SW)"] = time.perf_counter() - t0

    from avap import VideoEncoder
    enc = VideoEncoder(str(d / "b_vcn.mp4"), w, h, fps=25)
    t0 = time.perf_counter()
    for fr in frames(n, w, h):
        enc.write(fr)
    enc.close()
    results["VCN h264 (host write)"] = time.perf_counter() - t0

    # device path as production sees it: the DeviceFrame already exists
    # (decode produces it); pre-upload the pool once, time only write_device
    from avap.advanced.device import DeviceFrame
    enc = VideoEncoder(str(d / "b_vcn_dev.mp4"), w, h, fps=25, bt709=False)
    pool = [DeviceFrame.from_host(fr, enc.device_ordinal)
            for fr in frames(20, w, h)]
    t0 = time.perf_counter()
    for i in range(n):
        enc.write_device(pool[i % len(pool)])
    enc.close()
    results["VCN h264 (device frame write)"] = time.perf_counter() - t0
    for dev in pool:
        dev.close()

    print(f"encode {n} frames @ {w}x{h}:")
    for name, secs in results.items():
        print(f"  {name:38s} {secs:6.2f}s  ({n / secs:6.1f} fps)")


if __name__ == "__main__":
    main()
