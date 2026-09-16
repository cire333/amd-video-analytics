#!/bin/bash
S=$(cd "$(dirname "$0")" && pwd)
R=$S/run.sh
ST=/opt/nvidia/deepstream/deepstream-7.1/samples/streams
run() { name=$1; shift; echo "=== $name $(date +%T)"; timeout 300 "$@" --out /work/out/$name.jsonl 2>&1 | grep -v "GStreamer-WARNING\|^$\|NGC\|CUDA\|Container\|By pulling\|https://developer\|copy of this\|=====\|GST_PLUGIN_LOADING\|Failed to query" | tail -3; }
FILES="--source file:$ST/sample_1080p_h264.mp4 --source file:$ST/sample_720p.mp4 --source file:$ST/sample_qHD.mp4 --source file:/vids/1933_A22.mp4"
LIVE4="--source live:1280:720:30 --source live:1280:720:30 --source live:640:480:15 --source live:1920:1080:10"
run R1_old_file4  $R --mux old $FILES --batch-size 4 --timeout 40000 --width 1920 --height 1080 --max-batches 600
USE_NEW_NVSTREAMMUX=yes run R1_new_file4  $R --mux new $FILES --batch-size 4 --max-batches 600
USE_NEW_NVSTREAMMUX=yes run R3_new_live4_t33 $R --mux new $LIVE4 --batch-size 4 --timeout 33333 --duration 6
USE_NEW_NVSTREAMMUX=yes run R3_new_live4_t200 $R --mux new $LIVE4 --batch-size 4 --timeout 200000 --duration 6
run R3_old_live4_t100 $R --mux old $LIVE4 --batch-size 4 --timeout 100000 --live 1 --width 1280 --height 720 --duration 6
# old mux, non-live, 2 live sources, bs 4 : does it wait for repeats?
run R7_old_nolive_live2_bs4 $R --mux old --source live:1280:720:30 --source live:1280:720:30 --batch-size 4 --timeout 100000 --live 0 --width 1280 --height 720 --duration 4
# old mux, file sources, bs 2 with 4 sources (fewer slots than sources)
run R8_old_file4_bs2 $R --mux old $FILES --batch-size 2 --timeout 40000 --width 1920 --height 1080 --max-batches 400
USE_NEW_NVSTREAMMUX=yes run R8_new_file4_bs2 $R --mux new $FILES --batch-size 2 --config /work/cfg_bs2.txt --max-batches 400
# old mux: attach-sys-ts=0 (ntp from rtsp unavailable -> 0?)
run R9_old_ntp0 $R --mux old --source live:640:480:30 --batch-size 1 --timeout 33333 --live 1 --width 640 --height 480 --attach-sys-ts 0 --duration 2
echo "=== DONE $(date +%T)"
