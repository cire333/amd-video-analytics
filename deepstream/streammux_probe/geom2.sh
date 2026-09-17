#!/bin/bash
S=$(cd "$(dirname "$0")" && pwd)
R=$S/run.sh
run() { name=$1; shift; echo "=== $name $(date +%T)"; timeout 120 "$@" --out /work/out/$name.jsonl 2>&1 | grep -v "GStreamer-WARNING\|^$\|NGC\|CUDA\|Container\|By pulling\|https://developer\|copy of this\|=====\|GST_PLUGIN_LOADING\|Failed to query" | tail -3; }
G="--post-rgba --dump 1 --num-buffers 3 --batch-size 1 --timeout 40000"
run N1_nv12_1080_to_640sq_pad $R --mux old --source png:/work/pat_1920x1080.png:30 $G --width 640 --height 640 --padding 1 --dump-dir /work/dump/N1
run N2_nv12_1080_to_640sq_pad_nearest $R --mux old --source png:/work/pat_1920x1080.png:30 $G --width 640 --height 640 --padding 1 --interp 0 --dump-dir /work/dump/N2
run N3_nv12_1080_to_640sq_pad_cubic $R --mux old --source png:/work/pat_1920x1080.png:30 $G --width 640 --height 640 --padding 1 --interp 2 --dump-dir /work/dump/N3
run N4_nv12_720_to_1080_nopad $R --mux old --source png:/work/pat_1280x720.png:30 $G --width 1920 --height 1080 --padding 0 --dump-dir /work/dump/N4
run N5_nv12_1080_passthru $R --mux old --source png:/work/pat_1920x1080.png:30 $G --width 1920 --height 1080 --padding 0 --dump-dir /work/dump/N5
run N6_nv12_1080_to_960x544 $R --mux old --source png:/work/pat_1920x1080.png:30 $G --width 960 --height 544 --padding 0 --dump-dir /work/dump/N6
run N7_nv12_1000x700_to_1920x1056_pad $R --mux old --source png:/work/pat_1000x700.png:30 $G --width 1920 --height 1056 --padding 1 --dump-dir /work/dump/N7
# RGBA input but explicit bilinear (1) and Algo-1 cubic (2)
G2="--rgba --dump 1 --num-buffers 3 --batch-size 1 --timeout 40000"
run G12_rgba_1080_to_640sq_pad_bilinear $R --mux old --source png:/work/pat_1920x1080.png:30 $G2 --width 640 --height 640 --padding 1 --interp 1 --dump-dir /work/dump/G12
run G13_rgba_1080_to_640sq_pad_cubic $R --mux old --source png:/work/pat_1920x1080.png:30 $G2 --width 640 --height 640 --padding 1 --interp 2 --dump-dir /work/dump/G13
echo "=== DONE $(date +%T)"
