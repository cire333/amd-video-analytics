#!/bin/bash
S=$(cd "$(dirname "$0")" && pwd)
R=$S/run.sh
ST=/opt/nvidia/deepstream/deepstream-7.1/samples/streams
run() { name=$1; shift; echo "=== $name $(date +%T)"; timeout 300 "$@" --out /work/out/$name.jsonl 2>&1 | grep -v "GStreamer-WARNING\|^$\|NGC\|CUDA\|Container\|By pulling\|https://developer\|copy of this\|=====\|GST_PLUGIN_LOADING\|Failed to query" | tail -4; }
FILES="--source file:$ST/sample_1080p_h264.mp4 --source file:$ST/sample_720p.mp4 --source file:$ST/sample_qHD.mp4 --source file:/vids/1933_A22.mp4"
LIVE4="--source live:1280:720:30 --source live:1280:720:30 --source live:640:480:15 --source live:1920:1080:10"
# --- file scenarios, max throughput (sink sync=0)
run S1_old_file4  $R --mux old $FILES --batch-size 4 --timeout 40000 --width 1920 --height 1080 --max-batches 800
USE_NEW_NVSTREAMMUX=yes run S1_new_file4  $R --mux new $FILES --batch-size 4 --max-batches 800
# --- file scenarios paced real-time (sink sync=1), fps heterogeneity via drop-frame-interval
run S2_old_file_paced $R --mux old --source file:$ST/sample_1080p_h264.mp4 --source file:$ST/sample_720p.mp4:drop=2 --source file:$ST/sample_qHD.mp4:drop=3 --batch-size 3 --timeout 40000 --width 1280 --height 720 --sink-sync 1 --duration 8
USE_NEW_NVSTREAMMUX=yes run S2_new_file_paced $R --mux new --source file:$ST/sample_1080p_h264.mp4 --source file:$ST/sample_720p.mp4:drop=2 --source file:$ST/sample_qHD.mp4:drop=3 --batch-size 3 --sink-sync 1 --duration 8
# --- live sources (videotestsrc is-live), heterogeneous fps
run S3_old_live4 $R --mux old $LIVE4 --batch-size 4 --timeout 33333 --live 1 --width 1280 --height 720 --duration 8
USE_NEW_NVSTREAMMUX=yes run S3_new_live4 $R --mux new $LIVE4 --batch-size 4 --duration 8
USE_NEW_NVSTREAMMUX=yes run S3_new_live4_cfg $R --mux new $LIVE4 --batch-size 4 --config /work/cfg_minfps5.txt --duration 8
# --- fewer sources than batch-size
run S4_old_live2_bs4 $R --mux old --source live:1280:720:30 --source live:1280:720:30 --batch-size 4 --timeout 33333 --live 1 --width 1280 --height 720 --duration 5
USE_NEW_NVSTREAMMUX=yes run S4_new_live2_bs4 $R --mux new --source live:1280:720:30 --source live:1280:720:30 --batch-size 4 --duration 5
# --- short timeout -> partial batches
run S5_old_live2_t4000 $R --mux old --source live:1280:720:30 --source live:1280:720:30 --batch-size 2 --timeout 4000 --live 1 --width 1280 --height 720 --duration 5
run S5_old_live2_tinf $R --mux old --source live:1280:720:30 --source live:1280:720:15 --batch-size 2 --timeout -1 --live 1 --width 1280 --height 720 --duration 5
# --- same-source repeats: 1 source at 60fps, batch 4
run S7_old_live1_60_bs4 $R --mux old --source live:1280:720:60 --batch-size 4 --timeout 100000 --live 1 --width 1280 --height 720 --duration 4
USE_NEW_NVSTREAMMUX=yes run S7_new_live1_60_bs4 $R --mux new --source live:1280:720:60 --batch-size 4 --duration 4
USE_NEW_NVSTREAMMUX=yes run S7_new_live1_60_bs4_rep $R --mux new --source live:1280:720:60 --batch-size 4 --config /work/cfg_repeats.txt --duration 4
# --- sync-inputs
USE_NEW_NVSTREAMMUX=yes run S6_new_sync $R --mux new --source live:1280:720:30 --source live:640:480:15 --batch-size 2 --sync-inputs 1 --duration 5
run S6_old_sync $R --mux old --source live:1280:720:30 --source live:640:480:15 --batch-size 2 --timeout 33333 --live 1 --sync-inputs 1 --width 1280 --height 720 --duration 5
# --- live-source=0 with live sources (common misconfig)
run S3_old_live4_nolive $R --mux old $LIVE4 --batch-size 4 --timeout 33333 --live 0 --width 1280 --height 720 --duration 5
# --- geometry (RGBA dumps)
G="--rgba --dump 1 --num-buffers 3 --batch-size 1 --timeout 40000"
run G1_720_to_1080_nopad $R --mux old --source png:/work/pat_1280x720.png:30 $G --width 1920 --height 1080 --padding 0 --dump-dir /work/dump/G1
run G2_720_to_640sq_nopad $R --mux old --source png:/work/pat_1280x720.png:30 $G --width 640 --height 640 --padding 0 --dump-dir /work/dump/G2
run G3_720_to_640sq_pad $R --mux old --source png:/work/pat_1280x720.png:30 $G --width 640 --height 640 --padding 1 --dump-dir /work/dump/G3
run G4_1080_to_640sq_pad $R --mux old --source png:/work/pat_1920x1080.png:30 $G --width 640 --height 640 --padding 1 --dump-dir /work/dump/G4
run G5_480_to_1080_pad $R --mux old --source png:/work/pat_640x480.png:30 $G --width 1920 --height 1080 --padding 1 --dump-dir /work/dump/G5
run G6_1000x700_to_1920x1056_pad $R --mux old --source png:/work/pat_1000x700.png:30 $G --width 1920 --height 1056 --padding 1 --dump-dir /work/dump/G6
run G7_1080_to_640sq_pad_nearest $R --mux old --source png:/work/pat_1920x1080.png:30 $G --width 640 --height 640 --padding 1 --interp 0 --dump-dir /work/dump/G7
run G8_1080_to_640sq_pad_default $R --mux old --source png:/work/pat_1920x1080.png:30 $G --width 640 --height 640 --padding 1 --interp 6 --dump-dir /work/dump/G8
run G9_1080_passthru $R --mux old --source png:/work/pat_1920x1080.png:30 $G --width 1920 --height 1080 --padding 0 --dump-dir /work/dump/G9
run G10_1080_to_960x544_nopad $R --mux old --source png:/work/pat_1920x1080.png:30 $G --width 960 --height 544 --padding 0 --dump-dir /work/dump/G10
USE_NEW_NVSTREAMMUX=yes run G11_new_passthru $R --mux new --source png:/work/pat_1280x720.png:30 $G --dump-dir /work/dump/G11
echo "=== DONE $(date +%T)"
