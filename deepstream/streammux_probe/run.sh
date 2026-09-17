#!/bin/bash
# usage: run.sh <args to mux_probe.py>   (mounts this dir as /work, results/ds_work as /vids)
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
exec docker run --rm --device nvidia.com/gpu=all \
  -v "$HERE":/work -v "$REPO"/results/ds_work:/vids:ro \
  -e USE_NEW_NVSTREAMMUX=${USE_NEW_NVSTREAMMUX:-no} -e GST_DEBUG=${GST_DEBUG:-1} \
  ds-probe:7.1 python3 /work/mux_probe.py "$@"
