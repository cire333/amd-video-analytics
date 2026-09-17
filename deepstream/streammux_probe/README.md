# nvstreammux behavioral probe

Tooling used to reverse-engineer `nvstreammux` (DeepStream 7.1) on the RTX 3090 for the
AMD port (`avap.streammux`). See `docs/nvstreammux_reverse_engineering.md` for findings.

```
docker build -t ds-probe:7.1 .          # DS 7.1 samples image + gst dev headers + pyds + OpenCV
python3 make_pattern.py .               # synthetic test patterns (ramps + checker + corner marks)
./battery.sh                            # 17 timing scenarios + 11 RGBA geometry dumps -> out/
./rerun.sh; ./geom2.sh                  # clean re-runs, explicit-timeout tests, NV12-path geometry
python3 analyze.py out/*.jsonl -v       # per-trace summary
```

`mux_probe.py` builds N sources (file / live videotestsrc / PNG) -> nvstreammux -> fakesink and
records every input buffer per sink pad, every output batch with its full NvDsFrameMeta, and all
downstream custom events, as JSONL. `--dump` saves output surfaces as .npy (RGBA path via
`--rgba`, NV12 path via `--post-rgba`). Raw traces from the 2026-09-15 runs are in
`results/streammux_probe/` (gitignored).
