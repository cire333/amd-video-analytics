# nvstreammux, reverse-engineered for the AMD port

Date: 2026-09-15. Hardware: RTX 3090 (DeepStream 7.1, `nvcr.io/nvidia/deepstream:7.1-samples-multiarch`)
as the reference; AMD R9700 (gfx1201, ROCm 7.2.4) as the target. Tooling: `deepstream/streammux_probe/`.
Raw traces: `results/streammux_probe/*.jsonl` (gitignored), one line per input buffer / output batch /
downstream event.

DeepStream ships two muxers behind one element name:

| | "old" `nvstreammux` (default) | "new" `nvstreammux` (`USE_NEW_NVSTREAMMUX=yes`) |
|---|---|---|
| Source | closed (`libnvdsgst_multistream.so`) | **open**: `sources/libs/nvstreammux` (batch policy) + `sources/gst-plugins/gst-nvmultistream2` (GStreamer wrapper), MIT |
| Scales frames | yes, to `width`x`height`, optional letterbox | no, surfaces pass through at native size |
| Batching | arrival-order FIFO (non-live) / one-per-source (live) | round-robin over sources, adaptive batch size |
| Timeout semantics | wait `batched-push-timeout` after first buffer | min-fps / max-fps windows since last batch |

The old mux is what `deepstream-app` and the GridMatrix ingestion pipeline use, so its behaviour is
what `avap.streammux` reproduces by default (`policy="legacy"`). The new mux's algorithm is ported
from source as `policy="new"`. Both were verified against traces.

## 1. Metadata contract (both muxers)

Per output batch (`NvDsBatchMeta`): `num_frames_in_batch`, `max_frames_in_batch` (= configured
batch-size, even for partial batches), `frame_meta_list` in slot order.

Per frame (`NvDsFrameMeta`), verified on every trace:

| field | value |
|---|---|
| `pad_index` | the `sink_%u` index the buffer came in on |
| `source_id` | == `pad_index` unless upstream answers the `nvquery_sourceid` query (nvurisrcbin does) |
| `batch_id` | position in the batch (0..n-1), always |
| `frame_num` | per-pad counter. Old mux: counts every buffer **received** (drops leave gaps). New mux: counts buffers **batched** (`SinkPad::update_frame_count` in `copy_batch`), so no gaps |
| `buf_pts` | input buffer PTS, unchanged |
| `ntp_timestamp` | `attach-sys-ts=1` (default): wall clock (`gettimeofday`) at mux input, ns since epoch. `attach-sys-ts=0`: RTCP-SR-derived `ntp_epoch - ntp_frame_ts + pts` when rtspsrc provides it, else **0** |
| `source_frame_width/height` | input surface dims (old mux: pre-scaling dims) |
| `num_surfaces_per_frame` | 1 |

Downstream events (old mux): `nv-stream-segment-<pad>` per pad, `nv-pad-added-<pad>` once per pad, on per-source
EOS a `sink-message "stream-eos"` plus `nv-stream-eos-<pad>`, then a normal EOS only when **all** pads are at
EOS. New mux additionally emits `nv-stream-start-<pad>` and `update-caps-<pad>` (per-pad resolution, since it
does not scale). Adding a pad while running is supported by both (registry-style hot add is native).

## 2. Old mux batching (empirical)

Traces: `S1/R1_old_file4`, `R8_old_file4_bs2`, `R7_old_nolive_live2_bs4`, `S3_old_live4`,
`S3_old_live4_nolive`, `R3_old_live4_t100`, `S4_old_live2_bs4`, `S5_old_live2_t4000`, `S5_old_live2_tinf`,
`S7_old_live1_60_bs4`, `S6_old_sync`.

### live-source=0 (files)

* The batch is a **FIFO over all arrivals**, in arrival order, regardless of pad. The same source can
  occupy several slots (R8: `(0,0)`, `(2,2)`; R1: `(2,2,1,0)`; R7 with 2 sources and batch 4: `(1,0,1,0)`).
  No frames are dropped: throughput is bounded by upstream backpressure.
* Push when `batch-size` slots are filled, or `batched-push-timeout` µs after the batch's first buffer
  (R1 starts with two 2-frame batches at 40 ms while the slow decoders spin up). `-1` waits forever.
* Output PTS = `batch_index * frame_duration`, where `frame_duration` is `1/framerate` from the caps of
  the **first buffer received** (R1: 25 fps pad 3 arrived first → 40 ms steps; S3_nolive: 30 fps → 33.3 ms;
  smoke 25 fps → 40 ms). Not related to the timeout.
* After a source hits EOS the remaining sources simply fill all slots (R1 tail: `(2,926),(0,686),(2,927),(1,679)`).

### live-source=1

* **At most one frame per source per batch.** Batch = the oldest queued frame of each source that has
  one; push when every non-EOS source has contributed or `batched-push-timeout` after the first buffer
  (S3_old_live4, 30/30/15/10 fps, timeout 33 ms → batches of 2/3/4 every 33 ms; R3 timeout 100 ms →
  4-frame batches every 100 ms).
* The mux holds ~1 buffer per pad; extra frames block the upstream chain (S7: 60 fps source, batch 4,
  timeout 100 ms → 1-frame batches at exactly 10 Hz; upstream throttled to 10 fps). With the leaky
  queues nvurisrcbin puts in front, that turns into frame drops. Hence the DS rule of thumb
  `batched-push-timeout ≈ 1e6 / max_source_fps`.
* `batch-size` larger than the number of sources never fills: every batch waits the full timeout
  (S4: 2 sources, batch 4, 33 ms → 2-frame batches at 34 ms cadence).
* Output PTS = pipeline running time at push (S3/S4/S7: `pts_ns ≈ t_wall + ~1 ms`).
* `sync-inputs=1` (S6_old_sync): frames later than running time are dropped, batches become 1-2 frames.

### Scaling (`width`/`height`, `enable-padding`, `interpolation-method`)

Traces `G1..G13` (RGBA input, dumped via unified memory) and `N1..N7` (NV12 input, converted after the mux).
Synthetic patterns (x/y ramps + 8 px checker + corner marks) compared against OpenCV.

* `enable-padding=0`: anisotropic stretch to the full canvas.
* `enable-padding=1`: aspect-preserving **letterbox, centred**, black fill (RGBA fill is `0,0,0,0`).
  `scale = min(W/sw, H/sh)`, `dst_w = floor(sw*scale)`, `dst_h = floor(sh*scale)`, `x0 = (W-dst_w)//2`,
  `y0 = (H-dst_h)//2`. 1280x720→640x640: 640x360 at (0,140). 640x480→1920x1080: 1440x1080 at (240,0).
  1000x700→1920x1056: 1508x1056 at (206,0).
* Interpolation on the GPU path is **not** what the property says: RGBA surfaces with the default
  `Bilinear`(1) come out bit-identical to nearest-neighbour (`cv2.INTER_NEAREST`, mean abs diff 0.0,
  G1/G2/G3/G4/G10/G12). `Algo-1`(2) gives a genuine filter (bit-identical to `INTER_LINEAR`/`INTER_CUBIC`
  at 3x downscale, G13). On NV12 (N1) the default is a smoothing filter (closest to `INTER_AREA`) and
  `Nearest`(0) is nearest (N2). Pixel-exact reproduction of NvBufSurfTransform is therefore not a goal;
  `avap` uses half-pixel-centre bilinear by default and offers nearest.
* Passthrough (source == canvas) is lossless in RGBA (G9/G11); NV12→RGBA round trip costs ~1.3 LSB.

## 3. New mux batching (from source, confirmed by traces)

Core: `NvStreamMux::push_loop` + `BatchPolicy` (`nvstreammux.cpp`, `nvstreammux_batch.cpp`).

* **Per-pad queue** of buffers and events (`SinkPad::queue`, events keep their position; `get_available`
  = buffers only). Backpressure: `add_buffer` blocks when `buffer_count > max(batch_size, ...)`.
* **Batch size**: `adaptive-batching=1` (default) → `(#pads - #pads_at_EOS) * num_surfaces_per_frame`,
  i.e. one slot per live source; otherwise the configured `batch-size`. R1_new: 4-frame batches until the
  25 fps source ends, then 3-frame batches. `max_frames_in_batch` stays at the configured value.
* **Formation** (`form_batch`): iterate priority groups (lowest number first; default priority 0 for
  unlisted sources), within a group iterate sources round-robin starting where the previous batch stopped
  (`last_batch_state`). For each source take `min(available, allowed_repeats, remaining slots)` buffers,
  `allowed_repeats = min(max-same-source-frames, source max-num-frames-per-batch)` (both default 1), further
  capped by `enable-source-rate-control` (`allowed = elapsed_since_last_push * max_fps`). Slot order is
  therefore pad order rotated, e.g. always `(0,1,2,3)` when all sources have data (R1_new, S3_new).
* **Timing** (`update_last_batch_time`): `min_dur = last_push + 1/overall-max-fps`, `max_dur = last_push +
  1/overall-min-fps`. Loop: form batch; if full → copy+push (throttled to `min_dur` only when
  `max-fps-control=1`); else wait on the condvar until `max_dur` or a new buffer; a non-empty batch is pushed
  once `now ≥ max_dur` ("due"); an empty one just re-arms. Defaults: max-fps 120, **min-fps 5 → 200 ms**.
* **`batched-push-timeout` is effectively ignored** in DS 7.1's new mux: the property setter calls
  `set_batch_push_timeout` (min-fps = 1e6/timeout), but the first CAPS event runs `configure_module()`,
  which rebuilds the policy from the config file / defaults and re-applies only `batch-size` and
  `num-surfaces-per-frame`. Verified: `R3_new_live4_t33` (timeout 33333) still produced full 4-frame batches
  at 100 ms, identical to `t200`. Use the config file's `overall-min-fps-n/d` (or set the property after
  the pipeline is PLAYING).
* **Pacing**: with per-pad backpressure and one slot per source, the slowest source paces every batch
  (S3_new: 30/30/15/10 fps → 10 Hz batches; 30 fps sources throttled to 10 fps). No frames are dropped
  unless `sync-inputs=1`.
* `max-same-source-frames>1` with `max-num-frames-per-batch>1` (`S7_new_live1_60_bs4_rep`): one 4-frame
  batch from a single source, then the pipeline stalls (input blocked; DS bug). Not reproduced.
* **`sync-inputs=1`** (`NvTimeSync`): buffer running time = `segment_to_running_time(pts)`. Early if
  `rt > now - min_fps_dur` (mux waits, at most until the earliest-by time), late if
  `rt + max-latency + upstream_latency < now - min_fps_dur` (dropped), else on time. `S6_new_sync`: 30 fps
  source lost 82/132 frames while the 15 fps source lost 3.
* **Output PTS**: without sync-inputs `pts_offset + n * frame_duration` where `pts_offset` is the running
  time when the first buffer arrived and `frame_duration` comes from the first pad's caps framerate
  (S7_new: 16.7 ms steps for 60 fps; S4_new: 223 ms + n·33.3 ms). Equal consecutive PTS get +0.25 ms.
  With sync-inputs: max running time over the frames in the batch.
* **EOS**: per pad `handle_eos` queues the EOS as an event; the pad is idle once its queue drains,
  `num_sources_eos++` shrinks the adaptive batch, `stream-eos` events go downstream, and pipeline EOS is
  sent only when every pad is at EOS and empty (`all_pads_eos`). `frame-num-reset-on-eos` zeroes the pad's
  counter then.
* **NTP** (`gstnvstreammux_ntp.cpp`): system-time mode stamps `gettimeofday` per buffer; RTCP mode uses the
  latest sender report `(ntp_epoch - rtp_frame_ts) + pts` with frame-rate based monotonic correction
  (`frame-duration` property: -1 off, 0 auto from PTS deltas, >0 fixed) and warns when host and source NTP
  differ by >10 s.

## 4. What `avap.streammux` implements

`src/avap/streammux.py` (pure Python, GPU-free, deterministic clock for tests):

* `MuxConfig`: `batch_size`, `batched_push_timeout_us` (-1 = infinite), `live_source`, `policy`
  (`"legacy"` = old mux, `"new"` = ported BatchPolicy), `adaptive_batching`, `overall_min_fps`,
  `overall_max_fps`, `max_fps_control`, `max_same_source_frames`, per-source `SourceConfig(priority,
  max_fps, max_frames_per_batch)`, `sync_inputs`, `max_latency_ns`, `attach_sys_ts`,
  `frame_num_reset_on_eos`, canvas `width/height/enable_padding/interpolation`, `queue_depth`.
* `StreamMux.add_pad / remove_pad / push_frame / push_eos / poll(now_ns)`; `poll` returns `MuxBatch`
  objects (`frames: list[MuxFrameMeta]` with the exact `NvDsFrameMeta` fields above plus the caller's
  payload) and `MuxEvent`s (`pad-added`, `stream-segment`, `stream-eos`, `eos`, `pad-deleted`).
  `next_deadline_ns()` tells a driver how long it may sleep. `StreamMuxRunner` is the threaded driver.
* Differences from DS, deliberate: frames that do not fit are held in a bounded per-pad queue and the
  oldest is dropped when it overflows (`queue_depth`), instead of blocking the decoder thread; the drop is
  counted per pad. Output PTS follows the old-mux rules (running time in live mode, `n*frame_duration`
  otherwise).
* `src/avap/canvas.py` + `cpp/kernels.hip::nv12_to_rgb_canvas`: the scaling stage. Fused NV12→RGB, optional
  ROI crop, letterbox per §2 into slot `b` of a `[N,3,H,W]` float32 device tensor (black fill), bilinear
  or nearest. Returns the per-frame `CanvasTransform` (scale, x0, y0, source_w/h) so detections map back
  to source pixels the way DS rescales from the mux canvas to `image_resolution`.

## 5. Verification

* **Unit tests** (`tests/test_streammux.py`, 21 cases, deterministic clock): every behaviour in §2/§3 above.
* **Trace replay** (`deepstream/streammux_probe/replay.py`): the recorded 3090 input timelines are pushed
  into `avap.streammux` with the same settings and the batches compared with DeepStream's:

  | trace | DS batches | avap | identical source multiset (aligned) |
  |---|---|---|---|
  | S3_old_live4 (30/30/15/10 fps, live, 33 ms) | 240 {2:80,3:118,4:42} | 239 {2:79,3:118,4:42} | 239/239 |
  | R3_old_live4_t100 (live, 100 ms) | 61 | 63 | 57/61 (start-up frames DS dropped) |
  | S4_old_live2_bs4 | 143 x2 | 146 | 142/143 |
  | S5_old_live2_t4000 | 150 x2 | 151 | 147/150 |
  | S7_old_live1_60_bs4 | 39 x1 @ 10 Hz | 42 x1 | 39/39 |
  | R7_old_nolive_live2_bs4 (files, 2 src, bs 4) | 61, (1,0,1,0) x59 | 61, (1,0,1,0) x59 | 61/61 |
  | R1_old_file4 (files, bs 4) | 601 {4:599}, max 2/source | 602 {4:601}, max 2/source | histogram + per-pad totals match; slot order differs because probe timestamps do not resolve sub-ms arrival order |
  | R8_old_file4_bs2 | 402 {2:402} | 403 | same as above |

* **GPU canvas** (`tests/test_canvas_gpu.py`, R9700): letterbox placement equals the measured
  nvstreammux geometry, padding is black, content is bit-identical to the single-frame bridge kernel,
  nearest mode has no intermediate values.
* **End-to-end on the R9700** (`examples/multi_stream_mux.py`, 2026-09-15): 4 file sources
  (1080p30, 720p30, qHD25, 1920x1056@25) → `StreamMux(legacy, bs=4, 40 ms)` → `BatchCanvas` 640x640
  letterbox → MIGraphX yolo26m fp16 (batch 4) → detections in source coordinates.

  | | |
  |---|---|
  | frames decoded / emitted | 4774 / 4774 (0 drops; file sources block like DS) |
  | batches | 1284: {4: 1104, 2: 178 (single-source tail), 1: 2} |
  | throughput | 137.6 frames/s, 37 batches/s |
  | per batch | canvas + D2H 8.4 ms, inference 14.3 ms |
  | events | pad-added x4, stream-segment x4, stream-eos x4, eos |

  Follow-ups: hand the canvas pointer to MIGraphX directly (`offload_copy=False`, as in the
  model-chaining branch) to remove the 8 ms host round-trip; rocDecode/detile so the canvas kernel
  reads dmabufs instead of the host-fallback NV12; wire `MuxConfig` into `AMDGPUManager`.
